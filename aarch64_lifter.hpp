// UltraTears AArch64 -> LLVM IR Instruction Lifter (tools/lifter_llvm/aarch64_lifter.hpp)
//
// Translates ARMv8.0-A (Nintendo Switch Cortex-A57) instructions directly into SSA LLVM IR
// over the `ultrs::llvm_ir::IRBuilder` and `Function` CFG:
//   - Integer data processing (immediate & register), bitfields, funnel shifts, 128-bit multiplies.
//   - Branches, conditional branches, jump tables, direct & indirect calls.
//   - Horizon OS system instructions (`SVC`, `BRK`, `UDF`, `MRS`, `MSR`, `DC ZVA`, barriers) emitted as
//     semantic `OpTag::Horizon*` IR nodes for `HorizonLoweringPass` to lower.
//   - Loads, stores, pairs (`LDP`/`STP`), and atomics/exclusives (`LDXR`/`STXR`/`LDAR`/`STLR` using native
//     LLVM `load atomic`, `store atomic`, and `cmpxchg`).
//   - Scalar FP, AdvSIMD vector/scalar, Crypto (`AES*`, `SHA1*`, `SHA256*`), and `CRC32`/`CRC32C` lowered to
//     native LLVM vector types (`<16 x i8>`, `<8 x i16>`, `<4 x i32>`, `<2 x i64>`, `<4 x float>`, `<2 x double>`),
//     `shufflevector`, saturating/min/max intrinsics, and `rt_*` helpers for FPCR-dependent paths.
#pragma once

#include "llvm_ir.hpp"
#include "nso_loader.hpp"

namespace ultrs::lifter {

using namespace ultrs::llvm_ir;
using ultrs::nso::bits;
using ultrs::nso::decode_bit_masks;
using ultrs::nso::ones;
using ultrs::nso::ror_val;
using ultrs::nso::sext;

// Lane-op enum values matching src/runtime/simd.h
enum LaneOp : int {
    LOP_SQADD = 0, LOP_UQADD, LOP_SQSUB, LOP_UQSUB, LOP_SUQADD, LOP_USQADD,
    LOP_SHL, LOP_RSHL, LOP_QSHL, LOP_QRSHL,
    LOP_SHLI, LOP_QSHLI, LOP_QSHLUI, LOP_SHRI, LOP_RSHRI, LOP_SRAI, LOP_RSRAI, LOP_SRI, LOP_SLI,
    LOP_SQDMULH, LOP_SQRDMULH, LOP_PMUL,
    LOP_SQABS, LOP_SQNEG, LOP_ABS, LOP_NEG, LOP_CLS, LOP_CLZ, LOP_CNT, LOP_RBIT, LOP_URECPE, LOP_URSQRTE,
    LOP_HADD, LOP_RHADD, LOP_HSUB, LOP_ABD, LOP_ABA,
    LOP_ADDL, LOP_SUBL, LOP_ABDL, LOP_ABAL, LOP_MLAL, LOP_MLSL, LOP_MULL, LOP_ADDW, LOP_SUBW,
    LOP_SQDMULL, LOP_SQDMLAL, LOP_SQDMLSL, LOP_PMULL, LOP_SHLL,
    LOP_XTN, LOP_SQXTN, LOP_UQXTN, LOP_SQXTUN, LOP_SHRN, LOP_RSHRN, LOP_QSHRN, LOP_QRSHRN, LOP_QSHRUN,
    LOP_QRSHRUN, LOP_ADDHN, LOP_RADDHN, LOP_SUBHN, LOP_RSUBHN,
    LOP_SHLLI,
    LOP_U = 256
};

enum FpOp : int {
    FOP_ADD = 0, FOP_SUB, FOP_MUL, FOP_MULX, FOP_NMUL, FOP_DIV, FOP_ABD, FOP_SQRT, FOP_MADD, FOP_RECPS, FOP_RSQRTS,
    FOP_MAX, FOP_MIN, FOP_MAXNM, FOP_MINNM
};

inline std::string block_label(uint64_t pc) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "L_%" PRIx64, pc);
    return buf;
}

inline std::string fn_symbol(uint64_t pc) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "f_%" PRIx64, pc);
    return buf;
}

inline uint64_t vfp_expand_imm(uint32_t imm8, unsigned N) {
    unsigned E = (N == 32) ? 8 : 11;
    unsigned F = N - E - 1;
    uint64_t sign = (imm8 >> 7) & 1;
    uint64_t b6 = (imm8 >> 6) & 1;
    uint64_t exp = ((b6 ^ 1) << (E - 1)) | ((b6 ? ((1ULL << (E - 3)) - 1ULL) : 0ULL) << 2) | ((imm8 >> 4) & 3);
    uint64_t frac = static_cast<uint64_t>(imm8 & 15) << (F - 4);
    return (sign << (N - 1)) | (exp << F) | frac;
}

inline uint64_t simd_expand_imm(unsigned op, unsigned cmode, uint32_t imm8) {
    unsigned c = cmode >> 1;
    uint64_t u8 = imm8;
    if (c == 0) return (u8 << 32) | u8;
    if (c == 1) return ((u8 << 8) << 32) | (u8 << 8);
    if (c == 2) return ((u8 << 16) << 32) | (u8 << 16);
    if (c == 3) return ((u8 << 24) << 32) | (u8 << 24);
    if (c == 4) {
        uint64_t r = 0;
        for (unsigned s = 0; s < 64; s += 16) r |= (u8 << s);
        return r;
    }
    if (c == 5) {
        uint64_t r = 0;
        for (unsigned s = 0; s < 64; s += 16) r |= ((u8 << 8) << s);
        return r;
    }
    if (c == 6) {
        uint64_t v = !(cmode & 1) ? ((u8 << 8) | 0xFFULL) : ((u8 << 16) | 0xFFFFULL);
        return (v << 32) | v;
    }
    if (!(cmode & 1) && !op) {
        uint64_t r = 0;
        for (unsigned s = 0; s < 64; s += 8) r |= (u8 << s);
        return r;
    }
    if (!(cmode & 1) && op) {
        uint64_t r = 0;
        for (unsigned i = 0; i < 8; ++i) {
            if ((u8 >> i) & 1) r |= (0xFFULL << (8 * i));
        }
        return r;
    }
    if ((cmode & 1) && !op) {
        uint64_t v = vfp_expand_imm(imm8, 32);
        return (v << 32) | v;
    }
    return vfp_expand_imm(imm8, 64);
}

class AArch64ToLLVMLifter {
public:
    AArch64ToLLVMLifter(uint64_t text_lo, uint64_t text_hi)
        : text_lo_(text_lo), text_hi_(text_hi) {}

    bool in_text(uint64_t a) const {
        return a >= text_lo_ && a < text_hi_ && (a % 4) == 0;
    }

    // Lift an entire guest function into an LLVM IR `Function`
    void lift_function(Module& M, Function* fn, const std::vector<uint32_t>& text_words,
                       uint64_t text_lo) {
        if (!fn->hle_symbol.empty()) {
            BasicBlock* entry_bb = fn->create_block("entry", fn->entry_pc);
            IRBuilder B(fn, entry_bb);
            B.CreateTailCallRet(fn->hle_symbol, llvm::CallInst::TCK_Tail, OpTag::DirectCall, fn->entry_pc);
            M.external_hle_fns.insert(fn->hle_symbol);
            return;
        }

        BasicBlock* entry_bb = fn->create_block("entry", fn->entry_pc);
        // Pre-create basic blocks for all label PCs in ascending order
        for (uint64_t lpc : fn->label_pcs) {
            fn->create_block(block_label(lpc), lpc);
        }

        IRBuilder B(fn, entry_bb);
        // Emit FN_ENTER profiler hook check in entry block, then branch to first label block
        emit_fn_enter(B, fn->entry_pc);
        emit_uncond_br(B, block_label(fn->entry_pc), fn->entry_pc);

        BasicBlock* cur_bb = fn->get_block(block_label(fn->entry_pc));
        B.set_insert_point(cur_bb);

        for (uint64_t pc = fn->entry_pc; pc < fn->end_pc; pc += 4) {
            uint32_t w = text_words[(pc - text_lo) / 4];
            fn->raw_words[pc] = w;

            if (pc != fn->entry_pc && fn->label_pcs.count(pc)) {
                BasicBlock* next_bb = fn->get_block(block_label(pc));
                if (B.insert_block() != next_bb) {
                    if (!B.insert_block()->has_terminator()) {
                        emit_uncond_br(B, next_bb->name, pc);
                    }
                    B.set_insert_point(next_bb);
                }
            } else if (B.insert_block()->has_terminator()) {
                // Dead/unreachable instructions after an unconditional terminator before the next label:
                // place in a synthetic unreachable block so CFG stays valid.
                BasicBlock* dead_bb = fn->create_block(block_label(pc) + "_unreach", pc);
                B.set_insert_point(dead_bb);
            }

            B.emit_guest_marker(pc, w);
            uint64_t fallthrough_pc = pc + 4;
            std::string fallthrough_label;
            if (fallthrough_pc < fn->end_pc) {
                if (!fn->label_pcs.count(fallthrough_pc)) {
                    // Ensure conditional branches have a fallthrough basic block if needed
                }
            }
            if (!lift_instruction(M, B, w, pc, fn)) {
                emit_udf(B, pc);
            }
        }

        // If the last block falls through past fn->end_pc:
        if (!B.insert_block()->has_terminator()) {
            if (fn->end_pc < text_hi_) {
                emit_jump_target(M, B, fn, fn->end_pc);
            } else {
                emit_udf(B, fn->end_pc);
            }
        }
    }

private:
    uint64_t text_lo_;
    uint64_t text_hi_;

    void emit_fn_enter(IRBuilder& B, uint64_t entry_pc) {
        Function* fn = B.function();
        BasicBlock* prof_hit_bb = fn->create_block("prof_hit");
        BasicBlock* prof_cont_bb = fn->create_block("prof_cont");

        Value pval = B.CreateVolatileLoadGlobalI32("g_fnprof", 4, "prof");
        Value pcmp = B.CreateICmpNE(TypeKind::Int32, pval, Value::imm_i32(0), "pcmp");
        B.CreateCondBr(pcmp, prof_hit_bb->name, prof_cont_bb->name, -1, entry_pc, OpTag::RawIR);

        B.set_insert_point(prof_hit_bb);
        Value lr = B.read_arch_reg(30, TypeKind::Int64);
        Value x0 = B.read_arch_reg(0, TypeKind::Int64);
        B.CreateVoidCall("fnprof_hit", {Value::imm_u64(entry_pc), lr, x0});
        B.CreateBr(prof_cont_bb->name, 0, OpTag::RawIR);

        B.set_insert_point(prof_cont_bb);
    }

    void emit_uncond_br(IRBuilder& B, const std::string& target_label, uint64_t target_pc = 0) {
        B.CreateBr(target_label, target_pc, OpTag::UncondBranch);
    }

    void emit_udf(IRBuilder& B, uint64_t pc) {
        B.CreateHorizonUdf(pc);
    }

    void emit_jump_target(Module& M, IRBuilder& B, Function* fn, uint64_t target_pc) {
        if (target_pc >= fn->entry_pc && target_pc < fn->end_pc) {
            emit_uncond_br(B, block_label(target_pc), target_pc);
        } else if (in_text(target_pc)) {
            std::string callee = fn_symbol(target_pc);
            M.external_guest_fns.insert(callee);
            fn->direct_callee_pcs.insert(target_pc);
            B.CreateTailCallRet(callee, llvm::CallInst::TCK_MustTail, OpTag::Return, target_pc);
        } else {
            B.write_cpu_field_i64(264, Value::imm_u64(target_pc));
            B.CreateTailCallRet("rt_dispatch", llvm::CallInst::TCK_MustTail, OpTag::Return, 0);
        }
    }

    Value get_gpr(IRBuilder& B, unsigned sf, unsigned r, bool sp_ok = false) {
        TypeKind t = sf ? TypeKind::Int64 : TypeKind::Int32;
        if (r == 31 && !sp_ok) {
            return Value::imm(t, 0);
        }
        return B.read_arch_reg(static_cast<int>(r), t);
    }

    void set_gpr(IRBuilder& B, unsigned sf, unsigned r, const Value& v, bool sp_ok = false) {
        (void)sf;
        if (r == 31 && !sp_ok) return;
        B.write_arch_reg(static_cast<int>(r), v);
    }

    Value emit_shift(IRBuilder& B, unsigned sf, const Value& val, unsigned typ, unsigned amt) {
        TypeKind t = sf ? TypeKind::Int64 : TypeKind::Int32;
        if (amt == 0) return val;
        Value a = Value::imm(t, amt);
        if (typ == 0) return B.binop("shl", t, val, a);
        if (typ == 1) return B.binop("lshr", t, val, a);
        if (typ == 2) return B.binop("ashr", t, val, a);
        // ROR via @llvm.fshr
        std::string intr = sf ? "llvm.fshr.i64" : "llvm.fshr.i32";
        return B.CreateCall(t, intr, {val, val, a}, "ror");
    }

    Value emit_extend(IRBuilder& B, unsigned rm, unsigned option, unsigned shift) {
        Value base = get_gpr(B, 1, rm, false);
        Value ext = base;
        switch (option) {
            case 0: ext = B.cast("zext", B.cast("trunc", base, TypeKind::Int8), TypeKind::Int64); break;
            case 1: ext = B.cast("zext", B.cast("trunc", base, TypeKind::Int16), TypeKind::Int64); break;
            case 2: ext = B.cast("zext", B.cast("trunc", base, TypeKind::Int32), TypeKind::Int64); break;
            case 3: ext = base; break;
            case 4: ext = B.cast("sext", B.cast("trunc", base, TypeKind::Int8), TypeKind::Int64); break;
            case 5: ext = B.cast("sext", B.cast("trunc", base, TypeKind::Int16), TypeKind::Int64); break;
            case 6: ext = B.cast("sext", B.cast("trunc", base, TypeKind::Int32), TypeKind::Int64); break;
            default: ext = base; break;
        }
        if (shift) {
            ext = B.binop("shl", TypeKind::Int64, ext, Value::imm_u64(shift));
        }
        return ext;
    }

    Value emit_pack_flags(IRBuilder& B, const Value& n_i1, const Value& z_i1, const Value& c_i1, const Value& v_i1,
                          int cmp_kind = 0, unsigned cmp_sf = 1, Value lhs = {}, Value rhs = {}, std::string res_reg = "") {
        Function* fn = B.function();
        auto& RB = B.raw();
        llvm::Type* i64_ty = RB.getInt64Ty();
        std::string nz64 = fn->next_reg("nz");
        std::string zz64 = fn->next_reg("zz");
        std::string cz64 = fn->next_reg("cz");
        std::string vz64 = fn->next_reg("vz");
        std::string n64  = fn->next_reg("ns");
        std::string z64  = fn->next_reg("zs");
        std::string c64  = fn->next_reg("cs");
        std::string v64  = fn->next_reg("vs");
        std::string nz   = fn->next_reg("nzor");
        std::string cv   = fn->next_reg("cvor");
        std::string nzcv = fn->next_reg("nzcv");

        llvm::Value* v_nz64 = RB.CreateZExt(B.resolve_llvm_value(n_i1), i64_ty, strip_ssa_pct(nz64));
        llvm::Value* v_zz64 = RB.CreateZExt(B.resolve_llvm_value(z_i1), i64_ty, strip_ssa_pct(zz64));
        llvm::Value* v_cz64 = RB.CreateZExt(B.resolve_llvm_value(c_i1), i64_ty, strip_ssa_pct(cz64));
        llvm::Value* v_vz64 = RB.CreateZExt(B.resolve_llvm_value(v_i1), i64_ty, strip_ssa_pct(vz64));
        llvm::Value* v_n64  = RB.CreateShl(v_nz64, RB.getInt64(31), strip_ssa_pct(n64));
        llvm::Value* v_z64  = RB.CreateShl(v_zz64, RB.getInt64(30), strip_ssa_pct(z64));
        llvm::Value* v_c64  = RB.CreateShl(v_cz64, RB.getInt64(29), strip_ssa_pct(c64));
        llvm::Value* v_v64  = RB.CreateShl(v_vz64, RB.getInt64(28), strip_ssa_pct(v64));
        llvm::Value* v_nz   = RB.CreateOr(v_n64, v_z64, strip_ssa_pct(nz));
        llvm::Value* v_cv   = RB.CreateOr(v_c64, v_v64, strip_ssa_pct(cv));
        llvm::Value* v_nzcv = RB.CreateOr(v_nz, v_cv, strip_ssa_pct(nzcv));
        fn->register_ssa_value(nzcv, v_nzcv);

        std::string pf = IRBuilder::format_insts({
            v_nz64, v_zz64, v_cz64, v_vz64, v_n64, v_z64, v_c64, v_v64, v_nz, v_cv, v_nzcv
        });
        Value nzcv_val = Value::reg(TypeKind::Int64, nzcv, v_nzcv);
        B.write_arch_reg(32, nzcv_val, OpTag::FlagWrite, std::move(pf), cmp_kind, cmp_sf, lhs, rhs, std::move(res_reg));
        return nzcv_val;
    }

    Value emit_adds_subs(IRBuilder& B, unsigned sf, bool is_sub, const Value& a, const Value& b) {
        TypeKind t = sf ? TypeKind::Int64 : TypeKind::Int32;
        Value r = B.binop(is_sub ? "sub" : "add", t, a, b);
        Function* fn = B.function();
        auto& RB = B.raw();
        llvm::Type* ty = B.llvm_type(t);
        llvm::Type* i64_ty = RB.getInt64Ty();
        llvm::Value* av = B.resolve_llvm_value(a);
        llvm::Value* bv = B.resolve_llvm_value(b);
        llvm::Value* rv = B.resolve_llvm_value(r);
        llvm::Value* zero = llvm::ConstantInt::get(ty, 0);

        std::string n_bit = fn->next_reg("fn");
        std::string z_bit = fn->next_reg("fz");
        std::string c_bit = fn->next_reg("fc");
        std::string xor1  = fn->next_reg("fx1");
        std::string xor2  = fn->next_reg("fx2");
        std::string vand  = fn->next_reg("fva");
        std::string v_bit = fn->next_reg("fv");

        llvm::Value* v_n = RB.CreateICmpSLT(rv, zero, strip_ssa_pct(n_bit));
        llvm::Value* v_z = RB.CreateICmpEQ(rv, zero, strip_ssa_pct(z_bit));
        llvm::Value* v_c = is_sub
                         ? RB.CreateICmpUGE(av, bv, strip_ssa_pct(c_bit))
                         : RB.CreateICmpULT(rv, av, strip_ssa_pct(c_bit));
        llvm::Value* v_x1 = RB.CreateXor(av, is_sub ? bv : rv, strip_ssa_pct(xor1));
        llvm::Value* v_x2 = RB.CreateXor(is_sub ? av : bv, rv, strip_ssa_pct(xor2));
        llvm::Value* v_va = RB.CreateAnd(v_x1, v_x2, strip_ssa_pct(vand));
        llvm::Value* v_v  = RB.CreateICmpSLT(v_va, zero, strip_ssa_pct(v_bit));

        std::string nz64 = fn->next_reg("nz");
        std::string zz64 = fn->next_reg("zz");
        std::string cz64 = fn->next_reg("cz");
        std::string vz64 = fn->next_reg("vz");
        std::string n64  = fn->next_reg("ns");
        std::string z64  = fn->next_reg("zs");
        std::string c64  = fn->next_reg("cs");
        std::string v64  = fn->next_reg("vs");
        std::string nz   = fn->next_reg("nzor");
        std::string cv   = fn->next_reg("cvor");
        std::string nzcv = fn->next_reg("nzcv");

        llvm::Value* v_nz64 = RB.CreateZExt(v_n, i64_ty, strip_ssa_pct(nz64));
        llvm::Value* v_zz64 = RB.CreateZExt(v_z, i64_ty, strip_ssa_pct(zz64));
        llvm::Value* v_cz64 = RB.CreateZExt(v_c, i64_ty, strip_ssa_pct(cz64));
        llvm::Value* v_vz64 = RB.CreateZExt(v_v, i64_ty, strip_ssa_pct(vz64));
        llvm::Value* v_n64  = RB.CreateShl(v_nz64, RB.getInt64(31), strip_ssa_pct(n64));
        llvm::Value* v_z64  = RB.CreateShl(v_zz64, RB.getInt64(30), strip_ssa_pct(z64));
        llvm::Value* v_c64  = RB.CreateShl(v_cz64, RB.getInt64(29), strip_ssa_pct(c64));
        llvm::Value* v_v64  = RB.CreateShl(v_vz64, RB.getInt64(28), strip_ssa_pct(v64));
        llvm::Value* v_nz   = RB.CreateOr(v_n64, v_z64, strip_ssa_pct(nz));
        llvm::Value* v_cv   = RB.CreateOr(v_c64, v_v64, strip_ssa_pct(cv));
        llvm::Value* v_nzcv = RB.CreateOr(v_nz, v_cv, strip_ssa_pct(nzcv));
        fn->register_ssa_value(nzcv, v_nzcv);

        std::string pre = IRBuilder::format_insts({
            v_n, v_z, v_c, v_x1, v_x2, v_va, v_v,
            v_nz64, v_zz64, v_cz64, v_vz64, v_n64, v_z64, v_c64, v_v64, v_nz, v_cv, v_nzcv
        });
        Value nzcv_val = Value::reg(TypeKind::Int64, nzcv, v_nzcv);
        B.write_arch_reg(32, nzcv_val, OpTag::FlagWrite, std::move(pre), is_sub ? 1 : 2, sf, a, b, r.repr);
        return r;
    }

    Value emit_logic_flags(IRBuilder& B, unsigned sf, const Value& r) {
        TypeKind t = sf ? TypeKind::Int64 : TypeKind::Int32;
        Function* fn = B.function();
        auto& RB = B.raw();
        llvm::Type* ty = B.llvm_type(t);
        llvm::Type* i64_ty = RB.getInt64Ty();
        llvm::Value* rv = B.resolve_llvm_value(r);
        llvm::Value* zero = llvm::ConstantInt::get(ty, 0);

        std::string n_bit = fn->next_reg("fn");
        std::string z_bit = fn->next_reg("fz");
        std::string nz64  = fn->next_reg("nz");
        std::string zz64  = fn->next_reg("zz");
        std::string n64   = fn->next_reg("ns");
        std::string z64   = fn->next_reg("zs");
        std::string nzcv  = fn->next_reg("nzcv");

        llvm::Value* v_n    = RB.CreateICmpSLT(rv, zero, strip_ssa_pct(n_bit));
        llvm::Value* v_z    = RB.CreateICmpEQ(rv, zero, strip_ssa_pct(z_bit));
        llvm::Value* v_nz64 = RB.CreateZExt(v_n, i64_ty, strip_ssa_pct(nz64));
        llvm::Value* v_zz64 = RB.CreateZExt(v_z, i64_ty, strip_ssa_pct(zz64));
        llvm::Value* v_n64  = RB.CreateShl(v_nz64, RB.getInt64(31), strip_ssa_pct(n64));
        llvm::Value* v_z64  = RB.CreateShl(v_zz64, RB.getInt64(30), strip_ssa_pct(z64));
        llvm::Value* v_nzcv = RB.CreateOr(v_n64, v_z64, strip_ssa_pct(nzcv));
        fn->register_ssa_value(nzcv, v_nzcv);

        std::string pre = IRBuilder::format_insts({
            v_n, v_z, v_nz64, v_zz64, v_n64, v_z64, v_nzcv
        });
        Value nzcv_val = Value::reg(TypeKind::Int64, nzcv, v_nzcv);
        B.write_arch_reg(32, nzcv_val, OpTag::FlagWrite, std::move(pre), 3, sf, r, Value::imm(t, 0), r.repr);
        return r;
    }

    Value emit_cond_i1(IRBuilder& B, unsigned cond) {
        if (cond >= 14) return Value::imm(TypeKind::Int1, 1);
        Value nzcv = B.read_arch_reg(32, TypeKind::Int64);
        Function* fn = B.function();
        auto& RB = B.raw();
        llvm::Value* nzcv_v = B.resolve_llvm_value(nzcv);
        llvm::Type* i1_ty = RB.getInt1Ty();
        std::string out_reg = fn->next_reg("cond");
        std::vector<const llvm::Value*> seq;
        llvm::Value* out_v = nullptr;

        auto emit_bit = [&](unsigned shift) -> llvm::Value* {
            std::string sh = fn->next_reg("fsh");
            std::string tr = fn->next_reg("ftr");
            llvm::Value* v_sh = RB.CreateLShr(nzcv_v, RB.getInt64(shift), strip_ssa_pct(sh));
            llvm::Value* v_tr = RB.CreateTrunc(v_sh, i1_ty, strip_ssa_pct(tr));
            seq.push_back(v_sh);
            seq.push_back(v_tr);
            return v_tr;
        };
        switch (cond) {
            case 0: { // EQ: Z==1
                std::string sh = fn->next_reg("fsh");
                llvm::Value* v_sh = RB.CreateLShr(nzcv_v, RB.getInt64(30), strip_ssa_pct(sh));
                out_v = RB.CreateTrunc(v_sh, i1_ty, strip_ssa_pct(out_reg));
                seq.push_back(v_sh);
                seq.push_back(out_v);
                break;
            }
            case 1: { // NE: Z==0
                llvm::Value* z = emit_bit(30);
                out_v = RB.CreateXor(z, RB.getTrue(), strip_ssa_pct(out_reg));
                seq.push_back(out_v);
                break;
            }
            case 2: { // CS: C==1
                std::string sh = fn->next_reg("fsh");
                llvm::Value* v_sh = RB.CreateLShr(nzcv_v, RB.getInt64(29), strip_ssa_pct(sh));
                out_v = RB.CreateTrunc(v_sh, i1_ty, strip_ssa_pct(out_reg));
                seq.push_back(v_sh);
                seq.push_back(out_v);
                break;
            }
            case 3: { // CC: C==0
                llvm::Value* c = emit_bit(29);
                out_v = RB.CreateXor(c, RB.getTrue(), strip_ssa_pct(out_reg));
                seq.push_back(out_v);
                break;
            }
            case 4: { // MI: N==1
                std::string sh = fn->next_reg("fsh");
                llvm::Value* v_sh = RB.CreateLShr(nzcv_v, RB.getInt64(31), strip_ssa_pct(sh));
                out_v = RB.CreateTrunc(v_sh, i1_ty, strip_ssa_pct(out_reg));
                seq.push_back(v_sh);
                seq.push_back(out_v);
                break;
            }
            case 5: { // PL: N==0
                llvm::Value* n = emit_bit(31);
                out_v = RB.CreateXor(n, RB.getTrue(), strip_ssa_pct(out_reg));
                seq.push_back(out_v);
                break;
            }
            case 6: { // VS: V==1
                std::string sh = fn->next_reg("fsh");
                llvm::Value* v_sh = RB.CreateLShr(nzcv_v, RB.getInt64(28), strip_ssa_pct(sh));
                out_v = RB.CreateTrunc(v_sh, i1_ty, strip_ssa_pct(out_reg));
                seq.push_back(v_sh);
                seq.push_back(out_v);
                break;
            }
            case 7: { // VC: V==0
                llvm::Value* v = emit_bit(28);
                out_v = RB.CreateXor(v, RB.getTrue(), strip_ssa_pct(out_reg));
                seq.push_back(out_v);
                break;
            }
            case 8: { // HI: C==1 && Z==0
                llvm::Value* c = emit_bit(29);
                llvm::Value* z = emit_bit(30);
                std::string nz = fn->next_reg("fnz");
                llvm::Value* v_nz = RB.CreateXor(z, RB.getTrue(), strip_ssa_pct(nz));
                out_v = RB.CreateAnd(c, v_nz, strip_ssa_pct(out_reg));
                seq.push_back(v_nz);
                seq.push_back(out_v);
                break;
            }
            case 9: { // LS: !(C==1 && Z==0)
                llvm::Value* c = emit_bit(29);
                llvm::Value* z = emit_bit(30);
                std::string nz = fn->next_reg("fnz");
                std::string hi = fn->next_reg("fhi");
                llvm::Value* v_nz = RB.CreateXor(z, RB.getTrue(), strip_ssa_pct(nz));
                llvm::Value* v_hi = RB.CreateAnd(c, v_nz, strip_ssa_pct(hi));
                out_v = RB.CreateXor(v_hi, RB.getTrue(), strip_ssa_pct(out_reg));
                seq.push_back(v_nz);
                seq.push_back(v_hi);
                seq.push_back(out_v);
                break;
            }
            case 10: { // GE: N==V
                llvm::Value* n = emit_bit(31);
                llvm::Value* v = emit_bit(28);
                out_v = RB.CreateICmpEQ(n, v, strip_ssa_pct(out_reg));
                seq.push_back(out_v);
                break;
            }
            case 11: { // LT: N!=V
                llvm::Value* n = emit_bit(31);
                llvm::Value* v = emit_bit(28);
                out_v = RB.CreateICmpNE(n, v, strip_ssa_pct(out_reg));
                seq.push_back(out_v);
                break;
            }
            case 12: { // GT: Z==0 && N==V
                llvm::Value* z = emit_bit(30);
                llvm::Value* n = emit_bit(31);
                llvm::Value* v = emit_bit(28);
                std::string nz = fn->next_reg("fnz");
                std::string eq = fn->next_reg("feq");
                llvm::Value* v_nz = RB.CreateXor(z, RB.getTrue(), strip_ssa_pct(nz));
                llvm::Value* v_eq = RB.CreateICmpEQ(n, v, strip_ssa_pct(eq));
                out_v = RB.CreateAnd(v_nz, v_eq, strip_ssa_pct(out_reg));
                seq.push_back(v_nz);
                seq.push_back(v_eq);
                seq.push_back(out_v);
                break;
            }
            case 13: { // LE: !(Z==0 && N==V)
                llvm::Value* z = emit_bit(30);
                llvm::Value* n = emit_bit(31);
                llvm::Value* v = emit_bit(28);
                std::string nz = fn->next_reg("fnz");
                std::string eq = fn->next_reg("feq");
                std::string gt = fn->next_reg("fgt");
                llvm::Value* v_nz = RB.CreateXor(z, RB.getTrue(), strip_ssa_pct(nz));
                llvm::Value* v_eq = RB.CreateICmpEQ(n, v, strip_ssa_pct(eq));
                llvm::Value* v_gt = RB.CreateAnd(v_nz, v_eq, strip_ssa_pct(gt));
                out_v = RB.CreateXor(v_gt, RB.getTrue(), strip_ssa_pct(out_reg));
                seq.push_back(v_nz);
                seq.push_back(v_eq);
                seq.push_back(v_gt);
                seq.push_back(out_v);
                break;
            }
            default:
                return Value::imm(TypeKind::Int1, 1);
        }
        fn->register_ssa_value(out_reg, out_v);
        std::ostringstream os;
        for (size_t i = 0; i < seq.size(); ++i) {
            if (i) os << "\n  ";
            os << format_llvm_inst(seq[i]);
        }
        return B.CreateCondEval(static_cast<int>(cond), out_reg, nzcv, os.str(),
                                llvm::dyn_cast_or_null<llvm::Instruction>(out_v));
    }

    bool lift_instruction(Module& M, IRBuilder& B, uint32_t w, uint64_t pc, Function* fn) {
        uint32_t op0 = bits(w, 25, 4);
        if ((op0 & 0b1110) == 0b1000) return lift_dp_imm(B, w, pc);
        if ((op0 & 0b1110) == 0b1010) return lift_branch_sys(M, B, w, pc, fn);
        if ((op0 & 0b0101) == 0b0100) return lift_ldst(B, w, pc);
        if ((op0 & 0b0111) == 0b0101) return lift_dp_reg(B, w, pc);
        if ((op0 & 0b0111) == 0b0111) return lift_fp_simd(B, w, pc);
        return false;
    }

    // -------------------------------------------------------------------------
    // Data Processing (Immediate)
    // -------------------------------------------------------------------------
    bool lift_dp_imm(IRBuilder& B, uint32_t w, uint64_t pc) {
        unsigned op = bits(w, 23, 3);
        unsigned sf = bits(w, 31, 1);
        unsigned rd = bits(w, 0, 5), rn = bits(w, 5, 5);
        TypeKind t = sf ? TypeKind::Int64 : TypeKind::Int32;

        if (op == 0 || op == 1) { // ADR / ADRP
            int64_t imm = sext((bits(w, 5, 19) << 2) | bits(w, 29, 2), 21);
            uint64_t val = sf ? ((pc & ~0xFFFULL) + static_cast<uint64_t>(imm << 12))
                              : static_cast<uint64_t>(static_cast<int64_t>(pc) + imm);
            set_gpr(B, 1, rd, Value::imm_u64(val));
            return true;
        }
        if (op == 2) { // ADD/SUB imm
            unsigned opc = bits(w, 30, 1), S = bits(w, 29, 1), sh = bits(w, 22, 1);
            uint64_t imm = static_cast<uint64_t>(bits(w, 10, 12)) << (sh ? 12 : 0);
            Value a = get_gpr(B, sf, rn, true);
            Value b = Value::imm(t, imm);
            if (S) {
                Value r = emit_adds_subs(B, sf, opc != 0, a, b);
                set_gpr(B, sf, rd, r, false);
            } else {
                Value r = B.binop(opc ? "sub" : "add", t, a, b);
                set_gpr(B, sf, rd, r, true);
            }
            return true;
        }
        if (op == 4) { // Logical imm
            unsigned opc = bits(w, 29, 2), N = bits(w, 22, 1);
            if (!sf && N) return false;
            unsigned size = sf ? 64 : 32;
            uint64_t wmask = 0, tmask = 0;
            if (!decode_bit_masks(N, bits(w, 10, 6), bits(w, 16, 6), size, true, wmask, tmask)) return false;
            Value a = get_gpr(B, sf, rn);
            Value b = Value::imm(t, wmask);
            if (opc == 0) { set_gpr(B, sf, rd, B.binop("and", t, a, b), true); return true; }
            if (opc == 1) { set_gpr(B, sf, rd, B.binop("or",  t, a, b), true); return true; }
            if (opc == 2) { set_gpr(B, sf, rd, B.binop("xor", t, a, b), true); return true; }
            Value r = emit_logic_flags(B, sf, B.binop("and", t, a, b));
            set_gpr(B, sf, rd, r, false);
            return true;
        }
        if (op == 5) { // Move wide (MOVN / MOVZ / MOVK)
            unsigned opc = bits(w, 29, 2), hw = bits(w, 21, 2);
            if (!sf && hw > 1) return false;
            uint64_t imm = static_cast<uint64_t>(bits(w, 5, 16)) << (hw * 16);
            if (opc == 0) {
                set_gpr(B, sf, rd, Value::imm(t, (~imm) & ones(sf ? 64 : 32)));
                return true;
            }
            if (opc == 2) {
                set_gpr(B, sf, rd, Value::imm(t, imm));
                return true;
            }
            if (opc == 3) {
                uint64_t mask = ~(0xFFFFULL << (hw * 16));
                Value cur = get_gpr(B, 1, rd);
                Value masked = B.binop("and", TypeKind::Int64, cur, Value::imm_u64(mask));
                Value res = B.binop("or", TypeKind::Int64, masked, Value::imm_u64(imm));
                set_gpr(B, sf, rd, sf ? res : B.cast("trunc", res, TypeKind::Int32));
                return true;
            }
            return false;
        }
        if (op == 6) { // Bitfield (SBFM / BFM / UBFM)
            unsigned opc = bits(w, 29, 2), N = bits(w, 22, 1);
            unsigned immr = bits(w, 16, 6), imms = bits(w, 10, 6);
            unsigned size = sf ? 64 : 32;
            if (N != sf) return false;
            uint64_t wmask = 0, tmask = 0;
            if (!decode_bit_masks(N, imms, immr, size, false, wmask, tmask)) return false;
            Value src = get_gpr(B, sf, rn);
            Value rot = immr ? emit_shift(B, sf, src, 3, immr) : src;
            if (opc == 2) { // UBFM
                set_gpr(B, sf, rd, B.binop("and", t, rot, Value::imm(t, wmask & tmask)));
                return true;
            }
            if (opc == 0) { // SBFM
                Value sign_bit = B.binop("and", t, B.binop("lshr", t, src, Value::imm(t, imms)), Value::imm(t, 1));
                Value top = B.binop("sub", t, Value::imm(t, 0), sign_bit);
                Value hi = B.binop("and", t, top, Value::imm(t, (~tmask) & ones(size)));
                Value lo = B.binop("and", t, rot, Value::imm(t, wmask & tmask));
                set_gpr(B, sf, rd, B.binop("or", t, hi, lo));
                return true;
            }
            if (opc == 1) { // BFM
                Value dst = get_gpr(B, sf, rd);
                Value bot = B.binop("or", t, B.binop("and", t, dst, Value::imm(t, (~wmask) & ones(size))),
                                             B.binop("and", t, rot, Value::imm(t, wmask)));
                Value res = B.binop("or", t, B.binop("and", t, dst, Value::imm(t, (~tmask) & ones(size))),
                                             B.binop("and", t, bot, Value::imm(t, tmask)));
                set_gpr(B, sf, rd, res);
                return true;
            }
            return false;
        }
        if (op == 7) { // EXTR
            unsigned rm = bits(w, 16, 5), lsb = bits(w, 10, 6);
            Value vm = get_gpr(B, sf, rm);
            if (lsb == 0) {
                set_gpr(B, sf, rd, vm);
                return true;
            }
            Value vn = get_gpr(B, sf, rn);
            std::string intr = sf ? "llvm.fshr.i64" : "llvm.fshr.i32";
            set_gpr(B, sf, rd, B.CreateCall(t, intr, {vn, vm, Value::imm(t, lsb)}, "extr"));
            return true;
        }
        return false;
    }

    // -------------------------------------------------------------------------
    // Branches, Exception Generation & System Instructions
    // -------------------------------------------------------------------------
    bool lift_branch_sys(Module& M, IRBuilder& B, uint32_t w, uint64_t pc, Function* fn) {
        if (bits(w, 26, 5) == 0b00101) { // B / BL
            uint64_t t = static_cast<uint64_t>(static_cast<int64_t>(pc) + sext(bits(w, 0, 26), 26) * 4);
            if (!bits(w, 31, 1)) { // B
                emit_jump_target(M, B, fn, t);
                return true;
            }
            // BL
            B.write_arch_reg(30, Value::imm_u64(pc + 4));
            if (in_text(t)) {
                std::string callee = fn_symbol(t);
                M.external_guest_fns.insert(callee);
                fn->direct_callee_pcs.insert(t);
                B.CreateDirectCall(callee, pc, t);
            } else {
                B.write_cpu_field_i64(264, Value::imm_u64(t));
                B.CreateIndirectCall(Value::imm_u64(t), pc);
            }
            return true;
        }
        if (bits(w, 25, 7) == 0b0101010 && !bits(w, 4, 1)) { // B.cond
            uint64_t t = static_cast<uint64_t>(static_cast<int64_t>(pc) + sext(bits(w, 5, 19), 19) * 4);
            unsigned cond = bits(w, 0, 4);
            if (cond >= 14) {
                emit_jump_target(M, B, fn, t);
                return true;
            }
            Value c_i1 = emit_cond_i1(B, cond);
            emit_conditional_jump(M, B, fn, pc, c_i1, t, static_cast<int>(cond));
            return true;
        }
        if (bits(w, 25, 6) == 0b011010) { // CBZ / CBNZ
            unsigned sf = bits(w, 31, 1), nz = bits(w, 24, 1), rt = bits(w, 0, 5);
            uint64_t t = static_cast<uint64_t>(static_cast<int64_t>(pc) + sext(bits(w, 5, 19), 19) * 4);
            TypeKind tk = sf ? TypeKind::Int64 : TypeKind::Int32;
            Value v = get_gpr(B, sf, rt);
            Value c_i1 = B.icmp(nz ? "ne" : "eq", tk, v, Value::imm(tk, 0));
            emit_conditional_jump(M, B, fn, pc, c_i1, t, -1);
            return true;
        }
        if (bits(w, 25, 6) == 0b011011) { // TBZ / TBNZ
            unsigned b = (bits(w, 31, 1) << 5) | bits(w, 19, 5);
            unsigned nz = bits(w, 24, 1), rt = bits(w, 0, 5);
            uint64_t t = static_cast<uint64_t>(static_cast<int64_t>(pc) + sext(bits(w, 5, 14), 14) * 4);
            Value v = get_gpr(B, 1, rt);
            Value masked = B.binop("and", TypeKind::Int64, v, Value::imm_u64(1ULL << b));
            Value c_i1 = B.icmp(nz ? "ne" : "eq", TypeKind::Int64, masked, Value::imm_u64(0));
            emit_conditional_jump(M, B, fn, pc, c_i1, t, -1);
            return true;
        }
        if (bits(w, 25, 7) == 0b1101011) { // Branch register (BR / BLR / RET)
            unsigned opc = bits(w, 21, 4), rn = bits(w, 5, 5);
            if (bits(w, 16, 5) != 31 || bits(w, 10, 6) != 0 || bits(w, 0, 5) != 0) return false;
            if (opc == 0) { // BR Xn -> lowered by JumpTableLoweringPass
                Value target = get_gpr(B, 1, rn);
                B.CreateIndirectJump(target, pc);
                return true;
            }
            if (opc == 1) { // BLR Xn
                Value target = get_gpr(B, 1, rn);
                B.write_cpu_field_i64(264, target);
                B.write_arch_reg(30, Value::imm_u64(pc + 4));
                B.CreateIndirectCall(target, pc);
                return true;
            }
            if (opc == 2) { // RET
                B.CreateRetVoid(pc);
                return true;
            }
            return false;
        }
        if (bits(w, 24, 8) == 0b11010100) { // Exception generation (SVC / BRK)
            unsigned opc = bits(w, 21, 3), ll = bits(w, 0, 5), imm = bits(w, 5, 16);
            if (opc == 0 && ll == 1) {
                B.CreateHorizonSvc(imm, pc);
                return true;
            }
            if (opc == 1 && ll == 0) {
                B.CreateHorizonBrk(imm, pc);
                return true;
            }
            return false;
        }
        if (bits(w, 22, 10) == 0b1101010100) {
            return lift_system(B, w, pc);
        }
        return false;
    }

    void emit_conditional_jump(Module& M, IRBuilder& B, Function* fn, uint64_t pc,
                               const Value& cond_i1, uint64_t target_pc, int cond_code) {
        std::string fall_name = block_label(pc + 4) + (fn->label_pcs.count(pc + 4) ? "" : "_ft");
        BasicBlock* fall_bb = fn->create_block(fall_name, pc + 4);

        if (target_pc >= fn->entry_pc && target_pc < fn->end_pc) {
            std::string taken_name = block_label(target_pc);
            B.CreateGuestCondBr(cond_i1, taken_name, fall_bb->name, cond_code, pc, target_pc);
        } else {
            BasicBlock* tail_bb = fn->create_block(block_label(pc) + "_ext_taken", pc);
            B.CreateGuestCondBr(cond_i1, tail_bb->name, fall_bb->name, cond_code, pc, target_pc);

            B.set_insert_point(tail_bb);
            emit_jump_target(M, B, fn, target_pc);
        }
        B.set_insert_point(fall_bb);
    }

    bool lift_system(IRBuilder& B, uint32_t w, uint64_t pc) {
        unsigned L = bits(w, 21, 1);
        unsigned op0 = bits(w, 19, 2), op1 = bits(w, 16, 3), crn = bits(w, 12, 4),
                 crm = bits(w, 8, 4), op2 = bits(w, 5, 3), rt = bits(w, 0, 5);
        if (op0 == 0 && !L) {
            if (crn == 2 && op1 == 3 && rt == 31) return true; // NOP / YIELD / WFE / WFI hints
            if (crn == 3 && op1 == 3 && rt == 31) {
                B.CreateHorizonBarrier(op2, pc);
                return true;
            }
            return false;
        }
        if (op0 == 1) {
            if (!L && op1 == 3 && crn == 7 && crm == 4 && op2 == 1) { // DC ZVA
                Value addr = get_gpr(B, 1, rt);
                B.CreateHorizonDcZva(addr, pc);
            }
            return true;
        }
        uint32_t key = (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2;
        if (L) { // MRS
            B.CreateHorizonSysRegRead(static_cast<int>(rt), key, pc);
            return true;
        } else { // MSR
            Value v = get_gpr(B, 1, rt);
            B.CreateHorizonSysRegWrite(v, key, pc);
            return true;
        }
    }

    // -------------------------------------------------------------------------
    // Loads & Stores (Integer, Exclusive/Atomic, Pairs, SIMD, Structures)
    // -------------------------------------------------------------------------
    bool lift_ldst(IRBuilder& B, uint32_t w, uint64_t pc) {
        if (bits(w, 26, 1)) return lift_ldst_simd(B, w, pc);
        if (bits(w, 24, 6) == 0b001000) return lift_ldst_excl(B, w, pc);
        if (bits(w, 27, 3) == 0b011 && bits(w, 24, 2) == 0) { // Literal
            unsigned opc = bits(w, 30, 2), rt = bits(w, 0, 5);
            uint64_t addr = static_cast<uint64_t>(static_cast<int64_t>(pc) + sext(bits(w, 5, 19), 19) * 4);
            Value ea = Value::imm_u64(addr);
            if (opc == 0) { set_gpr(B, 0, rt, B.guest_load(TypeKind::Int32, ea, pc)); return true; }
            if (opc == 1) { set_gpr(B, 1, rt, B.guest_load(TypeKind::Int64, ea, pc)); return true; }
            if (opc == 2) { set_gpr(B, 1, rt, B.sext(B.guest_load(TypeKind::Int32, ea, pc), TypeKind::Int64)); return true; }
            return true; // PRFM
        }
        if (bits(w, 27, 3) == 0b101) return lift_ldst_pair(B, w, pc);
        if (bits(w, 27, 3) == 0b111) return lift_ldst_reg(B, w, pc);
        return false;
    }

    bool lift_ldst_excl(IRBuilder& B, uint32_t w, uint64_t pc) {
        (void)pc;
        unsigned size = bits(w, 30, 2), o2 = bits(w, 23, 1), L = bits(w, 22, 1), o1 = bits(w, 21, 1),
                 rs = bits(w, 16, 5), rn = bits(w, 5, 5), rt = bits(w, 0, 5);
        if (o1) return false;
        unsigned nbits = 8u << size;
        unsigned align = 1u << size;
        TypeKind mt = int_type_for_bits(nbits);
        Value addr = get_gpr(B, 1, rn, true);
        Value p = B.CreateIntToPtr(addr, "atp");

        if (o2 == 0) { // LDXR / LDAXR / STXR / STLXR
            if (L) {
                Value v = B.CreateAtomicLoad(mt, p, align, llvm::AtomicOrdering::Acquire, "ldx", true);
                Value v64 = (mt == TypeKind::Int64) ? v : B.cast("zext", v, TypeKind::Int64);
                B.write_cpu_field_i64(296, addr); // excl_addr
                B.write_cpu_field_i64(304, v64);  // excl_val
                set_gpr(B, 1, rt, v64);
                return true;
            } else {
                // STXR: compare excl_addr == addr, then cmpxchg
                Value excl_a = B.read_cpu_field_i64(296);
                Value excl_v = B.read_cpu_field_i64(304);
                B.write_cpu_field_i64(296, Value::imm_u64(~0ULL));
                Value addr_match = B.icmp("eq", TypeKind::Int64, excl_a, addr);
                Value exp_v = (mt == TypeKind::Int64) ? excl_v : B.cast("trunc", excl_v, mt);
                Value new_v = (mt == TypeKind::Int64) ? get_gpr(B, 1, rt) : B.cast("trunc", get_gpr(B, 0, rt), mt);

                Value cx_ok = B.CreateAtomicCmpXchgOk(mt, p, exp_v, new_v, align);
                Value both_ok = B.binop("and", TypeKind::Int1, addr_match, cx_ok);
                Value status = B.select(TypeKind::Int32, both_ok, Value::imm_i32(0), Value::imm_i32(1));
                set_gpr(B, 0, rs, status);
                return true;
            }
        } else { // LDAR / STLR
            if (L) {
                Value v = B.CreateAtomicLoad(mt, p, align, llvm::AtomicOrdering::Acquire, "lda", true);
                Value v64 = (mt == TypeKind::Int64) ? v : B.cast("zext", v, TypeKind::Int64);
                set_gpr(B, 1, rt, v64);
                return true;
            } else {
                Value val = (mt == TypeKind::Int64) ? get_gpr(B, 1, rt) : B.cast("trunc", get_gpr(B, 0, rt), mt);
                B.CreateAtomicStore(mt, val, p, align, llvm::AtomicOrdering::Release, true);
                return true;
            }
        }
    }

    bool lift_ldst_pair(IRBuilder& B, uint32_t w, uint64_t pc) {
        unsigned opc = bits(w, 30, 2), mode = bits(w, 23, 2), L = bits(w, 22, 1);
        int64_t imm7 = sext(bits(w, 15, 7), 7);
        unsigned rt2 = bits(w, 10, 5), rn = bits(w, 5, 5), rt = bits(w, 0, 5);
        if (opc == 3) return false;
        unsigned scale = (opc == 2) ? 3 : 2;
        int64_t off = imm7 << scale;
        Value base = get_gpr(B, 1, rn, true);
        bool post = (mode == 1);
        bool wb = (mode == 1 || mode == 3);
        Value off_val = Value::imm_u64(static_cast<uint64_t>(off));
        Value ea = post ? base : (off == 0 ? base : B.binop("add", TypeKind::Int64, base, off_val));
        Value ea2 = B.binop("add", TypeKind::Int64, ea, Value::imm_u64(1ULL << scale));

        Value a, b;
        if (L) {
            if (opc == 2) {
                a = B.guest_load(TypeKind::Int64, ea, pc);
                b = B.guest_load(TypeKind::Int64, ea2, pc);
            } else if (opc == 1) { // LDPSW
                a = B.sext(B.guest_load(TypeKind::Int32, ea, pc), TypeKind::Int64);
                b = B.sext(B.guest_load(TypeKind::Int32, ea2, pc), TypeKind::Int64);
            } else {
                a = B.cast("zext", B.guest_load(TypeKind::Int32, ea, pc), TypeKind::Int64);
                b = B.cast("zext", B.guest_load(TypeKind::Int32, ea2, pc), TypeKind::Int64);
            }
        } else {
            if (opc == 2) {
                B.guest_store(TypeKind::Int64, ea, get_gpr(B, 1, rt), pc);
                B.guest_store(TypeKind::Int64, ea2, get_gpr(B, 1, rt2), pc);
            } else {
                B.guest_store(TypeKind::Int32, ea, get_gpr(B, 0, rt), pc);
                B.guest_store(TypeKind::Int32, ea2, get_gpr(B, 0, rt2), pc);
            }
        }
        if (wb) {
            Value wb_val = post ? B.binop("add", TypeKind::Int64, ea, off_val) : ea;
            set_gpr(B, 1, rn, wb_val, true);
        }
        if (L) {
            set_gpr(B, 1, rt, a);
            set_gpr(B, 1, rt2, b);
        }
        return true;
    }

    bool lift_ldst_reg(IRBuilder& B, uint32_t w, uint64_t pc) {
        unsigned size = bits(w, 30, 2), opc = bits(w, 22, 2), rn = bits(w, 5, 5), rt = bits(w, 0, 5);
        Value base = get_gpr(B, 1, rn, true);
        Value ea = base;
        std::optional<Value> wb;

        if (bits(w, 24, 2) == 1) {
            uint64_t off = static_cast<uint64_t>(bits(w, 10, 12)) << size;
            if (off) ea = B.binop("add", TypeKind::Int64, base, Value::imm_u64(off));
        } else {
            unsigned kind = bits(w, 10, 2);
            if (bits(w, 21, 1)) {
                if (kind != 2) return false;
                unsigned rm = bits(w, 16, 5), option = bits(w, 13, 3), S = bits(w, 12, 1);
                if (option != 2 && option != 3 && option != 6 && option != 7) return false;
                ea = B.binop("add", TypeKind::Int64, base, emit_extend(B, rm, option, S ? size : 0));
            } else {
                int64_t imm9 = sext(bits(w, 12, 9), 9);
                Value off_v = Value::imm_u64(static_cast<uint64_t>(imm9));
                if (kind == 0 || kind == 2) {
                    if (imm9) ea = B.binop("add", TypeKind::Int64, base, off_v);
                } else if (kind == 1) {
                    ea = base;
                    wb = B.binop("add", TypeKind::Int64, base, off_v);
                } else {
                    ea = B.binop("add", TypeKind::Int64, base, off_v);
                    wb = ea;
                }
            }
        }

        unsigned nbits = 8u << size;
        TypeKind mt = int_type_for_bits(nbits);
        std::optional<Value> loaded;
        unsigned dst_sf = 1;

        if (opc == 0) {
            Value sv = (size == 3) ? get_gpr(B, 1, rt) : B.cast("trunc", get_gpr(B, 0, rt), mt);
            B.guest_store(mt, ea, sv, pc);
        } else if (opc == 1) {
            Value lv = B.guest_load(mt, ea, pc);
            loaded = (mt == TypeKind::Int64) ? lv : B.cast("zext", lv, TypeKind::Int64);
        } else if (size == 3) {
            return true; // PRFM
        } else if (size == 2 && opc == 3) {
            return false;
        } else {
            Value lv = B.guest_load(mt, ea, pc);
            loaded = B.sext(lv, opc == 2 ? TypeKind::Int64 : TypeKind::Int32);
            dst_sf = (opc == 2) ? 1 : 0;
        }

        if (wb.has_value()) {
            set_gpr(B, 1, rn, *wb, true);
        }
        if (loaded.has_value()) {
            set_gpr(B, dst_sf, rt, *loaded);
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // Data Processing (Register)
    // -------------------------------------------------------------------------
    bool lift_dp_reg(IRBuilder& B, uint32_t w, uint64_t pc) {
        (void)pc;
        unsigned sf = bits(w, 31, 1);
        unsigned rd = bits(w, 0, 5), rn = bits(w, 5, 5), rm = bits(w, 16, 5);
        unsigned size = sf ? 64 : 32;
        TypeKind t = sf ? TypeKind::Int64 : TypeKind::Int32;

        if (!bits(w, 28, 1)) {
            if (bits(w, 24, 4) == 0b1010) { // Logical shifted
                unsigned opc = bits(w, 29, 2), sh = bits(w, 22, 2), N = bits(w, 21, 1), amt = bits(w, 10, 6);
                if (!sf && amt >= 32) return false;
                Value b = emit_shift(B, sf, get_gpr(B, sf, rm), sh, amt);
                if (N) b = B.binop("xor", t, b, Value::imm(t, ones(size)));
                Value a = get_gpr(B, sf, rn);
                const char* bop = (opc == 0 || opc == 3) ? "and" : (opc == 1) ? "or" : "xor";
                Value r = B.binop(bop, t, a, b);
                if (opc == 3) r = emit_logic_flags(B, sf, r);
                set_gpr(B, sf, rd, r);
                return true;
            }
            if (bits(w, 24, 4) == 0b1011) { // Add/sub shifted or extended
                unsigned op = bits(w, 30, 1), S = bits(w, 29, 1);
                Value a, b;
                bool sp_ok = false;
                if (!bits(w, 21, 1)) {
                    unsigned sh = bits(w, 22, 2), amt = bits(w, 10, 6);
                    if (sh == 3) return false;
                    a = get_gpr(B, sf, rn);
                    b = emit_shift(B, sf, get_gpr(B, sf, rm), sh, amt);
                } else {
                    if (bits(w, 22, 2)) return false;
                    unsigned option = bits(w, 13, 3), amt = bits(w, 10, 3);
                    a = get_gpr(B, sf, rn, true);
                    b = emit_extend(B, rm, option, amt);
                    if (!sf) b = B.cast("trunc", b, TypeKind::Int32);
                    sp_ok = !S;
                }
                if (S) {
                    Value r = emit_adds_subs(B, sf, op != 0, a, b);
                    set_gpr(B, sf, rd, r, false);
                } else {
                    set_gpr(B, sf, rd, B.binop(op ? "sub" : "add", t, a, b), sp_ok);
                }
                return true;
            }
            return false;
        }

        if (bits(w, 24, 4) == 0b1011) { // 3-source (MADD, MSUB, SMADDL, UMADDL, SMULH, UMULH)
            unsigned op54 = bits(w, 29, 2), op31 = bits(w, 21, 3), o0 = bits(w, 15, 1), ra = bits(w, 10, 5);
            if (op54) return false;
            const char* add_op = o0 ? "sub" : "add";
            if (op31 == 0) {
                Value prod = B.binop("mul", t, get_gpr(B, sf, rn), get_gpr(B, sf, rm));
                set_gpr(B, sf, rd, B.binop(add_op, t, get_gpr(B, sf, ra), prod));
                return true;
            }
            if (op31 == 1 || op31 == 5) { // SMADDL / SMSUBL / UMADDL / UMSUBL
                const char* ext = (op31 == 1) ? "sext" : "zext";
                Value n64 = B.cast(ext, get_gpr(B, 0, rn), TypeKind::Int64);
                Value m64 = B.cast(ext, get_gpr(B, 0, rm), TypeKind::Int64);
                Value prod = B.binop("mul", TypeKind::Int64, n64, m64);
                set_gpr(B, 1, rd, B.binop(add_op, TypeKind::Int64, get_gpr(B, 1, ra), prod));
                return true;
            }
            if (op31 == 2 || op31 == 6) { // SMULH / UMULH
                const char* ext = (op31 == 2) ? "sext" : "zext";
                Value n128 = B.cast(ext, get_gpr(B, 1, rn), TypeKind::Int128);
                Value m128 = B.cast(ext, get_gpr(B, 1, rm), TypeKind::Int128);
                Value prod = B.binop("mul", TypeKind::Int128, n128, m128);
                Value hi   = B.binop("lshr", TypeKind::Int128, prod, Value::imm(TypeKind::Int128, 64));
                set_gpr(B, 1, rd, B.cast("trunc", hi, TypeKind::Int64));
                return true;
            }
            return false;
        }

        unsigned op2 = bits(w, 21, 4);
        if (op2 == 0) { // ADC / ADCS / SBC / SBCS
            unsigned op = bits(w, 30, 1), S = bits(w, 29, 1);
            Value a = get_gpr(B, sf, rn);
            Value b = get_gpr(B, sf, rm);
            if (op) b = B.binop("xor", t, b, Value::imm(t, ones(size))); // ~b for SBC/SBCS
            Value nzcv = B.read_arch_reg(32, TypeKind::Int64);
            Value cin64 = B.binop("and", TypeKind::Int64, B.binop("lshr", TypeKind::Int64, nzcv, Value::imm_u64(29)), Value::imm_u64(1));
            Value cin = sf ? cin64 : B.cast("trunc", cin64, TypeKind::Int32);
            Value ab = B.binop("add", t, a, b);
            Value r  = B.binop("add", t, ab, cin);
            if (S) {
                Value n_bit = B.icmp("slt", t, r, Value::imm(t, 0));
                Value z_bit = B.icmp("eq", t, r, Value::imm(t, 0));
                Value c1 = B.icmp("ult", t, ab, a);
                Value c2 = B.icmp("ult", t, r, ab);
                Value c_bit = B.binop("or", TypeKind::Int1, c1, c2);
                Value xor1 = B.binop("xor", t, a, r);
                Value xor2 = B.binop("xor", t, b, r);
                Value v_bit = B.icmp("slt", t, B.binop("and", t, xor1, xor2), Value::imm(t, 0));
                emit_pack_flags(B, n_bit, z_bit, c_bit, v_bit);
            }
            set_gpr(B, sf, rd, r);
            return true;
        }
        if (op2 == 2) { // CCMP / CCMN
            unsigned op = bits(w, 30, 1), imm_form = bits(w, 11, 1), cond = bits(w, 12, 4), nzcv_imm = bits(w, 0, 4);
            Value c_i1 = emit_cond_i1(B, cond);
            Value a = get_gpr(B, sf, rn);
            Value b = imm_form ? Value::imm(t, rm) : get_gpr(B, sf, rm);
            // Compute flags if condition holds, else select immediate nzcv << 28
            Value r = B.binop(op ? "sub" : "add", t, a, b);
            Value n_bit = B.icmp("slt", t, r, Value::imm(t, 0));
            Value z_bit = B.icmp("eq", t, r, Value::imm(t, 0));
            Value c_bit = op ? B.icmp("uge", t, a, b) : B.icmp("ult", t, r, a);
            Value xor1 = B.binop("xor", t, a, op ? b : r);
            Value xor2 = B.binop("xor", t, op ? a : b, r);
            Value v_bit = B.icmp("slt", t, B.binop("and", t, xor1, xor2), Value::imm(t, 0));

            Value n64 = B.binop("shl", TypeKind::Int64, B.cast("zext", n_bit, TypeKind::Int64), Value::imm_u64(31));
            Value z64 = B.binop("shl", TypeKind::Int64, B.cast("zext", z_bit, TypeKind::Int64), Value::imm_u64(30));
            Value c64 = B.binop("shl", TypeKind::Int64, B.cast("zext", c_bit, TypeKind::Int64), Value::imm_u64(29));
            Value v64 = B.binop("shl", TypeKind::Int64, B.cast("zext", v_bit, TypeKind::Int64), Value::imm_u64(28));
            Value packed = B.binop("or", TypeKind::Int64, B.binop("or", TypeKind::Int64, n64, z64),
                                                          B.binop("or", TypeKind::Int64, c64, v64));
            Value final_nzcv = B.select(TypeKind::Int64, c_i1, packed, Value::imm_u64(static_cast<uint64_t>(nzcv_imm) << 28));
            B.write_arch_reg(32, final_nzcv, OpTag::FlagWrite);
            return true;
        }
        if (op2 == 4) { // CSEL / CSINC / CSINV / CSNEG
            unsigned op = bits(w, 30, 1), o2 = bits(w, 10, 2), cond = bits(w, 12, 4);
            Value a = get_gpr(B, sf, rn);
            Value b = get_gpr(B, sf, rm);
            Value alt = b;
            if (op == 0 && o2 == 1) alt = B.binop("add", t, b, Value::imm(t, 1));
            else if (op == 1 && o2 == 0) alt = B.binop("xor", t, b, Value::imm(t, ones(size)));
            else if (op == 1 && o2 == 1) alt = B.binop("sub", t, Value::imm(t, 0), b);
            else if (op != 0 || o2 != 0) return false;
            Value c_i1 = emit_cond_i1(B, cond);
            set_gpr(B, sf, rd, B.select(t, c_i1, a, alt));
            return true;
        }
        if (op2 == 6) {
            unsigned opcode = bits(w, 10, 6);
            if (bits(w, 30, 1)) { // 1-source
                if (bits(w, 16, 5)) return false;
                Value a = get_gpr(B, sf, rn);
                if (opcode == 0) { // RBIT
                    std::string intr = sf ? "llvm.bitreverse.i64" : "llvm.bitreverse.i32";
                    set_gpr(B, sf, rd, B.CreateCall(t, intr, {a}, "rbit"));
                    return true;
                }
                if (opcode == 1) { // REV16
                    uint64_t mask = sf ? 0x00FF00FF00FF00FFULL : 0x00FF00FFULL;
                    Value lo = B.binop("shl", t, B.binop("and", t, a, Value::imm(t, mask)), Value::imm(t, 8));
                    Value hi = B.binop("and", t, B.binop("lshr", t, a, Value::imm(t, 8)), Value::imm(t, mask));
                    set_gpr(B, sf, rd, B.binop("or", t, lo, hi));
                    return true;
                }
                if (opcode == 2) { // REV32
                    if (!sf) {
                        set_gpr(B, 0, rd, B.CreateCall(TypeKind::Int32, "llvm.bswap.i32", {a}, "bsw"));
                    } else {
                        Value bsw = B.CreateCall(TypeKind::Int64, "llvm.bswap.i64", {a}, "bsw");
                        Value r = emit_shift(B, 1, bsw, 3, 32);
                        set_gpr(B, 1, rd, r);
                    }
                    return true;
                }
                if (opcode == 3 && sf) { // REV64
                    set_gpr(B, 1, rd, B.CreateCall(TypeKind::Int64, "llvm.bswap.i64", {a}, "bsw"));
                    return true;
                }
                if (opcode == 4) { // CLZ
                    std::string intr = sf ? "llvm.ctlz.i64" : "llvm.ctlz.i32";
                    set_gpr(B, sf, rd, B.CreateCall(t, intr, {a, Value::imm(TypeKind::Int1, 0)}, "clz"));
                    return true;
                }
                if (opcode == 5) { // CLS
                    Value sar = B.binop("ashr", t, a, Value::imm(t, 1));
                    Value x   = B.binop("xor", t, a, sar);
                    std::string intr = sf ? "llvm.ctlz.i64" : "llvm.ctlz.i32";
                    Value clz = B.CreateCall(t, intr, {x, Value::imm(TypeKind::Int1, 0)}, "clz");
                    set_gpr(B, sf, rd, B.binop("sub", t, clz, Value::imm(t, 1)));
                    return true;
                }
                return false;
            }
            if (opcode >= 16 && opcode <= 23) { // CRC32 / CRC32C
                unsigned C = bits(w, 12, 1), sz = bits(w, 10, 2);
                if ((sz == 3) != (sf != 0)) return false;
                Value acc = get_gpr(B, 0, rn);
                Value val = (sz == 3) ? get_gpr(B, 1, rm) : B.cast("zext", get_gpr(B, 0, rm), TypeKind::Int64);
                Value r = B.CreateCall(TypeKind::Int32, "rt_crc32",
                                       {acc, val, Value::imm_i32(1u << sz), Value::imm_i32(C)}, "crc");
                set_gpr(B, 0, rd, r);
                return true;
            }
            Value a = get_gpr(B, sf, rn);
            Value b = get_gpr(B, sf, rm);
            if (opcode == 2) { // UDIV (x / 0 == 0 in ARM64)
                Value is_zero = B.icmp("eq", t, b, Value::imm(t, 0));
                Value safe_b = B.select(t, is_zero, Value::imm(t, 1), b);
                Value q = B.binop("udiv", t, a, safe_b);
                set_gpr(B, sf, rd, B.select(t, is_zero, Value::imm(t, 0), q));
                return true;
            }
            if (opcode == 3) { // SDIV (x / 0 == 0, INT_MIN / -1 == INT_MIN in ARM64)
                Value is_zero = B.icmp("eq", t, b, Value::imm(t, 0));
                Value is_neg1 = B.icmp("eq", t, b, Value::imm(t, ones(size)));
                uint64_t int_min = 1ULL << (size - 1);
                Value is_min  = B.icmp("eq", t, a, Value::imm(t, int_min));
                Value ovf     = B.binop("and", TypeKind::Int1, is_min, is_neg1);
                Value bad     = B.binop("or", TypeKind::Int1, is_zero, ovf);
                Value safe_b  = B.select(t, bad, Value::imm(t, 1), b);
                Value q       = B.binop("sdiv", t, a, safe_b);
                Value res     = B.select(t, is_zero, Value::imm(t, 0), B.select(t, ovf, a, q));
                set_gpr(B, sf, rd, res);
                return true;
            }
            if (opcode >= 8 && opcode <= 11) {
                Value sh = B.binop("and", t, b, Value::imm(t, size - 1));
                if (opcode == 8) { set_gpr(B, sf, rd, B.binop("shl", t, a, sh)); return true; }
                if (opcode == 9) { set_gpr(B, sf, rd, B.binop("lshr", t, a, sh)); return true; }
                if (opcode == 10) { set_gpr(B, sf, rd, B.binop("ashr", t, a, sh)); return true; }
                if (opcode == 11) {
                    std::string intr = sf ? "llvm.fshr.i64" : "llvm.fshr.i32";
                    set_gpr(B, sf, rd, B.CreateCall(t, intr, {a, a, sh}, "rorv"));
                    return true;
                }
            }
        }
        return false;
    }

    // -------------------------------------------------------------------------
    // Vector / SIMD Register Helpers (V0..V31 stored at ArchReg 64..95)
    // -------------------------------------------------------------------------
    Value read_vq(IRBuilder& B, TypeKind vt, unsigned Q, unsigned n) {
        Value v128 = B.read_arch_reg(64 + static_cast<int>(n), TypeKind::Int128);
        if (!Q) {
            Value lo64 = B.cast("trunc", v128, TypeKind::Int64);
            v128 = B.cast("zext", lo64, TypeKind::Int128);
        }
        return (vt == TypeKind::Int128) ? v128 : B.bitcast(v128, vt);
    }

    void write_vq(IRBuilder& B, unsigned Q, unsigned d, const Value& v) {
        Value v128 = (v.type == TypeKind::Int128) ? v : B.bitcast(v, TypeKind::Int128);
        if (!Q) {
            Value lo64 = B.cast("trunc", v128, TypeKind::Int64);
            v128 = B.cast("zext", lo64, TypeKind::Int128);
        }
        B.write_arch_reg(64 + static_cast<int>(d), v128);
    }

    Value emit_shufflevector(IRBuilder& B, TypeKind vt, const Value& a, const Value& b,
                             const std::vector<int>& mask) {
        return B.CreateShuffleVector(vt, a, b, mask, "shuf");
    }

    void emit_sl_gen(IRBuilder& B, int op, int sz, int ne, int d, int szd, int offd, int keep,
                     int n, int szn, int offn, int m = -1, int szm = 0, int offm = 0, int bidx = -1, uint64_t bimm = 0) {
        B.CreateVoidCall("rt_sl_gen", {
            Value::cpu_ptr(),
            Value::imm_i32(op), Value::imm_i32(sz), Value::imm_i32(ne),
            Value::imm_i32(d), Value::imm_i32(szd), Value::imm_i32(offd), Value::imm_i32(keep),
            Value::imm_i32(n), Value::imm_i32(szn), Value::imm_i32(offn),
            Value::imm_i32(m), Value::imm_i32(szm), Value::imm_i32(offm),
            Value::imm_i32(bidx), Value::imm_u64(bimm)
        });
    }

    void emit_same_lop(IRBuilder& B, int op, int sz, int ne, int d, int n, int m = -1, uint64_t bimm = 0, int bidx = -1) {
        emit_sl_gen(B, op, sz, ne, d, sz, 0, 0, n, sz, 0, m, sz, 0, bidx, bimm);
    }

    // -------------------------------------------------------------------------
    // SIMD Loads & Stores
    // -------------------------------------------------------------------------
    bool lift_ldst_simd(IRBuilder& B, uint32_t w, uint64_t pc) {
        unsigned rt = bits(w, 0, 5);
        if (bits(w, 27, 3) == 0b011 && bits(w, 24, 2) == 0) { // Literal
            unsigned opc = bits(w, 30, 2);
            if (opc == 3) return false;
            uint64_t addr = static_cast<uint64_t>(static_cast<int64_t>(pc) + sext(bits(w, 5, 19), 19) * 4);
            TypeKind mt = (opc == 0) ? TypeKind::Int32 : (opc == 1) ? TypeKind::Int64 : TypeKind::Int128;
            B.write_arch_reg(64 + static_cast<int>(rt), B.guest_load(mt, Value::imm_u64(addr), pc));
            return true;
        }
        if (bits(w, 31, 1) == 0 && (bits(w, 24, 6) == 0b001100 || bits(w, 24, 6) == 0b001101)) {
            return lift_ldst_struct(B, w, pc);
        }
        if (bits(w, 27, 3) == 0b101) { // Pair
            unsigned opc = bits(w, 30, 2), mode = bits(w, 23, 2), L = bits(w, 22, 1);
            int64_t imm7 = sext(bits(w, 15, 7), 7);
            unsigned rt2 = bits(w, 10, 5), rn = bits(w, 5, 5);
            if (opc == 3) return false;
            unsigned scale = 2 + opc;
            int64_t off = imm7 << scale;
            Value base = get_gpr(B, 1, rn, true);
            bool post = (mode == 1), wb = (mode == 1 || mode == 3);
            Value off_v = Value::imm_u64(static_cast<uint64_t>(off));
            Value ea = post ? base : (off == 0 ? base : B.binop("add", TypeKind::Int64, base, off_v));
            Value ea2 = B.binop("add", TypeKind::Int64, ea, Value::imm_u64(1ULL << scale));
            TypeKind mt = (scale == 2) ? TypeKind::Int32 : (scale == 3) ? TypeKind::Int64 : TypeKind::Int128;

            if (L) {
                Value a = B.guest_load(mt, ea, pc);
                Value b = B.guest_load(mt, ea2, pc);
                B.write_arch_reg(64 + static_cast<int>(rt), a);
                B.write_arch_reg(64 + static_cast<int>(rt2), b);
            } else {
                Value v1 = B.read_arch_reg(64 + static_cast<int>(rt), TypeKind::Int128);
                Value v2 = B.read_arch_reg(64 + static_cast<int>(rt2), TypeKind::Int128);
                if (mt != TypeKind::Int128) {
                    v1 = B.cast("trunc", v1, mt);
                    v2 = B.cast("trunc", v2, mt);
                }
                B.guest_store(mt, ea, v1, pc);
                B.guest_store(mt, ea2, v2, pc);
            }
            if (wb) {
                set_gpr(B, 1, rn, post ? B.binop("add", TypeKind::Int64, ea, off_v) : ea, true);
            }
            return true;
        }
        if (bits(w, 27, 3) == 0b111) { // Register
            unsigned size = bits(w, 30, 2), opc = bits(w, 22, 2), rn = bits(w, 5, 5);
            unsigned scale = (opc & 2) ? (size ? 99 : 4) : size;
            if (scale > 4) return false;
            unsigned L = opc & 1;
            Value base = get_gpr(B, 1, rn, true);
            Value ea = base;
            std::optional<Value> wb;
            if (bits(w, 24, 2) == 1) {
                uint64_t off = static_cast<uint64_t>(bits(w, 10, 12)) << scale;
                if (off) ea = B.binop("add", TypeKind::Int64, base, Value::imm_u64(off));
            } else if (bits(w, 24, 2) == 0) {
                unsigned kind = bits(w, 10, 2);
                if (bits(w, 21, 1)) {
                    if (kind != 2) return false;
                    unsigned rm = bits(w, 16, 5), option = bits(w, 13, 3), S = bits(w, 12, 1);
                    if (option != 2 && option != 3 && option != 6 && option != 7) return false;
                    ea = B.binop("add", TypeKind::Int64, base, emit_extend(B, rm, option, S ? scale : 0));
                } else {
                    int64_t imm9 = sext(bits(w, 12, 9), 9);
                    Value off_v = Value::imm_u64(static_cast<uint64_t>(imm9));
                    if (kind == 0) {
                        if (imm9) ea = B.binop("add", TypeKind::Int64, base, off_v);
                    } else if (kind == 1) {
                        ea = base;
                        wb = B.binop("add", TypeKind::Int64, base, off_v);
                    } else if (kind == 3) {
                        ea = B.binop("add", TypeKind::Int64, base, off_v);
                        wb = ea;
                    } else return false;
                }
            } else return false;

            TypeKind mt = int_type_for_bits(8u << scale);
            if (L) {
                B.write_arch_reg(64 + static_cast<int>(rt), B.guest_load(mt, ea, pc));
            } else {
                Value v = B.read_arch_reg(64 + static_cast<int>(rt), TypeKind::Int128);
                if (mt != TypeKind::Int128) v = B.cast("trunc", v, mt);
                B.guest_store(mt, ea, v, pc);
            }
            if (wb.has_value()) {
                set_gpr(B, 1, rn, *wb, true);
            }
            return true;
        }
        return false;
    }

    bool lift_ldst_struct(IRBuilder& B, uint32_t w, uint64_t pc) {
        unsigned Q = bits(w, 30, 1), L = bits(w, 22, 1), post = bits(w, 23, 1),
                 rm = bits(w, 16, 5), rn = bits(w, 5, 5), rt = bits(w, 0, 5), sz = bits(w, 10, 2);
        if (!post && rm) return false;
        Value ea = get_gpr(B, 1, rn, true);
        uint64_t total = 0;

        if (bits(w, 24, 1) == 0) { // Multiple structures
            if (bits(w, 21, 1)) return false;
            unsigned opc = bits(w, 12, 4);
            unsigned selem = 0, rpt = 0;
            switch (opc) {
                case 0:  selem = 4; rpt = 1; break;
                case 2:  selem = 1; rpt = 4; break;
                case 4:  selem = 3; rpt = 1; break;
                case 6:  selem = 1; rpt = 3; break;
                case 7:  selem = 1; rpt = 1; break;
                case 8:  selem = 2; rpt = 1; break;
                case 10: selem = 1; rpt = 2; break;
                default: return false;
            }
            if (sz == 3 && !Q && selem > 1) return false;
            total = static_cast<uint64_t>(8u << Q) * selem * rpt;
            if (selem == 1) {
                unsigned step = Q ? 16 : 8;
                TypeKind mt = Q ? TypeKind::Int128 : TypeKind::Int64;
                for (unsigned i = 0; i < rpt; ++i) {
                    unsigned r = (rt + i) % 32;
                    Value cur_ea = (i == 0) ? ea : B.binop("add", TypeKind::Int64, ea, Value::imm_u64(i * step));
                    if (L) {
                        B.write_arch_reg(64 + static_cast<int>(r), B.guest_load(mt, cur_ea, pc));
                    } else {
                        Value v = B.read_arch_reg(64 + static_cast<int>(r), TypeKind::Int128);
                        if (!Q) v = B.cast("trunc", v, TypeKind::Int64);
                        B.guest_store(mt, cur_ea, v, pc);
                    }
                }
            } else {
                std::string fn_name = L ? "rt_sl_ldm" : "rt_sl_stm";
                B.CreateVoidCall(fn_name, {
                    Value::cpu_ptr(), Value::imm_i32(Q), Value::imm_i32(sz),
                    Value::imm_i32(selem), Value::imm_i32(rpt), Value::imm_i32(rt), ea
                });
            }
        } else { // Single structure
            unsigned opc = bits(w, 13, 3), S = bits(w, 12, 1), R = bits(w, 21, 1);
            unsigned scale = opc >> 1;
            unsigned selem = (((opc & 1) << 1) | R) + 1;
            if (scale == 3) {
                if (!L || S) return false;
                total = static_cast<uint64_t>(selem) << sz;
                B.CreateVoidCall("rt_sl_ldr", {
                    Value::cpu_ptr(), Value::imm_i32(Q), Value::imm_i32(sz),
                    Value::imm_i32(selem), Value::imm_i32(rt), ea
                });
            } else {
                unsigned idx = 0;
                if (scale == 0) {
                    idx = (Q << 3) | (S << 2) | sz;
                } else if (scale == 1) {
                    if (sz & 1) return false;
                    idx = (Q << 2) | (S << 1) | (sz >> 1);
                } else {
                    if (sz & 2) return false;
                    if (sz & 1) {
                        if (S) return false;
                        idx = Q;
                        scale = 3;
                    } else {
                        idx = (Q << 1) | S;
                    }
                }
                total = static_cast<uint64_t>(selem) << scale;
                std::string fn_name = L ? "rt_sl_lds" : "rt_sl_sts";
                B.CreateVoidCall(fn_name, {
                    Value::cpu_ptr(), Value::imm_i32(scale), Value::imm_i32(selem),
                    Value::imm_i32(idx), Value::imm_i32(rt), ea
                });
            }
        }

        if (post) {
            Value inc = (rm == 31) ? Value::imm_u64(total) : get_gpr(B, 1, rm);
            set_gpr(B, 1, rn, B.binop("add", TypeKind::Int64, ea, inc), true);
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // FP / AdvSIMD / Crypto Data Processing
    // -------------------------------------------------------------------------
    bool lift_fp_simd(IRBuilder& B, uint32_t w, uint64_t pc) {
        (void)pc;
        unsigned op0 = bits(w, 28, 4), op1 = bits(w, 23, 2), op2 = bits(w, 19, 4), op3 = bits(w, 10, 9);
        if (op0 == 0b0100 && !(op1 & 2) && (op2 & 7) == 5 && (op3 & 0b110000011) == 0b10) return lift_crypto_aes(B, w);
        if (op0 == 0b0101 && !(op1 & 2) && !(op2 & 4) && (op3 & 0b000100011) == 0) return lift_crypto_sha3(B, w);
        if (op0 == 0b0101 && !(op1 & 2) && (op2 & 7) == 5 && (op3 & 0b110000011) == 0b10) return lift_crypto_sha2(B, w);
        if ((op0 & 0b1101) == 0b0101) { // AdvSIMD scalar
            if (op1 == 0 && !(op2 & 0b1100) && (op3 & 0b000100001) == 1) return lift_scalar_copy(B, w);
            if (!(op1 & 2) && (op2 & 7) == 4 && (op3 & 0b110000011) == 0b10) return lift_misc2(B, w, true);
            if (!(op1 & 2) && (op2 & 7) == 6 && (op3 & 0b110000011) == 0b10) return lift_scalar_pairwise(B, w);
            if (!(op1 & 2) && (op2 & 4) && (op3 & 3) == 0) return lift_three_diff(B, w, true);
            if (!(op1 & 2) && (op2 & 4) && (op3 & 1)) return lift_three_same(B, w, true);
            if (op1 == 2 && (op3 & 1)) return lift_shift_imm(B, w, true);
            if ((op1 & 2) && !(op3 & 1)) return lift_indexed(B, w, true);
            return false;
        }
        if ((op0 & 0b1011) == 0 && !(op1 & 2) && !(op2 & 4) && (op3 & 0b000100011) == 0) return lift_tbl(B, w);
        if ((op0 & 0b1011) == 0 && !(op1 & 2) && !(op2 & 4) && (op3 & 0b000100011) == 2) return lift_permute(B, w);
        if ((op0 & 0b1011) == 0b0010 && !(op1 & 2) && !(op2 & 4) && (op3 & 0b000100001) == 0) return lift_ext(B, w);
        if ((op0 & 0b1001) == 0) { // AdvSIMD vector
            if (op1 == 0 && !(op2 & 0b1100) && (op3 & 0b000100001) == 1) return lift_copy(B, w);
            if (!(op1 & 2) && (op2 & 4) && (op3 & 1)) return lift_three_same(B, w, false);
            if (!(op1 & 2) && (op2 & 4) && (op3 & 3) == 0) return lift_three_diff(B, w, false);
            if (!(op1 & 2) && (op2 & 7) == 4 && (op3 & 0b110000011) == 0b10) return lift_misc2(B, w, false);
            if (!(op1 & 2) && (op2 & 7) == 6 && (op3 & 0b110000011) == 0b10) return lift_across(B, w);
            if (op1 == 2 && op2 == 0 && (op3 & 1)) return lift_modimm(B, w);
            if (op1 == 2 && op2 != 0 && (op3 & 1)) return lift_shift_imm(B, w, false);
            if ((op1 & 2) && !(op3 & 1)) return lift_indexed(B, w, false);
            return false;
        }
        if ((op0 & 0b0101) == 0b0001) { // Scalar FP
            if (bits(w, 29, 1) || (bits(w, 31, 1) && ((op1 & 2) || ((op2 & 4) && (op3 & 0b111111))))) return false;
            if (op1 & 2) return lift_fp_dp3(B, w);
            if (!(op2 & 4)) return lift_fp_fixed(B, w);
            if ((op3 & 0b111111) == 0) return lift_fp_int(B, w);
            if ((op3 & 0b11111) == 0b10000) return lift_fp_dp1(B, w);
            if ((op3 & 0b1111) == 0b1000) return lift_fp_cmp(B, w);
            if ((op3 & 0b111) == 0b100) return lift_fp_imm(B, w);
            if ((op3 & 3) == 1) return lift_fp_ccmp(B, w);
            if ((op3 & 3) == 2) return lift_fp_dp2(B, w);
            if ((op3 & 3) == 3) return lift_fp_csel(B, w);
            return false;
        }
        return false;
    }

    bool lift_crypto_aes(IRBuilder& B, uint32_t w) {
        if (bits(w, 22, 2)) return false;
        unsigned opc = bits(w, 12, 5);
        if (opc < 4 || opc > 7) return false;
        B.CreateVoidCall("rt_sv_aes", {
            Value::cpu_ptr(), Value::imm_i32(opc - 4),
            Value::imm_i32(bits(w, 0, 5)), Value::imm_i32(bits(w, 5, 5))
        });
        return true;
    }

    bool lift_crypto_sha3(IRBuilder& B, uint32_t w) {
        if (bits(w, 22, 2)) return false;
        unsigned opc = bits(w, 12, 3), d = bits(w, 0, 5), n = bits(w, 5, 5), m = bits(w, 16, 5);
        if (opc < 3) {
            B.CreateVoidCall("rt_sv_sha1hash", {
                Value::cpu_ptr(), Value::imm_i32(opc),
                Value::imm_i32(d), Value::imm_i32(n), Value::imm_i32(m)
            });
            return true;
        }
        if (opc == 3 || opc == 6) {
            int misc_op = (opc == 3) ? 0 : 3;
            B.CreateVoidCall("rt_sv_sha_misc", {
                Value::cpu_ptr(), Value::imm_i32(misc_op),
                Value::imm_i32(d), Value::imm_i32(n), Value::imm_i32(m)
            });
            return true;
        }
        if (opc == 4 || opc == 5) {
            B.CreateVoidCall("rt_sv_sha256hash", {
                Value::cpu_ptr(), Value::imm_i32(opc == 4 ? 1 : 0),
                Value::imm_i32(d), Value::imm_i32(n), Value::imm_i32(m)
            });
            return true;
        }
        return false;
    }

    bool lift_crypto_sha2(IRBuilder& B, uint32_t w) {
        if (bits(w, 22, 2)) return false;
        unsigned opc = bits(w, 12, 5), d = bits(w, 0, 5), n = bits(w, 5, 5);
        int op = (opc == 0) ? 4 : (opc == 1) ? 1 : (opc == 2) ? 2 : -1;
        if (op < 0) return false;
        B.CreateVoidCall("rt_sv_sha_misc", {
            Value::cpu_ptr(), Value::imm_i32(op),
            Value::imm_i32(d), Value::imm_i32(n), Value::imm_i32(0)
        });
        return true;
    }

    static bool decode_imm5(unsigned imm5, unsigned& sz, unsigned& idx) {
        if ((imm5 & 15) == 0) return false;
        sz = __builtin_ctz(imm5);
        idx = imm5 >> (sz + 1);
        return sz <= 3;
    }

    bool lift_scalar_copy(IRBuilder& B, uint32_t w) {
        if (bits(w, 29, 1) || bits(w, 11, 4)) return false;
        unsigned sz = 0, idx = 0;
        if (!decode_imm5(bits(w, 16, 5), sz, idx)) return false;
        unsigned d = bits(w, 0, 5), n = bits(w, 5, 5);
        TypeKind vt = vec_int_type(sz);
        TypeKind et = int_type_for_bits(8u << sz);
        Value vn = read_vq(B, vt, 1, n);
        Value el = B.CreateExtractElement(vt, et, vn, idx, "el");
        B.write_arch_reg(64 + static_cast<int>(d), el);
        return true;
    }

    bool lift_copy(IRBuilder& B, uint32_t w) {
        unsigned Q = bits(w, 30, 1), op = bits(w, 29, 1), imm5 = bits(w, 16, 5), imm4 = bits(w, 11, 4);
        unsigned d = bits(w, 0, 5), n = bits(w, 5, 5);
        unsigned sz = 0, idx = 0;
        if (!decode_imm5(imm5, sz, idx)) return false;
        TypeKind vt = vec_int_type(sz);
        TypeKind et = int_type_for_bits(8u << sz);
        unsigned nl = 16u >> sz;

        if (op) { // INS (element)
            if (!Q) return false;
            unsigned idx2 = imm4 >> sz;
            Value vd = read_vq(B, vt, 1, d);
            Value vn = read_vq(B, vt, 1, n);
            Value el = B.CreateExtractElement(vt, et, vn, idx2, "el");
            Value ins = B.CreateInsertElement(vt, vd, el, idx, "ins");
            write_vq(B, 1, d, ins);
            return true;
        }
        if (imm4 == 0) { // DUP (element)
            if (sz == 3 && !Q) return false;
            Value vn = read_vq(B, vt, 1, n);
            std::vector<int> mask(nl, static_cast<int>(idx));
            write_vq(B, Q, d, emit_shufflevector(B, vt, vn, vn, mask));
            return true;
        }
        if (imm4 == 1) { // DUP (general)
            if (sz == 3 && !Q) return false;
            Value g = get_gpr(B, 1, n);
            Value el = (et == TypeKind::Int64) ? g : B.cast("trunc", g, et);
            Value ins = B.CreateInsertElement(vt, Value::zero_vec(vt), el, 0, "dup0");
            std::vector<int> mask(nl, 0);
            write_vq(B, Q, d, emit_shufflevector(B, vt, ins, ins, mask));
            return true;
        }
        if (imm4 == 3) { // INS (general)
            if (!Q) return false;
            Value vd = read_vq(B, vt, 1, d);
            Value g = get_gpr(B, 1, n);
            Value el = (et == TypeKind::Int64) ? g : B.cast("trunc", g, et);
            Value ins = B.CreateInsertElement(vt, vd, el, idx, "insg");
            write_vq(B, 1, d, ins);
            return true;
        }
        if (imm4 == 5) { // SMOV
            if (sz >= 2 + Q) return false;
            Value vn = read_vq(B, vt, 1, n);
            Value el = B.CreateExtractElement(vt, et, vn, idx, "smov");
            TypeKind dst_t = Q ? TypeKind::Int64 : TypeKind::Int32;
            set_gpr(B, Q, d, B.sext(el, dst_t));
            return true;
        }
        if (imm4 == 7) { // UMOV
            if ((Q && sz != 3) || (!Q && sz == 3)) return false;
            Value vn = read_vq(B, vt, 1, n);
            Value el = B.CreateExtractElement(vt, et, vn, idx, "umov");
            TypeKind dst_t = Q ? TypeKind::Int64 : TypeKind::Int32;
            Value ext = (et == dst_t) ? el : B.cast("zext", el, dst_t);
            set_gpr(B, Q, d, ext);
            return true;
        }
        return false;
    }

    bool lift_permute(IRBuilder& B, uint32_t w) {
        unsigned Q = bits(w, 30, 1), sz = bits(w, 22, 2), m = bits(w, 16, 5), opc = bits(w, 12, 3),
                 n = bits(w, 5, 5), d = bits(w, 0, 5);
        if (opc == 0 || opc == 4 || (sz == 3 && !Q)) return false;
        unsigned ne = (8u << Q) >> sz, NL = 16u >> sz;
        unsigned part = opc >> 2, kind = opc & 3;
        std::vector<int> idx;
        for (unsigned i = 0; i < ne; ++i) {
            if (kind == 1) { // UZP
                unsigned j = 2 * i + part;
                idx.push_back(static_cast<int>(j < ne ? j : NL + j - ne));
            } else if (kind == 2) { // TRN
                unsigned p = i & ~1u;
                idx.push_back(static_cast<int>(!(i & 1) ? (p + part) : (NL + p + part)));
            } else { // ZIP
                unsigned base = part * ne / 2;
                idx.push_back(static_cast<int>(!(i & 1) ? (base + i / 2) : (NL + base + i / 2)));
            }
        }
        while (idx.size() < NL) idx.push_back(0);
        TypeKind vt = vec_int_type(sz);
        write_vq(B, Q, d, emit_shufflevector(B, vt, read_vq(B, vt, 1, n), read_vq(B, vt, 1, m), idx));
        return true;
    }

    bool lift_ext(IRBuilder& B, uint32_t w) {
        unsigned Q = bits(w, 30, 1), m = bits(w, 16, 5), imm4 = bits(w, 11, 4), n = bits(w, 5, 5), d = bits(w, 0, 5);
        if (bits(w, 22, 2) || (!Q && imm4 >= 8)) return false;
        std::vector<int> idx;
        if (Q) {
            for (int i = 0; i < 16; ++i) idx.push_back(static_cast<int>(imm4) + i);
        } else {
            for (int i = 0; i < 8; ++i) {
                int k = static_cast<int>(imm4) + i;
                idx.push_back(k < 8 ? k : 16 + k - 8);
            }
            for (int i = 0; i < 8; ++i) idx.push_back(0);
        }
        TypeKind vt = TypeKind::Vec16xI8;
        write_vq(B, Q, d, emit_shufflevector(B, vt, read_vq(B, vt, 1, n), read_vq(B, vt, 1, m), idx));
        return true;
    }

    bool lift_tbl(IRBuilder& B, uint32_t w) {
        unsigned Q = bits(w, 30, 1), m = bits(w, 16, 5), ln = bits(w, 13, 2), op = bits(w, 12, 1),
                 n = bits(w, 5, 5), d = bits(w, 0, 5);
        if (bits(w, 22, 2)) return false;
        B.CreateVoidCall("rt_sv_tbl", {
            Value::cpu_ptr(), Value::imm_i32(Q), Value::imm_i32(ln),
            Value::imm_i32(op), Value::imm_i32(d), Value::imm_i32(n), Value::imm_i32(m)
        });
        return true;
    }

    static std::pair<std::vector<int>, std::vector<int>> pair_indices(unsigned Q, unsigned sz) {
        unsigned ne = (8u << Q) >> sz, NL = 16u >> sz;
        std::vector<int> src;
        for (unsigned i = 0; i < ne; ++i) src.push_back(static_cast<int>(i));
        for (unsigned i = 0; i < ne; ++i) src.push_back(static_cast<int>(NL + i));
        std::vector<int> ev, od;
        for (unsigned k = 0; k < ne; ++k) {
            ev.push_back(src[2 * k]);
            od.push_back(src[2 * k + 1]);
        }
        while (ev.size() < NL) {
            ev.push_back(static_cast<int>(ne % NL));
            od.push_back(static_cast<int>(ne % NL));
        }
        return {ev, od};
    }

    Value emit_vec_intrinsic2(IRBuilder& B, const char* base_name, TypeKind vt, const Value& a, const Value& b) {
        const char* sfx = (vt == TypeKind::Vec16xI8) ? ".v16i8"
                        : (vt == TypeKind::Vec8xI16) ? ".v8i16"
                        : (vt == TypeKind::Vec4xI32) ? ".v4i32"
                        : ".v2i64";
        std::string intr = std::string("llvm.") + base_name + sfx;
        return B.CreateCall(vt, intr, {a, b}, "vi");
    }

    bool lift_three_same(IRBuilder& B, uint32_t w, bool scalar) {
        unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), sz = bits(w, 22, 2), m = bits(w, 16, 5),
                 opc = bits(w, 11, 5), n = bits(w, 5, 5), d = bits(w, 0, 5);
        if (opc >= 0b11000) return lift_three_same_fp(B, w, scalar);
        if (scalar) return lift_three_same_scalar(B, U, sz, opc, d, n, m);
        if (sz == 3 && !Q && opc != 3) return false;
        unsigned ne = (8u << Q) >> sz;
        TypeKind vt = vec_int_type(sz);

        if ((opc == 0 || opc == 2 || opc == 4 || opc == 14 || opc == 15 || opc == 12 || opc == 13) && sz == 3) {
            return false;
        }
        if (opc == 0) { emit_same_lop(B, LOP_HADD | (U ? LOP_U : 0), sz, ne, d, n, m); return true; }
        if (opc == 1) {
            Value a = read_vq(B, vt, 1, n), b = read_vq(B, vt, 1, m);
            write_vq(B, Q, d, emit_vec_intrinsic2(B, U ? "uadd.sat" : "sadd.sat", vt, a, b));
            return true;
        }
        if (opc == 2) { emit_same_lop(B, LOP_RHADD | (U ? LOP_U : 0), sz, ne, d, n, m); return true; }
        if (opc == 3) { // Bitwise AND/BIC/ORR/ORN/EOR/BSL/BIT/BIF
            TypeKind v64 = TypeKind::Vec2xI64;
            Value a = read_vq(B, v64, 1, n), b = read_vq(B, v64, 1, m);
            Value all1 = B.bitcast(Value::imm_i128(~0ULL, ~0ULL), v64);
            Value nb = B.binop("xor", v64, b, all1);
            Value res;
            if (!U) {
                if (sz == 0) res = B.binop("and", v64, a, b);
                else if (sz == 1) res = B.binop("and", v64, a, nb);
                else if (sz == 2) res = B.binop("or", v64, a, b);
                else res = B.binop("or", v64, a, nb);
            } else {
                if (sz == 0) res = B.binop("xor", v64, a, b);
                else {
                    Value o = read_vq(B, v64, 1, d);
                    Value no = B.binop("xor", v64, o, all1);
                    if (sz == 1) res = B.binop("or", v64, B.binop("and", v64, o, a), B.binop("and", v64, no, b));
                    else if (sz == 2) res = B.binop("or", v64, B.binop("and", v64, o, nb), B.binop("and", v64, a, b));
                    else res = B.binop("or", v64, B.binop("and", v64, o, b), B.binop("and", v64, a, nb));
                }
            }
            write_vq(B, Q, d, res);
            return true;
        }
        if (opc == 4) { emit_same_lop(B, LOP_HSUB | (U ? LOP_U : 0), sz, ne, d, n, m); return true; }
        if (opc == 5) {
            Value a = read_vq(B, vt, 1, n), b = read_vq(B, vt, 1, m);
            write_vq(B, Q, d, emit_vec_intrinsic2(B, U ? "usub.sat" : "ssub.sat", vt, a, b));
            return true;
        }
        if (opc == 6 || opc == 7) { // CMGT / CMGE / CMHI / CMHS
            Value a = read_vq(B, vt, 1, n), b = read_vq(B, vt, 1, m);
            const char* pred = (opc == 6) ? (U ? "ugt" : "sgt") : (U ? "uge" : "sge");
            write_vq(B, Q, d, B.CreateVecICmpSExt(pred, vt, a, b));
            return true;
        }
        if (opc >= 8 && opc <= 11) {
            static const int ops[4] = {LOP_SHL, LOP_QSHL, LOP_RSHL, LOP_QRSHL};
            emit_same_lop(B, ops[opc - 8] | (U ? LOP_U : 0), sz, ne, d, n, m);
            return true;
        }
        if (opc == 12 || opc == 13) {
            Value a = read_vq(B, vt, 1, n), b = read_vq(B, vt, 1, m);
            const char* intr = (opc == 12) ? (U ? "umax" : "smax") : (U ? "umin" : "smin");
            write_vq(B, Q, d, emit_vec_intrinsic2(B, intr, vt, a, b));
            return true;
        }
        if (opc == 14 || opc == 15) {
            Value a = read_vq(B, vt, 1, n), b = read_vq(B, vt, 1, m);
            Value mx = emit_vec_intrinsic2(B, U ? "umax" : "smax", vt, a, b);
            Value mn = emit_vec_intrinsic2(B, U ? "umin" : "smin", vt, a, b);
            Value diff = B.binop("sub", vt, mx, mn);
            if (opc == 15) diff = B.binop("add", vt, read_vq(B, vt, 1, d), diff);
            write_vq(B, Q, d, diff);
            return true;
        }
        if (opc == 16) {
            Value a = read_vq(B, vt, 1, n), b = read_vq(B, vt, 1, m);
            write_vq(B, Q, d, B.binop(U ? "sub" : "add", vt, a, b));
            return true;
        }
        if (opc == 17) { // CMEQ / CMTST
            Value a = read_vq(B, vt, 1, n), b = read_vq(B, vt, 1, m);
            if (U) {
                write_vq(B, Q, d, B.CreateVecICmpSExt("eq", vt, a, b));
            } else {
                Value ab = B.binop("and", vt, a, b);
                write_vq(B, Q, d, B.CreateVecICmpSExt("ne", vt, ab, Value::zero_vec(vt)));
            }
            return true;
        }
        if (sz == 3 && opc != 23) return false;
        if (opc == 18) { // MLA / MLS
            Value a = read_vq(B, vt, 1, n), b = read_vq(B, vt, 1, m), o = read_vq(B, vt, 1, d);
            Value prod = B.binop("mul", vt, a, b);
            write_vq(B, Q, d, B.binop(U ? "sub" : "add", vt, o, prod));
            return true;
        }
        if (opc == 19) { // MUL / PMUL
            if (U) {
                if (sz) return false;
                emit_same_lop(B, LOP_PMUL, 0, ne, d, n, m);
                return true;
            }
            write_vq(B, Q, d, B.binop("mul", vt, read_vq(B, vt, 1, n), read_vq(B, vt, 1, m)));
            return true;
        }
        if (opc == 20 || opc == 21 || opc == 23) { // SMAXP/UMAXP/SMINP/UMINP/ADDP
            if (opc == 23 && U) return false;
            auto [ev, od] = pair_indices(Q, sz);
            Value a = read_vq(B, vt, 1, n), b = read_vq(B, vt, 1, m);
            Value L = emit_shufflevector(B, vt, a, b, ev);
            Value R = emit_shufflevector(B, vt, a, b, od);
            if (opc == 23) {
                write_vq(B, Q, d, B.binop("add", vt, L, R));
            } else {
                const char* intr = (opc == 20) ? (U ? "umax" : "smax") : (U ? "umin" : "smin");
                write_vq(B, Q, d, emit_vec_intrinsic2(B, intr, vt, L, R));
            }
            return true;
        }
        if (opc == 22) {
            if (sz != 1 && sz != 2) return false;
            emit_same_lop(B, U ? LOP_SQRDMULH : LOP_SQDMULH, sz, ne, d, n, m);
            return true;
        }
        return false;
    }

    bool lift_three_same_scalar(IRBuilder& B, unsigned U, unsigned sz, unsigned opc, unsigned d, unsigned n, unsigned m) {
        if (opc == 1 || opc == 5) {
            int op = (opc == 1) ? (U ? LOP_UQADD : LOP_SQADD) : (U ? LOP_UQSUB : LOP_SQSUB);
            emit_same_lop(B, op, sz, 1, d, n, m);
            return true;
        }
        if (opc == 9 || opc == 11) {
            emit_same_lop(B, (opc == 9 ? LOP_QSHL : LOP_QRSHL) | (U ? LOP_U : 0), sz, 1, d, n, m);
            return true;
        }
        if (opc == 22) {
            if (sz != 1 && sz != 2) return false;
            emit_same_lop(B, U ? LOP_SQRDMULH : LOP_SQDMULH, sz, 1, d, n, m);
            return true;
        }
        if (sz != 3) return false;
        Value a = B.cast("trunc", B.read_arch_reg(64 + static_cast<int>(n), TypeKind::Int128), TypeKind::Int64);
        Value b = B.cast("trunc", B.read_arch_reg(64 + static_cast<int>(m), TypeKind::Int128), TypeKind::Int64);
        if (opc == 6 || opc == 7) {
            const char* pred = (opc == 6) ? (U ? "ugt" : "sgt") : (U ? "uge" : "sge");
            Value c_i1 = B.icmp(pred, TypeKind::Int64, a, b);
            B.write_arch_reg(64 + static_cast<int>(d), B.sext(c_i1, TypeKind::Int64));
            return true;
        }
        if (opc == 8 || opc == 10) {
            emit_same_lop(B, (opc == 8 ? LOP_SHL : LOP_RSHL) | (U ? LOP_U : 0), 3, 1, d, n, m);
            return true;
        }
        if (opc == 16) {
            B.write_arch_reg(64 + static_cast<int>(d), B.binop(U ? "sub" : "add", TypeKind::Int64, a, b));
            return true;
        }
        if (opc == 17) {
            Value c_i1 = U ? B.icmp("eq", TypeKind::Int64, a, b)
                           : B.icmp("ne", TypeKind::Int64, B.binop("and", TypeKind::Int64, a, b), Value::imm_u64(0));
            B.write_arch_reg(64 + static_cast<int>(d), B.sext(c_i1, TypeKind::Int64));
            return true;
        }
        return false;
    }

    bool lift_three_same_fp(IRBuilder& B, uint32_t w, bool scalar) {
        unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), a_ = bits(w, 23, 1), sz = bits(w, 22, 1),
                 m = bits(w, 16, 5), opc = bits(w, 11, 5), n = bits(w, 5, 5), d = bits(w, 0, 5);
        if (!scalar && sz && !Q) return false;
        // FCMP3: {(0, 0, 0b11100): 0, (1, 0, 0b11100): 1, (1, 1, 0b11100): 2, (1, 0, 0b11101): 3, (1, 1, 0b11101): 4}
        int cop = -1;
        if (opc == 0b11100 && !U && !a_) cop = 0;
        else if (opc == 0b11100 && U && !a_) cop = 1;
        else if (opc == 0b11100 && U && a_) cop = 2;
        else if (opc == 0b11101 && U && !a_) cop = 3;
        else if (opc == 0b11101 && U && a_) cop = 4;
        if (cop >= 0) {
            B.CreateVoidCall("rt_vec_fcmp", {
                Value::cpu_ptr(), Value::imm_i32(sz), Value::imm_i32(cop),
                Value::imm_i32(Q), Value::imm_i32(scalar ? 1 : 0),
                Value::imm_i32(d), Value::imm_i32(n), Value::imm_i32(m),
                Value::imm_i32(0), Value::imm_i32(0)
            });
            return true;
        }

        auto map_fop3 = [&]() -> int {
            if (!U && !a_ && opc == 0b11000) return FOP_MAXNM;
            if (!U && !a_ && opc == 0b11010) return FOP_ADD;
            if (!U && !a_ && opc == 0b11011) return FOP_MULX;
            if (!U && !a_ && opc == 0b11110) return FOP_MAX;
            if (!U && !a_ && opc == 0b11111) return FOP_RECPS;
            if (!U &&  a_ && opc == 0b11000) return FOP_MINNM;
            if (!U &&  a_ && opc == 0b11010) return FOP_SUB;
            if (!U &&  a_ && opc == 0b11110) return FOP_MIN;
            if (!U &&  a_ && opc == 0b11111) return FOP_RSQRTS;
            if ( U && !a_ && opc == 0b11011) return FOP_MUL;
            if ( U && !a_ && opc == 0b11111) return FOP_DIV;
            if ( U &&  a_ && opc == 0b11010) return FOP_ABD;
            return -1;
        };

        if (scalar) {
            int fop = map_fop3();
            if (fop != FOP_MULX && fop != FOP_RECPS && fop != FOP_RSQRTS && fop != FOP_ABD) return false;
            int neg_n = (fop == FOP_RECPS || fop == FOP_RSQRTS) ? 1 : 0;
            B.CreateVoidCall("rt_vec_fop", {
                Value::cpu_ptr(), Value::imm_i32(sz), Value::imm_i32(fop),
                Value::imm_i32(0), Value::imm_i32(d), Value::imm_i32(n),
                Value::imm_i32(m), Value::imm_i32(-1), Value::imm_i32(neg_n)
            });
            return true;
        }

        // Pairwise FP
        int pair_op = -1;
        if (U && !a_ && opc == 0b11000) pair_op = FOP_MAXNM;
        else if (U && a_ && opc == 0b11000) pair_op = FOP_MINNM;
        else if (U && !a_ && opc == 0b11010) pair_op = FOP_ADD;
        else if (U && !a_ && opc == 0b11110) pair_op = FOP_MAX;
        else if (U && a_ && opc == 0b11110) pair_op = FOP_MIN;
        if (pair_op >= 0) {
            B.CreateVoidCall("rt_vec_fpair", {
                Value::cpu_ptr(), Value::imm_i32(sz), Value::imm_i32(pair_op),
                Value::imm_i32(Q), Value::imm_i32(d), Value::imm_i32(n), Value::imm_i32(m)
            });
            return true;
        }

        if (!U && opc == 0b11001) { // FMLA / FMLS
            B.CreateVoidCall("rt_vec_fop", {
                Value::cpu_ptr(), Value::imm_i32(sz), Value::imm_i32(FOP_MADD),
                Value::imm_i32(Q), Value::imm_i32(d), Value::imm_i32(n),
                Value::imm_i32(m), Value::imm_i32(d), Value::imm_i32(a_)
            });
            return true;
        }

        int fop = map_fop3();
        if (fop < 0) return false;
        int neg_n = (fop == FOP_RECPS || fop == FOP_RSQRTS) ? 1 : 0;
        B.CreateVoidCall("rt_vec_fop", {
            Value::cpu_ptr(), Value::imm_i32(sz), Value::imm_i32(fop),
            Value::imm_i32(Q), Value::imm_i32(d), Value::imm_i32(n),
            Value::imm_i32(m), Value::imm_i32(-1), Value::imm_i32(neg_n)
        });
        return true;
    }

    bool lift_three_diff(IRBuilder& B, uint32_t w, bool scalar) {
        unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), sz = bits(w, 22, 2), m = bits(w, 16, 5),
                 opc = bits(w, 12, 4), n = bits(w, 5, 5), d = bits(w, 0, 5);
        if (scalar) {
            if (U || (opc != 9 && opc != 11 && opc != 13) || (sz != 1 && sz != 2)) return false;
            int op = (opc == 9) ? LOP_SQDMLAL : (opc == 11) ? LOP_SQDMLSL : LOP_SQDMULL;
            emit_sl_gen(B, op, sz, 1, d, sz + 1, 0, 0, n, sz, 0, m, sz, 0);
            return true;
        }
        if (opc == 14) { // PMULL
            if (U) return false;
            if (sz == 3) {
                B.CreateVoidCall("rt_sl_pmull64", {
                    Value::cpu_ptr(), Value::imm_i32(Q),
                    Value::imm_i32(d), Value::imm_i32(n), Value::imm_i32(m)
                });
                return true;
            }
            if (sz) return false;
        }
        if (sz == 3 || opc == 15) return false;
        int ne = 8 >> sz;
        int off = static_cast<int>(Q) * ne;
        if (opc == 4 || opc == 6) {
            static const int ops[4] = {LOP_ADDHN, LOP_RADDHN, LOP_SUBHN, LOP_RSUBHN};
            int op = ops[(opc == 6 ? 2 : 0) + U];
            emit_sl_gen(B, op, sz, ne, d, sz, off, Q, n, sz + 1, 0, m, sz + 1, 0);
            return true;
        }
        if (opc == 1 || opc == 3) {
            int op = ((opc == 1) ? LOP_ADDW : LOP_SUBW) | (U ? LOP_U : 0);
            emit_sl_gen(B, op, sz, ne, d, sz + 1, 0, 0, n, sz + 1, 0, m, sz, off);
            return true;
        }
        int op = -1;
        switch (opc) {
            case 0:  op = LOP_ADDL | (U ? LOP_U : 0); break;
            case 2:  op = LOP_SUBL | (U ? LOP_U : 0); break;
            case 5:  op = LOP_ABAL | (U ? LOP_U : 0); break;
            case 7:  op = LOP_ABDL | (U ? LOP_U : 0); break;
            case 8:  op = LOP_MLAL | (U ? LOP_U : 0); break;
            case 10: op = LOP_MLSL | (U ? LOP_U : 0); break;
            case 12: op = LOP_MULL | (U ? LOP_U : 0); break;
            case 9:  if (U || (sz != 1 && sz != 2)) return false; op = LOP_SQDMLAL; break;
            case 11: if (U || (sz != 1 && sz != 2)) return false; op = LOP_SQDMLSL; break;
            case 13: if (U || (sz != 1 && sz != 2)) return false; op = LOP_SQDMULL; break;
            case 14: op = LOP_PMULL; break;
            default: return false;
        }
        emit_sl_gen(B, op, sz, ne, d, sz + 1, 0, 0, n, sz, off, m, sz, off);
        return true;
    }

    bool lift_misc2(IRBuilder& B, uint32_t w, bool scalar) {
        unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), sz = bits(w, 22, 2), opc = bits(w, 12, 5),
                 n = bits(w, 5, 5), d = bits(w, 0, 5);
        if (opc >= 0b11000 || opc == 0b10110 || opc == 0b10111 || (opc >= 0b01100 && opc <= 0b01111)) {
            return lift_misc2_fp(B, w, scalar);
        }
        if (scalar) return lift_misc2_scalar(B, U, sz, opc, d, n);
        unsigned ne = (8u << Q) >> sz;
        TypeKind vt = vec_int_type(sz);

        if (opc == 0 || (opc == 1 && !U)) { // REV64 / REV32 / REV16
            unsigned gsz = (U == 0 && opc == 0) ? 3 : (U == 1 && opc == 0) ? 2 : (U == 0 && opc == 1) ? 1 : 0;
            if (!gsz || sz >= gsz) return false;
            unsigned es = 1u << sz, gb = 1u << gsz;
            std::vector<int> idx;
            for (unsigned i = 0; i < 16; ++i) {
                unsigned g = (i / gb) * gb, o = i % gb;
                unsigned e = o / es, b = o % es;
                idx.push_back(static_cast<int>(g + (gb / es - 1 - e) * es + b));
            }
            Value vn = read_vq(B, TypeKind::Vec16xI8, 1, n);
            write_vq(B, Q, d, emit_shufflevector(B, TypeKind::Vec16xI8, vn, vn, idx));
            return true;
        }
        if (sz == 3 && !Q && opc != 0b10010 && opc != 0b10100 && opc != 0b10011) return false;
        if (opc == 2 || opc == 6) {
            if (sz == 3) return false;
            B.CreateVoidCall("rt_sl_addlp", {
                Value::cpu_ptr(), Value::imm_i32(U), Value::imm_i32(opc == 6 ? 1 : 0),
                Value::imm_i32(Q), Value::imm_i32(sz), Value::imm_i32(d), Value::imm_i32(n)
            });
            return true;
        }
        if (opc == 3) { emit_same_lop(B, U ? LOP_USQADD : LOP_SUQADD, sz, ne, d, n); return true; }
        if (opc == 4) {
            if (sz == 3) return false;
            emit_same_lop(B, U ? LOP_CLZ : LOP_CLS, sz, ne, d, n);
            return true;
        }
        if (opc == 5) {
            if (!U) {
                if (sz) return false;
                emit_same_lop(B, LOP_CNT, 0, ne, d, n);
                return true;
            }
            if (sz == 0) {
                Value vn = read_vq(B, TypeKind::Vec2xI64, 1, n);
                Value all1 = B.bitcast(Value::imm_i128(~0ULL, ~0ULL), TypeKind::Vec2xI64);
                write_vq(B, Q, d, B.binop("xor", TypeKind::Vec2xI64, vn, all1));
                return true;
            }
            if (sz == 1) {
                emit_same_lop(B, LOP_RBIT, 0, 8u << Q, d, n);
                return true;
            }
            return false;
        }
        if (opc == 7) { emit_same_lop(B, U ? LOP_SQNEG : LOP_SQABS, sz, ne, d, n); return true; }
        if (opc == 8 || opc == 9 || (opc == 10 && !U)) {
            Value a = read_vq(B, vt, 1, n);
            const char* pred = (opc == 8) ? (U ? "sge" : "sgt") : (opc == 9) ? (U ? "sle" : "eq") : "slt";
            write_vq(B, Q, d, B.CreateVecICmpSExt(pred, vt, a, Value::zero_vec(vt)));
            return true;
        }
        if (opc == 11) { // NEG / ABS
            Value a = read_vq(B, vt, 1, n);
            Value neg = B.binop("sub", vt, Value::zero_vec(vt), a);
            if (U) {
                write_vq(B, Q, d, neg);
            } else {
                write_vq(B, Q, d, B.CreateVecAbs(vt, a, neg));
            }
            return true;
        }
        if (opc == 0b10010 || opc == 0b10100) { // XTN / SQXTUN / SQXTN / UQXTN
            if (sz == 3) return false;
            int op = (opc == 0b10010) ? (U ? LOP_SQXTUN : LOP_XTN) : (U ? LOP_UQXTN : LOP_SQXTN);
            int nlanes = 8 >> sz;
            emit_sl_gen(B, op, sz, nlanes, d, sz, static_cast<int>(Q) * nlanes, Q, n, sz + 1, 0);
            return true;
        }
        if (opc == 0b10011 && U) { // SHLL
            if (sz == 3) return false;
            int nlanes = 8 >> sz;
            emit_sl_gen(B, LOP_SHLL, sz, nlanes, d, sz + 1, 0, 0, n, sz, static_cast<int>(Q) * nlanes);
            return true;
        }
        return false;
    }

    bool lift_misc2_scalar(IRBuilder& B, unsigned U, unsigned sz, unsigned opc, unsigned d, unsigned n) {
        if (opc == 3) { emit_same_lop(B, U ? LOP_USQADD : LOP_SUQADD, sz, 1, d, n); return true; }
        if (opc == 7) { emit_same_lop(B, U ? LOP_SQNEG : LOP_SQABS, sz, 1, d, n); return true; }
        if (opc == 0b10010 || opc == 0b10100) {
            if (sz == 3 || (opc == 0b10010 && !U)) return false;
            int op = (opc == 0b10010) ? LOP_SQXTUN : (U ? LOP_UQXTN : LOP_SQXTN);
            emit_sl_gen(B, op, sz, 1, d, sz, 0, 0, n, sz + 1, 0);
            return true;
        }
        if (sz != 3) return false;
        Value a = B.cast("trunc", B.read_arch_reg(64 + static_cast<int>(n), TypeKind::Int128), TypeKind::Int64);
        if (opc == 8 || opc == 9 || (opc == 10 && !U)) {
            const char* pred = (opc == 8) ? (U ? "sge" : "sgt") : (opc == 9) ? (U ? "sle" : "eq") : "slt";
            Value c_i1 = B.icmp(pred, TypeKind::Int64, a, Value::imm_u64(0));
            B.write_arch_reg(64 + static_cast<int>(d), B.sext(c_i1, TypeKind::Int64));
            return true;
        }
        if (opc == 11) {
            Value neg = B.binop("sub", TypeKind::Int64, Value::imm_u64(0), a);
            if (U) {
                B.write_arch_reg(64 + static_cast<int>(d), neg);
            } else {
                Value is_neg = B.icmp("slt", TypeKind::Int64, a, Value::imm_u64(0));
                B.write_arch_reg(64 + static_cast<int>(d), B.select(TypeKind::Int64, is_neg, neg, a));
            }
            return true;
        }
        return false;
    }

    bool lift_misc2_fp(IRBuilder& B, uint32_t w, bool scalar) {
        unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), a_ = bits(w, 23, 1), sz = bits(w, 22, 1),
                 opc = bits(w, 12, 5), n = bits(w, 5, 5), d = bits(w, 0, 5);
        if (opc == 0b10110 || opc == 0b10111) { // FCVTN / FCVTL / FCVTXN
            if (a_) return false;
            if (opc == 0b10110 && U) {
                if (!sz) return false;
                int part = scalar ? -1 : static_cast<int>(Q);
                B.CreateVoidCall("rt_vec_fcvtn", {
                    Value::cpu_ptr(), Value::imm_i32(1), Value::imm_i32(part),
                    Value::imm_i32(1), Value::imm_i32(d), Value::imm_i32(n)
                });
                return true;
            }
            if (scalar || U) return false;
            if (opc == 0b10110) {
                B.CreateVoidCall("rt_vec_fcvtn", {
                    Value::cpu_ptr(), Value::imm_i32(sz), Value::imm_i32(Q),
                    Value::imm_i32(0), Value::imm_i32(d), Value::imm_i32(n)
                });
            } else {
                B.CreateVoidCall("rt_vec_fcvtl", {
                    Value::cpu_ptr(), Value::imm_i32(sz), Value::imm_i32(Q),
                    Value::imm_i32(d), Value::imm_i32(n)
                });
            }
            return true;
        }
        if (!scalar && sz && !Q) return false;
        if (opc >= 0b01100 && opc <= 0b01111) {
            if (!a_) return false;
            if (opc == 0b01111) { // FABS / FNEG (vector)
                if (scalar) return false;
                TypeKind vt = sz ? TypeKind::Vec2xI64 : TypeKind::Vec4xI32;
                uint64_t sign_m = sz ? 0x8000000000000000ULL : 0x8000000032ULL;
                if (!sz) sign_m = 0x80000000ULL;
                uint64_t lo = sz ? sign_m : ((sign_m << 32) | sign_m);
                Value mask_v = B.bitcast(Value::imm_i128(lo, lo), vt);
                Value vn = read_vq(B, vt, 1, n);
                if (U) {
                    write_vq(B, Q, d, B.binop("xor", vt, vn, mask_v));
                } else {
                    Value nmask = B.bitcast(Value::imm_i128(~lo, ~lo), vt);
                    write_vq(B, Q, d, B.binop("and", vt, vn, nmask));
                }
                return true;
            }
            int cop = -1, swap = 0;
            if (opc == 0b01100 && !U) { cop = 2; swap = 0; }
            else if (opc == 0b01100 && U) { cop = 1; swap = 0; }
            else if (opc == 0b01101 && !U) { cop = 0; swap = 0; }
            else if (opc == 0b01101 && U) { cop = 1; swap = 1; }
            else if (opc == 0b01110 && !U) { cop = 2; swap = 1; }
            else return false;
            B.CreateVoidCall("rt_vec_fcmp", {
                Value::cpu_ptr(), Value::imm_i32(sz), Value::imm_i32(cop),
                Value::imm_i32(Q), Value::imm_i32(scalar ? 1 : 0),
                Value::imm_i32(d), Value::imm_i32(n), Value::imm_i32(0),
                Value::imm_i32(1), Value::imm_i32(swap)
            });
            return true;
        }
        if (opc == 0b11111) {
            if (!a_) return false;
            if (U) {
                if (scalar) return false;
                B.CreateVoidCall("rt_vec_fop", {
                    Value::cpu_ptr(), Value::imm_i32(sz), Value::imm_i32(FOP_SQRT),
                    Value::imm_i32(Q), Value::imm_i32(d), Value::imm_i32(n),
                    Value::imm_i32(n), Value::imm_i32(-1), Value::imm_i32(0)
                });
                return true;
            }
            if (!scalar) return false;
            B.CreateVoidCall("rt_fp_recpx", {
                Value::cpu_ptr(), Value::imm_i32(sz), Value::imm_i32(d), Value::imm_i32(n)
            });
            return true;
        }
        if (opc == 0b11000 || opc == 0b11001) { // FRINT*
            if (scalar) return false;
            int rm = -1;
            unsigned o0 = opc & 1;
            if (!U && !a_ && !o0) rm = 0;
            else if (!U && !a_ && o0) rm = 2;
            else if (!U && a_ && !o0) rm = 1;
            else if (!U && a_ && o0) rm = 3;
            else if (U && !a_ && !o0) rm = 4;
            else if (U && !a_ && o0) rm = 5;
            else if (U && a_ && o0) rm = 6;
            if (rm < 0) return false;
            B.CreateVoidCall("rt_vec_frint", {
                Value::cpu_ptr(), Value::imm_i32(sz), Value::imm_i32(Q),
                Value::imm_i32(rm), Value::imm_i32(d), Value::imm_i32(n)
            });
            return true;
        }
        if (opc == 0b11010 || opc == 0b11011 || (opc == 0b11100 && !a_)) { // FCVT[NPMZA][SU]
            int rm = (opc == 0b11010 && !a_) ? 0 : (opc == 0b11010 && a_) ? 1
                   : (opc == 0b11011 && !a_) ? 2 : (opc == 0b11011 && a_) ? 3 : 4;
            B.CreateVoidCall("rt_vec_fcvt_int", {
                Value::cpu_ptr(), Value::imm_i32(sz), Value::imm_i32(1),
                Value::imm_i32(Q), Value::imm_i32(scalar ? 1 : 0), Value::imm_i32(rm),
                Value::imm_i32(0), Value::imm_i32(U), Value::imm_i32(d), Value::imm_i32(n)
            });
            return true;
        }
        if (opc == 0b11100) { // URECPE / URSQRTE
            if (scalar || sz) return false;
            emit_same_lop(B, U ? LOP_URSQRTE : LOP_URECPE, 2, (8u << Q) >> 2, d, n);
            return true;
        }
        if (opc == 0b11101) {
            if (!a_) { // SCVTF / UCVTF
                B.CreateVoidCall("rt_vec_fcvt_int", {
                    Value::cpu_ptr(), Value::imm_i32(sz), Value::imm_i32(0),
                    Value::imm_i32(Q), Value::imm_i32(scalar ? 1 : 0), Value::imm_i32(0),
                    Value::imm_i32(0), Value::imm_i32(U), Value::imm_i32(d), Value::imm_i32(n)
                });
                return true;
            }
            if (scalar) {
                B.CreateVoidCall("rt_fp_recpe", {
                    Value::cpu_ptr(), Value::imm_i32(sz), Value::imm_i32(d),
                    Value::imm_i32(n), Value::imm_i32(U)
                });
            } else {
                B.CreateVoidCall("rt_vec_frecpe", {
                    Value::cpu_ptr(), Value::imm_i32(sz), Value::imm_i32(Q),
                    Value::imm_i32(U), Value::imm_i32(d), Value::imm_i32(n)
                });
            }
            return true;
        }
        return false;
    }

    bool lift_across(IRBuilder& B, uint32_t w) {
        unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), sz = bits(w, 22, 2), opc = bits(w, 12, 5),
                 n = bits(w, 5, 5), d = bits(w, 0, 5);
        if (opc == 0b01100 || opc == 0b01111) {
            if (!U || !Q || (sz & 1)) return false;
            int fop = (opc == 0b01100) ? ((sz >> 1) ? FOP_MINNM : FOP_MAXNM) : ((sz >> 1) ? FOP_MIN : FOP_MAX);
            B.CreateVoidCall("rt_sv_fred_s", {
                Value::cpu_ptr(), Value::imm_i32(fop), Value::imm_i32(d), Value::imm_i32(n)
            });
            return true;
        }
        int op = (opc == 0b00011) ? 1 : (opc == 0b01010) ? 2 : (opc == 0b11010) ? 3 : (opc == 0b11011) ? 0 : -1;
        if (op < 0 || (op == 0 && U) || sz == 3 || (sz == 2 && !Q)) return false;
        B.CreateVoidCall("rt_sl_across", {
            Value::cpu_ptr(), Value::imm_i32(op), Value::imm_i32(U),
            Value::imm_i32(Q), Value::imm_i32(sz), Value::imm_i32(d), Value::imm_i32(n)
        });
        return true;
    }

    bool lift_scalar_pairwise(IRBuilder& B, uint32_t w) {
        unsigned U = bits(w, 29, 1), sz = bits(w, 22, 2), opc = bits(w, 12, 5), n = bits(w, 5, 5), d = bits(w, 0, 5);
        if (!U) {
            if (opc == 0b11011 && sz == 3) {
                Value vn = read_vq(B, TypeKind::Vec2xI64, 1, n);
                Value e0 = B.CreateExtractElement(TypeKind::Vec2xI64, TypeKind::Int64, vn, 0, "p0");
                Value e1 = B.CreateExtractElement(TypeKind::Vec2xI64, TypeKind::Int64, vn, 1, "p1");
                Value sum = B.binop("add", TypeKind::Int64, e0, e1);
                B.write_arch_reg(64 + static_cast<int>(d), sum);
                return true;
            }
            return false;
        }
        int fop = -1;
        unsigned hi = sz >> 1;
        if (opc == 0b01100 && !hi) fop = FOP_MAXNM;
        else if (opc == 0b01100 && hi) fop = FOP_MINNM;
        else if (opc == 0b01101 && !hi) fop = FOP_ADD;
        else if (opc == 0b01111 && !hi) fop = FOP_MAX;
        else if (opc == 0b01111 && hi) fop = FOP_MIN;
        if (fop < 0) return false;
        B.CreateVoidCall("rt_scalar_fpair", {
            Value::cpu_ptr(), Value::imm_i32(sz & 1), Value::imm_i32(fop),
            Value::imm_i32(d), Value::imm_i32(n)
        });
        return true;
    }

    bool lift_modimm(IRBuilder& B, uint32_t w) {
        unsigned Q = bits(w, 30, 1), op = bits(w, 29, 1), cmode = bits(w, 12, 4), o2 = bits(w, 11, 1), d = bits(w, 0, 5);
        uint32_t imm8 = (bits(w, 16, 3) << 5) | bits(w, 5, 5);
        if (o2 || (cmode == 15 && op && !Q)) return false;
        uint64_t imm = simd_expand_imm(op, cmode, imm8);
        if ((cmode < 8 && (cmode & 1)) || (cmode >= 8 && cmode < 12 && (cmode & 1))) { // ORR / BIC
            if (op) imm = ~imm;
            Value vd = read_vq(B, TypeKind::Vec2xI64, 1, d);
            Value iv = B.bitcast(Value::imm_i128(imm, imm), TypeKind::Vec2xI64);
            write_vq(B, Q, d, B.binop(op ? "and" : "or", TypeKind::Vec2xI64, vd, iv));
            return true;
        }
        if (op && cmode != 14 && cmode != 15) imm = ~imm;
        B.write_arch_reg(64 + static_cast<int>(d), Value::imm_i128(imm, Q ? imm : 0));
        return true;
    }

    bool lift_shift_imm(IRBuilder& B, uint32_t w, bool scalar) {
        unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), immh = bits(w, 19, 4), immb = bits(w, 16, 3),
                 opc = bits(w, 11, 5), n = bits(w, 5, 5), d = bits(w, 0, 5);
        if (!immh) return false;
        unsigned sz = 31u - __builtin_clz(immh);
        unsigned es = 8u << sz;
        unsigned ih = (immh << 3) | immb;
        unsigned rsh = 2 * es - ih, lsh = ih - es;
        if (scalar) Q = 1;
        unsigned ne = scalar ? 1 : ((8u << Q) >> sz);
        bool narrow = (opc >= 0b10000 && opc <= 0b10011);

        if (opc == 0b11100 || opc == 0b11111) { // Fixed-point conversions
            if (sz < 2 || (sz == 3 && !Q)) return false;
            int to_int = (opc == 0b11111) ? 1 : 0;
            B.CreateVoidCall("rt_vec_fcvt_int", {
                Value::cpu_ptr(), Value::imm_i32(sz - 2), Value::imm_i32(to_int),
                Value::imm_i32(Q), Value::imm_i32(scalar ? 1 : 0), Value::imm_i32(3),
                Value::imm_i32(rsh), Value::imm_i32(U), Value::imm_i32(d), Value::imm_i32(n)
            });
            return true;
        }
        if (narrow) {
            if (sz == 3) return false;
            int nlanes = scalar ? 1 : (8 >> sz);
            int op = -1;
            if (opc == 0b10000 && !U) { if (scalar) return false; op = LOP_SHRN; }
            else if (opc == 0b10001 && !U) { if (scalar) return false; op = LOP_RSHRN; }
            else if (opc == 0b10000 && U) op = LOP_QSHRUN;
            else if (opc == 0b10001 && U) op = LOP_QRSHRUN;
            else if (opc == 0b10010 && !U) op = LOP_QSHRN;
            else if (opc == 0b10010 && U) op = LOP_QSHRN | LOP_U;
            else if (opc == 0b10011 && !U) op = LOP_QRSHRN;
            else if (opc == 0b10011 && U) op = LOP_QRSHRN | LOP_U;
            int part = scalar ? 0 : static_cast<int>(Q);
            emit_sl_gen(B, op, sz, nlanes, d, sz, part * nlanes, part, n, sz + 1, 0, -1, 0, 0, -1, rsh);
            return true;
        }
        if (opc == 0b10100) { // SSHLL / USHLL
            if (scalar || sz == 3) return false;
            int nlanes = 8 >> sz;
            emit_sl_gen(B, LOP_SHLLI | (U ? LOP_U : 0), sz, nlanes, d, sz + 1, 0, 0, n, sz, static_cast<int>(Q) * nlanes, -1, 0, 0, -1, lsh);
            return true;
        }
        if (sz == 3 && !Q) return false;
        if (scalar && sz != 3 && opc != 0b01100 && opc != 0b01110) return false;

        if (opc == 0 || opc == 2) { // SSHR/USHR, SSRA/USRA
            int op = (opc == 0) ? LOP_SHRI : LOP_SRAI;
            emit_same_lop(B, op | (U ? LOP_U : 0), sz, ne, d, n, -1, rsh);
            return true;
        }
        if (opc == 4 || opc == 6) {
            emit_same_lop(B, (opc == 4 ? LOP_RSHRI : LOP_RSRAI) | (U ? LOP_U : 0), sz, ne, d, n, -1, rsh);
            return true;
        }
        if (opc == 8) {
            if (!U) return false;
            emit_same_lop(B, LOP_SRI, sz, ne, d, n, -1, rsh);
            return true;
        }
        if (opc == 10) {
            emit_same_lop(B, U ? LOP_SLI : LOP_SHLI, sz, ne, d, n, -1, lsh);
            return true;
        }
        if (opc == 12) {
            if (!U) return false;
            emit_same_lop(B, LOP_QSHLUI, sz, ne, d, n, -1, lsh);
            return true;
        }
        if (opc == 14) {
            emit_same_lop(B, LOP_QSHLI | (U ? LOP_U : 0), sz, ne, d, n, -1, lsh);
            return true;
        }
        return false;
    }

    bool lift_indexed(IRBuilder& B, uint32_t w, bool scalar) {
        unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), sz = bits(w, 22, 2), L = bits(w, 21, 1),
                 M = bits(w, 20, 1), rm = bits(w, 16, 4), opc = bits(w, 12, 4), H = bits(w, 11, 1),
                 n = bits(w, 5, 5), d = bits(w, 0, 5);
        bool fp = (opc == 1 || opc == 5 || opc == 9) && !(U && opc != 9);
        if (fp) {
            if (sz < 2) return false;
            unsigned fsz = sz & 1;
            if (fsz && L) return false;
            unsigned idx = fsz ? H : ((H << 1) | L);
            unsigned m = (M << 4) | rm;
            if (!scalar && fsz && !Q) return false;
            B.CreateVoidCall("rt_indexed_fp", {
                Value::cpu_ptr(), Value::imm_i32(fsz), Value::imm_i32(Q),
                Value::imm_i32(scalar ? 1 : 0), Value::imm_i32(opc), Value::imm_i32(U),
                Value::imm_i32(d), Value::imm_i32(n), Value::imm_i32(m), Value::imm_i32(idx)
            });
            return true;
        }
        unsigned idx = 0, m = 0;
        if (sz == 1) { idx = (H << 2) | (L << 1) | M; m = rm; }
        else if (sz == 2) { idx = (H << 1) | L; m = (M << 4) | rm; }
        else return false;

        if (scalar) {
            if (U) return false;
            int op = (opc == 3) ? LOP_SQDMLAL : (opc == 7) ? LOP_SQDMLSL : (opc == 11) ? LOP_SQDMULL
                   : (opc == 12) ? LOP_SQDMULH : (opc == 13) ? LOP_SQRDMULH : -1;
            if (op < 0) return false;
            int szd = (op == LOP_SQDMULH || op == LOP_SQRDMULH) ? static_cast<int>(sz) : static_cast<int>(sz + 1);
            emit_sl_gen(B, op, sz, 1, d, szd, 0, 0, n, sz, 0, m, sz, 0, idx);
            return true;
        }
        unsigned ne = (8u << Q) >> sz;
        TypeKind vt = vec_int_type(sz);
        if (!U && opc == 8) {
            Value vm = read_vq(B, vt, 1, m);
            Value el = emit_shufflevector(B, vt, vm, vm, std::vector<int>(16u >> sz, static_cast<int>(idx)));
            write_vq(B, Q, d, B.binop("mul", vt, read_vq(B, vt, 1, n), el));
            return true;
        }
        if (U && (opc == 0 || opc == 4)) {
            Value vm = read_vq(B, vt, 1, m);
            Value el = emit_shufflevector(B, vt, vm, vm, std::vector<int>(16u >> sz, static_cast<int>(idx)));
            Value prod = B.binop("mul", vt, read_vq(B, vt, 1, n), el);
            write_vq(B, Q, d, B.binop(opc == 0 ? "add" : "sub", vt, read_vq(B, vt, 1, d), prod));
            return true;
        }
        if (!U && (opc == 12 || opc == 13)) {
            emit_same_lop(B, opc == 12 ? LOP_SQDMULH : LOP_SQRDMULH, sz, ne, d, n, m, 0, idx);
            return true;
        }
        int lng = -1;
        switch (opc) {
            case 2:  lng = LOP_MLAL | (U ? LOP_U : 0); break;
            case 6:  lng = LOP_MLSL | (U ? LOP_U : 0); break;
            case 10: lng = LOP_MULL | (U ? LOP_U : 0); break;
            case 3:  if (U) return false; lng = LOP_SQDMLAL; break;
            case 7:  if (U) return false; lng = LOP_SQDMLSL; break;
            case 11: if (U) return false; lng = LOP_SQDMULL; break;
            default: return false;
        }
        int nlanes = 8 >> sz;
        emit_sl_gen(B, lng, sz, nlanes, d, sz + 1, 0, 0, n, sz, static_cast<int>(Q) * nlanes, m, sz, 0, idx);
        return true;
    }

    // -------------------------------------------------------------------------
    // Scalar Floating-Point
    // -------------------------------------------------------------------------
    bool lift_fp_dp3(IRBuilder& B, uint32_t w) {
        unsigned t = bits(w, 22, 2);
        if (t > 1) return false;
        unsigned o1 = bits(w, 21, 1), o0 = bits(w, 15, 1);
        unsigned d = bits(w, 0, 5), n = bits(w, 5, 5), m = bits(w, 16, 5), a = bits(w, 10, 5);
        unsigned na = o1, nn = o1 ^ o0;
        std::string fn_name = t ? "rt_fp_fma_d" : "rt_fp_fma_s";
        B.CreateVoidCall(fn_name, {
            Value::cpu_ptr(), Value::imm_i32(na), Value::imm_i32(nn),
            Value::imm_i32(d), Value::imm_i32(n), Value::imm_i32(m), Value::imm_i32(a)
        });
        return true;
    }

    bool lift_fp_dp2(IRBuilder& B, uint32_t w) {
        unsigned t = bits(w, 22, 2);
        if (t > 1) return false;
        unsigned opc = bits(w, 12, 4), d = bits(w, 0, 5), n = bits(w, 5, 5), m = bits(w, 16, 5);
        if (opc > 8) return false;
        static const int fops[9] = {FOP_MUL, FOP_DIV, FOP_ADD, FOP_SUB, FOP_MAX, FOP_MIN, FOP_MAXNM, FOP_MINNM, FOP_NMUL};
        std::string fn_name = t ? "rt_fp_op_d" : "rt_fp_op_s";
        B.CreateVoidCall(fn_name, {
            Value::cpu_ptr(), Value::imm_i32(fops[opc]),
            Value::imm_i32(d), Value::imm_i32(n), Value::imm_i32(m)
        });
        return true;
    }

    bool lift_fp_dp1(IRBuilder& B, uint32_t w) {
        unsigned t = bits(w, 22, 2), opc = bits(w, 15, 6), d = bits(w, 0, 5), n = bits(w, 5, 5);
        if (t == 2) return false;
        if (t == 3 && opc != 4 && opc != 5) return false;
        if (t <= 1) {
            if (opc == 0) { // FMOV
                Value vn = B.read_arch_reg(64 + static_cast<int>(n), TypeKind::Int128);
                Value lo = B.cast("trunc", vn, t ? TypeKind::Int64 : TypeKind::Int32);
                B.write_arch_reg(64 + static_cast<int>(d), lo);
                return true;
            }
            if (opc == 1 || opc == 2) { // FABS / FNEG
                TypeKind it = t ? TypeKind::Int64 : TypeKind::Int32;
                uint64_t sign_m = t ? 0x8000000000000000ULL : 0x80000000ULL;
                Value vn = B.cast("trunc", B.read_arch_reg(64 + static_cast<int>(n), TypeKind::Int128), it);
                Value res = (opc == 1) ? B.binop("and", it, vn, Value::imm(it, ~sign_m & ones(t ? 64 : 32)))
                                       : B.binop("xor", it, vn, Value::imm(it, sign_m));
                B.write_arch_reg(64 + static_cast<int>(d), res);
                return true;
            }
            if (opc != 3 && opc != 4 && opc != 5 && opc != 7 &&
                opc != 8 && opc != 9 && opc != 10 && opc != 11 && opc != 12 && opc != 14 && opc != 15) {
                return false;
            }
            if ((opc == 4 && t == 0) || (opc == 5 && t == 1)) return false;
        }
        B.CreateVoidCall("rt_fp_dp1", {
            Value::cpu_ptr(), Value::imm_i32(t), Value::imm_i32(opc),
            Value::imm_i32(d), Value::imm_i32(n)
        });
        return true;
    }

    bool lift_fp_cmp(IRBuilder& B, uint32_t w) {
        unsigned t = bits(w, 22, 2);
        if (t > 1 || bits(w, 14, 2) || bits(w, 0, 3)) return false;
        unsigned n = bits(w, 5, 5), m = bits(w, 16, 5), zero = bits(w, 3, 1);
        std::string fn_name = t ? "rt_fp_cmp_d" : "rt_fp_cmp_s";
        Value r = B.CreateCall(TypeKind::Int64, fn_name, {
            Value::cpu_ptr(), Value::imm_i32(n), Value::imm_i32(m), Value::imm_i32(zero)
        }, "fcmp");
        B.write_arch_reg(32, r, OpTag::FlagWrite);
        return true;
    }

    bool lift_fp_ccmp(IRBuilder& B, uint32_t w) {
        unsigned t = bits(w, 22, 2);
        if (t > 1) return false;
        unsigned n = bits(w, 5, 5), m = bits(w, 16, 5), cnd = bits(w, 12, 4), nzcv = bits(w, 0, 4);
        Value c_i1 = emit_cond_i1(B, cnd);
        std::string fn_name = t ? "rt_fp_cmp_d" : "rt_fp_cmp_s";
        Value r = B.CreateCall(TypeKind::Int64, fn_name, {
            Value::cpu_ptr(), Value::imm_i32(n), Value::imm_i32(m), Value::imm_i32(0)
        }, "fccmp");
        Value res = B.select(TypeKind::Int64, c_i1, r,
                             Value::imm_u64(static_cast<uint64_t>(nzcv) << 28));
        B.write_arch_reg(32, res, OpTag::FlagWrite);
        return true;
    }

    bool lift_fp_csel(IRBuilder& B, uint32_t w) {
        unsigned t = bits(w, 22, 2);
        if (t > 1) return false;
        unsigned d = bits(w, 0, 5), n = bits(w, 5, 5), m = bits(w, 16, 5), cnd = bits(w, 12, 4);
        TypeKind it = t ? TypeKind::Int64 : TypeKind::Int32;
        Value vn = B.cast("trunc", B.read_arch_reg(64 + static_cast<int>(n), TypeKind::Int128), it);
        Value vm = B.cast("trunc", B.read_arch_reg(64 + static_cast<int>(m), TypeKind::Int128), it);
        Value c_i1 = emit_cond_i1(B, cnd);
        B.write_arch_reg(64 + static_cast<int>(d), B.select(it, c_i1, vn, vm));
        return true;
    }

    bool lift_fp_imm(IRBuilder& B, uint32_t w) {
        unsigned t = bits(w, 22, 2);
        if (t > 1 || bits(w, 5, 5)) return false;
        uint32_t imm8 = bits(w, 13, 8), d = bits(w, 0, 5);
        uint64_t val = vfp_expand_imm(imm8, 32u << t);
        B.write_arch_reg(64 + static_cast<int>(d), Value::imm_i128(val, 0));
        return true;
    }

    bool lift_fp_fixed(IRBuilder& B, uint32_t w) {
        unsigned sf = bits(w, 31, 1), t = bits(w, 22, 2), rmode = bits(w, 19, 2), opc = bits(w, 16, 3),
                 scale = bits(w, 10, 6), n = bits(w, 5, 5), d = bits(w, 0, 5);
        if (t > 1 || (!sf && scale < 32)) return false;
        unsigned fb = 64 - scale, ib = sf ? 64 : 32;
        if (rmode == 0 && (opc == 2 || opc == 3)) {
            Value src = sf ? get_gpr(B, 1, n) : B.cast("zext", get_gpr(B, 0, n), TypeKind::Int64);
            B.CreateVoidCall("rt_fp_fromint", {
                Value::cpu_ptr(), Value::imm_i32(t), Value::imm_i32(d),
                src, Value::imm_i32(ib), Value::imm_i32(opc & 1), Value::imm_i32(fb)
            });
            return true;
        }
        if (rmode == 3 && (opc == 0 || opc == 1)) {
            Value rv = B.CreateCall(TypeKind::Int64, "rt_fp_toint", {
                Value::cpu_ptr(), Value::imm_i32(t), Value::imm_i32(n),
                Value::imm_i32(3), Value::imm_i32(fb), Value::imm_i32(ib), Value::imm_i32(opc)
            }, "fptoi");
            set_gpr(B, sf, d, sf ? rv : B.cast("trunc", rv, TypeKind::Int32));
            return true;
        }
        return false;
    }

    bool lift_fp_int(IRBuilder& B, uint32_t w) {
        unsigned sf = bits(w, 31, 1), t = bits(w, 22, 2), rmode = bits(w, 19, 2), opc = bits(w, 16, 3),
                 n = bits(w, 5, 5), d = bits(w, 0, 5);
        if (opc == 6 || opc == 7) { // FMOV (general)
            if (t == 2 && sf && rmode == 1) {
                if (opc == 6) {
                    Value vn = read_vq(B, TypeKind::Vec2xI64, 1, n);
                    Value hi = B.CreateExtractElement(TypeKind::Vec2xI64, TypeKind::Int64, vn, 1, "vhi");
                    set_gpr(B, 1, d, hi);
                } else {
                    Value vd = read_vq(B, TypeKind::Vec2xI64, 1, d);
                    Value xn = get_gpr(B, 1, n);
                    Value ins = B.CreateInsertElement(TypeKind::Vec2xI64, vd, xn, 1, "vhiw");
                    write_vq(B, 1, d, ins);
                }
                return true;
            }
            if (rmode || t != sf || t > 1) return false;
            if (opc == 6) {
                TypeKind it = sf ? TypeKind::Int64 : TypeKind::Int32;
                Value vn = B.cast("trunc", B.read_arch_reg(64 + static_cast<int>(n), TypeKind::Int128), it);
                set_gpr(B, sf, d, vn);
            } else {
                Value g = get_gpr(B, sf, n);
                B.write_arch_reg(64 + static_cast<int>(d), g);
            }
            return true;
        }
        if (t > 1) return false;
        unsigned ib = sf ? 64 : 32;
        if (opc == 2 || opc == 3) {
            if (rmode) return false;
            Value src = sf ? get_gpr(B, 1, n) : B.cast("zext", get_gpr(B, 0, n), TypeKind::Int64);
            B.CreateVoidCall("rt_fp_fromint", {
                Value::cpu_ptr(), Value::imm_i32(t), Value::imm_i32(d),
                src, Value::imm_i32(ib), Value::imm_i32(opc & 1), Value::imm_i32(0)
            });
            return true;
        }
        int rm = 0;
        if (opc == 4 || opc == 5) {
            if (rmode) return false;
            rm = 4;
        } else {
            rm = static_cast<int>(rmode);
        }
        Value rv = B.CreateCall(TypeKind::Int64, "rt_fp_toint", {
            Value::cpu_ptr(), Value::imm_i32(t), Value::imm_i32(n),
            Value::imm_i32(rm), Value::imm_i32(0), Value::imm_i32(ib), Value::imm_i32(opc & 1)
        }, "fptoi");
        set_gpr(B, sf, d, sf ? rv : B.cast("trunc", rv, TypeKind::Int32));
        return true;
    }
};

} // namespace ultrs::lifter
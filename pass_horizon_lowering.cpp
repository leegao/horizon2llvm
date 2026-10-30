// UltraTears LLVM Pass Implementation: Pass 9 - HorizonLoweringPass (tools/lifter/pass_horizon_lowering.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class HorizonLoweringPassImpl {
public:
    static const char* static_name() { return "horizon-lowering-pass"; }
    const char* pass_name() const { return "horizon-lowering-pass"; }

    bool run(Module& M) {
        bool changed = false;
        for (auto& fn : M.functions) {
            for (auto& bb : fn->blocks) {
                for (size_t i = 0; i < bb->insts.size(); ++i) {
                    Instruction& inst = bb->insts[i];
                    if (inst.dead) continue;
                    switch (inst.tag) {
                        case OpTag::HorizonSvc: {
                            inst.text = "call void @rt_svc(ptr %c, i32 " + std::to_string(inst.sys_key) +
                                        ", i64 " + std::to_string(inst.guest_pc) + ")";
                            inst.tag = OpTag::IndirectCall;
                            changed = true;
                            break;
                        }
                        case OpTag::HorizonBrk: {
                            inst.text = "call void @rt_brk(ptr %c, i32 " + std::to_string(inst.sys_key) +
                                        ", i64 " + std::to_string(inst.guest_pc) + ")\n  unreachable";
                            changed = true;
                            break;
                        }
                        case OpTag::HorizonBarrier: {
                            if (inst.sys_key == 2) { // CLREX
                                std::string p = fn->next_reg("clrex");
                                inst.text = p + " = getelementptr inbounds i8, ptr %c, i64 296\n  "
                                            "store i64 -1, ptr " + p + ", align 8, !alias.scope !3, !noalias !4";
                            } else {
                                inst.text = "fence seq_cst";
                            }
                            inst.tag = OpTag::RawIR;
                            changed = true;
                            break;
                        }
                        case OpTag::HorizonDcZva: {
                            std::string aligned = fn->next_reg("zva_a");
                            std::string ptr_r   = fn->next_reg("zva_p");
                            inst.text = aligned + " = and i64 " + inst.op0.repr + ", -64\n  " +
                                        ptr_r + " = inttoptr i64 " + aligned + " to ptr\n  " +
                                        "call void @llvm.memset.p0.i64(ptr align 64 " + ptr_r + ", i8 0, i64 64, i1 false)";
                            inst.tag = OpTag::RawIR;
                            changed = true;
                            break;
                        }
                        case OpTag::HorizonSysRegRead: {
                            lower_mrs(*fn, *bb, i);
                            changed = true;
                            break;
                        }
                        case OpTag::HorizonSysRegWrite: {
                            lower_msr(*fn, *bb, i);
                            changed = true;
                            break;
                        }
                        case OpTag::GuestMemLoad:
                        case OpTag::GuestMemStore: {
                            if (inst.text.find("!noalias") == std::string::npos) {
                                inst.text += ", !alias.scope !4, !noalias !3";
                                changed = true;
                            }
                            break;
                        }
                        default:
                            break;
                    }
                }
            }
        }
        return changed;
    }

private:
    static constexpr uint32_t pack_sys(unsigned op0, unsigned op1, unsigned crn, unsigned crm, unsigned op2) {
        return (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2;
    }

    static void lower_mrs(Function& fn, BasicBlock& bb, size_t idx) {
        Instruction& inst = bb.insts[idx];
        int rt = inst.arch_reg;
        if (rt == 31) {
            inst.dead = true;
            return;
        }
        uint32_t k = inst.sys_key;
        std::string val_repr;
        Instruction pre_inst;
        bool has_pre = false;

        if (k == pack_sys(3, 3, 13, 0, 3)) { // TPIDRRO_EL0 -> c->tpidrro (offset 288)
            std::string p = fn.next_reg("tpro_p");
            std::string v = fn.next_reg("tpro");
            pre_inst.tag = OpTag::RawIR;
            pre_inst.dst_reg = v;
            pre_inst.dst_type = TypeKind::Int64;
            pre_inst.sys_key = k;
            pre_inst.text = p + " = getelementptr inbounds i8, ptr %c, i64 288\n  " +
                            v + " = load i64, ptr " + p + ", align 8, !alias.scope !3, !noalias !4";
            has_pre = true;
            val_repr = v;
        } else if (k == pack_sys(3, 3, 13, 0, 2)) { // TPIDR_EL0 -> c->tpidr (offset 280)
            std::string p = fn.next_reg("tp_p");
            std::string v = fn.next_reg("tp");
            pre_inst.tag = OpTag::RawIR;
            pre_inst.dst_reg = v;
            pre_inst.dst_type = TypeKind::Int64;
            pre_inst.text = p + " = getelementptr inbounds i8, ptr %c, i64 280\n  " +
                            v + " = load i64, ptr " + p + ", align 8, !alias.scope !3, !noalias !4";
            has_pre = true;
            val_repr = v;
        } else if (k == pack_sys(3, 3, 14, 0, 0)) { // CNTFRQ_EL0
            val_repr = "19200000";
        } else if (k == pack_sys(3, 3, 14, 0, 1) || k == pack_sys(3, 3, 14, 0, 2)) { // CNTPCT_EL0 / CNTVCT_EL0
            std::string v = fn.next_reg("ticks");
            pre_inst.tag = OpTag::RawIR;
            pre_inst.text = v + " = call i64 @rt_ticks()";
            has_pre = true;
            val_repr = v;
        } else if (k == pack_sys(3, 3, 4, 2, 0)) { // NZCV -> arch_reg 32
            std::string v = fn.next_reg("nzcv_v");
            std::string p = v + ".p";
            pre_inst.tag = OpTag::CpuRegRead;
            pre_inst.arch_reg = 32;
            pre_inst.dst_reg = v;
            pre_inst.dst_type = TypeKind::Int64;
            pre_inst.text = p + " = getelementptr inbounds i8, ptr %c, i64 272\n  " +
                            v + " = load i64, ptr " + p + ", align 8, !alias.scope !3, !noalias !4";
            has_pre = true;
            val_repr = v;
        } else if (k == pack_sys(3, 3, 4, 4, 0)) { // FPCR -> c->fpcr (offset 320)
            std::string p = fn.next_reg("fpcr_p");
            std::string v = fn.next_reg("fpcr_v");
            pre_inst.tag = OpTag::RawIR;
            pre_inst.text = p + " = getelementptr inbounds i8, ptr %c, i64 320\n  " +
                            v + " = load i64, ptr " + p + ", align 8, !alias.scope !3, !noalias !4";
            has_pre = true;
            val_repr = v;
        } else if (k == pack_sys(3, 3, 4, 4, 1)) { // FPSR -> c->fpsr (offset 328)
            std::string p = fn.next_reg("fpsr_p");
            std::string v = fn.next_reg("fpsr_v");
            pre_inst.tag = OpTag::RawIR;
            pre_inst.text = p + " = getelementptr inbounds i8, ptr %c, i64 328\n  " +
                            v + " = load i64, ptr " + p + ", align 8, !alias.scope !3, !noalias !4";
            has_pre = true;
            val_repr = v;
        } else if (k == pack_sys(3, 3, 0, 0, 7)) { // DCZID_EL0
            val_repr = "4";
        } else if (k == pack_sys(3, 3, 0, 0, 1)) { // CTR_EL0
            val_repr = "2219098116"; // 0x8444c004
        } else {
            inst.tag = OpTag::HorizonUdf;
            inst.text = "call void @rt_udf(ptr %c, i64 " + std::to_string(inst.guest_pc) + ")\n  unreachable";
            return;
        }

        std::string wp = fn.next_reg("mrsw");
        inst.tag = OpTag::CpuRegWrite;
        inst.arch_reg = rt;
        inst.op0 = Value::reg(TypeKind::Int64, val_repr);
        inst.text = wp + " = getelementptr inbounds i8, ptr %c, i64 " + std::to_string(rt * 8) + "\n  " +
                    "store i64 " + val_repr + ", ptr " + wp + ", align 8, !alias.scope !3, !noalias !4";
        if (has_pre) {
            bb.insts.insert(bb.insts.begin() + static_cast<ptrdiff_t>(idx), std::move(pre_inst));
        }
    }

    static void lower_msr(Function& fn, BasicBlock& bb, size_t idx) {
        Instruction& inst = bb.insts[idx];
        uint32_t k = inst.sys_key;
        std::string v = inst.op0.repr;
        if (k == pack_sys(3, 3, 13, 0, 2)) { // TPIDR_EL0
            std::string p = fn.next_reg("msrp");
            inst.text = p + " = getelementptr inbounds i8, ptr %c, i64 280\n  " +
                        "store i64 " + v + ", ptr " + p + ", align 8, !alias.scope !3, !noalias !4";
            inst.tag = OpTag::RawIR;
        } else if (k == pack_sys(3, 3, 4, 2, 0)) { // NZCV -> arch_reg 32
            std::string m = fn.next_reg("nzcv_m");
            std::string p = fn.next_reg("nzcv_p");
            std::string pre = m + " = and i64 " + v + ", 4026531840";

            inst.tag = OpTag::FlagWrite;
            inst.arch_reg = 32;
            inst.op0 = Value::reg(TypeKind::Int64, m);
            inst.prefix_ir = pre;
            inst.text = pre + "\n  " + p + " = getelementptr inbounds i8, ptr %c, i64 272\n  " +
                        "store i64 " + m + ", ptr " + p + ", align 8, !alias.scope !3, !noalias !4";
        } else if (k == pack_sys(3, 3, 4, 4, 0)) { // FPCR -> rt_fpcr_write_ext (updates host MXCSR FTZ/DAZ)
            inst.text = "call void @rt_fpcr_write_ext(ptr %c, i64 " + v + ")";
            inst.tag = OpTag::RawIR;
        } else if (k == pack_sys(3, 3, 4, 4, 1)) { // FPSR
            std::string m = fn.next_reg("fpsr_m");
            std::string p = fn.next_reg("fpsr_p");
            inst.text = m + " = and i64 " + v + ", 4160749727\n  " +
                        p + " = getelementptr inbounds i8, ptr %c, i64 328\n  " +
                        "store i64 " + m + ", ptr " + p + ", align 8, !alias.scope !3, !noalias !4";
            inst.tag = OpTag::RawIR;
        } else {
            inst.tag = OpTag::HorizonUdf;
            inst.text = "call void @rt_udf(ptr %c, i64 " + std::to_string(inst.guest_pc) + ")\n  unreachable";
        }
    }
};


} // namespace

bool HorizonLoweringPass::run(Module& M) {
    HorizonLoweringPassImpl impl;
    return impl.run(M);
}

} // namespace ultrs::passes

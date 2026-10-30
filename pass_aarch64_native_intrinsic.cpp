// UltraTears LLVM Pass Implementation: Pass 6 - AArch64NativeIntrinsicPass (tools/lifter/pass_aarch64_native_intrinsic.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class AArch64NativeIntrinsicPassImpl {
public:
    explicit AArch64NativeIntrinsicPassImpl(bool force_enable = false) : force_enable_(force_enable) {}

    static const char* static_name() { return "aarch64-native-intrinsic-pass"; }
    const char* pass_name() const { return "aarch64-native-intrinsic-pass"; }

    bool run(Module& M) {
        if (!force_enable_ && !M.is_aarch64_target()) return false;
        bool changed = false;

        for (auto& fn : M.functions) {
            if (!fn->hle_symbol.empty()) continue;
            for (auto& bb : fn->blocks) {
                std::vector<Instruction> out;
                out.reserve(bb->insts.size() * 2);

                for (auto& inst : bb->insts) {
                    if (inst.dead || inst.tag != OpTag::RawIR) {
                        out.push_back(std::move(inst));
                        continue;
                    }

                    // 1. @rt_crc32: %crc = call i32 @rt_crc32(i32 %acc, i64 %val, i32 <bytes>, i32 <C>)
                    if (inst.text.find("call i32 @rt_crc32(") != std::string::npos) {
                        char dst[64] = {}, acc[64] = {}, val[64] = {};
                        int bytes = 0, is_c = 0;
                        if (std::sscanf(inst.text.c_str(),
                                        "%63s = call i32 @rt_crc32(i32 %63[^,], i64 %63[^,], i32 %d, i32 %d)",
                                        dst, acc, val, &bytes, &is_c) == 5) {
                            const char* intr = (bytes == 1) ? (is_c ? "@llvm.aarch64.crc32cb" : "@llvm.aarch64.crc32b")
                                             : (bytes == 2) ? (is_c ? "@llvm.aarch64.crc32ch" : "@llvm.aarch64.crc32h")
                                             : (bytes == 4) ? (is_c ? "@llvm.aarch64.crc32cw" : "@llvm.aarch64.crc32w")
                                                            : (is_c ? "@llvm.aarch64.crc32cx" : "@llvm.aarch64.crc32x");
                            if (bytes == 8) {
                                inst.text = std::string("; [AArch64NativeIntrinsicPass] hardware CRC32X\n  ") +
                                            dst + " = call i32 " + intr + "(i32 " + acc + ", i64 " + val + ")";
                            } else {
                                std::string vt = fn->next_reg("crct");
                                inst.text = std::string("; [AArch64NativeIntrinsicPass] hardware CRC32\n  ") +
                                            vt + " = trunc i64 " + val + " to i32\n  " +
                                            dst + " = call i32 " + intr + "(i32 " + acc + ", i32 " + vt + ")";
                            }
                            out.push_back(std::move(inst));
                            M.uses_aarch64_intrinsics = true;
                            changed = true;
                            continue;
                        }
                    }

                    // 2. @rt_sv_aes(ptr %c, i32 <op>, i32 <d>, i32 <n>)
                    int a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0;
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_sv_aes(ptr %%c, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2) == 3 && a0 >= 0 && a0 <= 3) {
                        emit_aes_intrinsic(*fn, out, a0, a1, a2);
                        M.uses_aarch64_intrinsics = true;
                        changed = true;
                        continue;
                    }

                    // 3. @rt_sl_pmull64(ptr %c, i32 <Q>, i32 <d>, i32 <n>, i32 <m>)
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_sl_pmull64(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3) == 4) {
                        emit_pmull64_intrinsic(*fn, out, a0, a1, a2, a3);
                        M.uses_aarch64_intrinsics = true;
                        changed = true;
                        continue;
                    }

                    // 4. @rt_sv_tbl(ptr %c, i32 <Q>, i32 0, i32 <op>, i32 <d>, i32 <n>, i32 <m>)
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_sv_tbl(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3, &a4, &a5) == 6 && a1 == 0) {
                        emit_tbl1_intrinsic(*fn, out, a0, a2, a3, a4, a5);
                        M.uses_aarch64_intrinsics = true;
                        changed = true;
                        continue;
                    }

                    // 5. @rt_sl_across(ptr %c, i32 <op>, i32 <U>, i32 <Q>, i32 <sz>, i32 <d>, i32 <n>)
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_sl_across(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3, &a4, &a5) == 6 &&
                        a2 == 1 && (a0 == 0 || a0 == 2 || a0 == 3) && a3 <= 2) {
                        emit_across_intrinsic(*fn, out, a0, a1, a3, a4, a5);
                        M.uses_aarch64_intrinsics = true;
                        changed = true;
                        continue;
                    }

                    out.push_back(std::move(inst));
                }
                bb->insts = std::move(out);
            }
        }
        return changed;
    }

private:
    bool force_enable_;

    static Instruction make_vreg_read(Function& fn, int vreg, std::string& out_reg) {
        out_reg = fn.next_reg("avr");
        int off = 512 + vreg * 16;
        std::string p = out_reg + ".p";
        Instruction inst;
        inst.tag = OpTag::CpuRegRead;
        inst.arch_reg = 64 + vreg;
        inst.dst_reg = out_reg;
        inst.dst_type = TypeKind::Int128;
        inst.text = p + " = getelementptr inbounds i8, ptr %c, i64 " + std::to_string(off) + "\n  " +
                    out_reg + " = load i128, ptr " + p + ", align 16, !alias.scope !3, !noalias !4";
        return inst;
    }

    static Instruction make_vreg_write(Function& fn, int vreg, const std::string& val_i128) {
        int off = 512 + vreg * 16;
        std::string p = fn.next_reg("avw");
        Instruction inst;
        inst.tag = OpTag::CpuRegWrite;
        inst.arch_reg = 64 + vreg;
        inst.op0 = Value::reg(TypeKind::Int128, val_i128);
        inst.text = p + " = getelementptr inbounds i8, ptr %c, i64 " + std::to_string(off) + "\n  " +
                    "store i128 " + val_i128 + ", ptr " + p + ", align 16, !alias.scope !3, !noalias !4";
        return inst;
    }

    static void emit_aes_intrinsic(Function& fn, std::vector<Instruction>& out, int op, int d, int n) {
        std::string vn128, vd128;
        if (op <= 1) out.push_back(make_vreg_read(fn, d, vd128));
        out.push_back(make_vreg_read(fn, n, vn128));

        std::string vn = fn.next_reg("aesn"), vd = fn.next_reg("aesd");
        std::string vr = fn.next_reg("aesr"), r128 = fn.next_reg("aesq");
        std::ostringstream os;
        os << "; [AArch64NativeIntrinsicPass] hardware AES instruction\n  "
           << vn << " = bitcast i128 " << vn128 << " to <16 x i8>\n  ";
        if (op <= 1) {
            const char* intr = (op == 0) ? "@llvm.aarch64.crypto.aese" : "@llvm.aarch64.crypto.aesd";
            os << vd << " = bitcast i128 " << vd128 << " to <16 x i8>\n  "
               << vr << " = call <16 x i8> " << intr << "(<16 x i8> " << vd << ", <16 x i8> " << vn << ")\n  ";
        } else {
            const char* intr = (op == 2) ? "@llvm.aarch64.crypto.aesmc" : "@llvm.aarch64.crypto.aesimc";
            os << vr << " = call <16 x i8> " << intr << "(<16 x i8> " << vn << ")\n  ";
        }
        os << r128 << " = bitcast <16 x i8> " << vr << " to i128";
        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }

    static void emit_pmull64_intrinsic(Function& fn, std::vector<Instruction>& out, int Q, int d, int n, int m) {
        std::string vn128, vm128;
        out.push_back(make_vreg_read(fn, n, vn128));
        out.push_back(make_vreg_read(fn, m, vm128));
        std::string vn = fn.next_reg("pmn"), vm = fn.next_reg("pmm");
        std::string ln = fn.next_reg("pmln"), lm = fn.next_reg("pmlm");
        std::string vr = fn.next_reg("pmr"), r128 = fn.next_reg("pmq");
        std::ostringstream os;
        os << "; [AArch64NativeIntrinsicPass] hardware PMULL64\n  "
           << vn << " = bitcast i128 " << vn128 << " to <2 x i64>\n  "
           << vm << " = bitcast i128 " << vm128 << " to <2 x i64>\n  "
           << ln << " = extractelement <2 x i64> " << vn << ", i32 " << Q << "\n  "
           << lm << " = extractelement <2 x i64> " << vm << ", i32 " << Q << "\n  "
           << vr << " = call <16 x i8> @llvm.aarch64.neon.pmull64(i64 " << ln << ", i64 " << lm << ")\n  "
           << r128 << " = bitcast <16 x i8> " << vr << " to i128";
        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }

    static void emit_tbl1_intrinsic(Function& fn, std::vector<Instruction>& out, int Q, int op, int d, int n, int m) {
        std::string vn128, vm128, vd128;
        if (op == 1) out.push_back(make_vreg_read(fn, d, vd128));
        out.push_back(make_vreg_read(fn, n, vn128));
        out.push_back(make_vreg_read(fn, m, vm128));
        std::string vn = fn.next_reg("tbn"), vm = fn.next_reg("tbm"), vd = fn.next_reg("tbd");
        std::string vr = fn.next_reg("tbr"), r128 = fn.next_reg("tbq");
        std::ostringstream os;
        os << "; [AArch64NativeIntrinsicPass] hardware TBL1/TBX1\n  "
           << vn << " = bitcast i128 " << vn128 << " to <16 x i8>\n  "
           << vm << " = bitcast i128 " << vm128 << " to <16 x i8>\n  ";
        if (op == 0) {
            os << vr << " = call <16 x i8> @llvm.aarch64.neon.tbl1.v16i8(<16 x i8> " << vn << ", <16 x i8> " << vm << ")\n  ";
        } else {
            os << vd << " = bitcast i128 " << vd128 << " to <16 x i8>\n  "
               << vr << " = call <16 x i8> @llvm.aarch64.neon.tbx1.v16i8(<16 x i8> " << vd << ", <16 x i8> " << vn << ", <16 x i8> " << vm << ")\n  ";
        }
        if (Q) {
            os << r128 << " = bitcast <16 x i8> " << vr << " to i128";
        } else {
            std::string raw128 = fn.next_reg("tbrw"), lo64 = fn.next_reg("tblo");
            os << raw128 << " = bitcast <16 x i8> " << vr << " to i128\n  "
               << lo64 << " = trunc i128 " << raw128 << " to i64\n  "
               << r128 << " = zext i64 " << lo64 << " to i128";
        }
        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }

    static void emit_across_intrinsic(Function& fn, std::vector<Instruction>& out, int op, int U, int sz, int d, int n) {
        std::string vn128;
        out.push_back(make_vreg_read(fn, n, vn128));
        const char* vt = (sz == 0) ? "<16 x i8>" : (sz == 1) ? "<8 x i16>" : "<4 x i32>";
        const char* vsfx = (sz == 0) ? "v16i8" : (sz == 1) ? "v8i16" : "v4i32";
        const char* base = (op == 0) ? "saddv"
                         : (op == 2) ? (U ? "umaxv" : "smaxv")
                                     : (U ? "uminv" : "sminv");
        uint64_t mask = (sz == 0) ? 0xFFULL : (sz == 1) ? 0xFFFFULL : 0xFFFFFFFFULL;
        std::string vn = fn.next_reg("acn"), r32 = fn.next_reg("acr"), m32 = fn.next_reg("acm"), r128 = fn.next_reg("acq");
        std::ostringstream os;
        os << "; [AArch64NativeIntrinsicPass] hardware NEON across-vector reduction\n  "
           << vn << " = bitcast i128 " << vn128 << " to " << vt << "\n  "
           << r32 << " = call i32 @llvm.aarch64.neon." << base << ".i32." << vsfx << "(" << vt << " " << vn << ")\n  "
           << m32 << " = and i32 " << r32 << ", " << mask << "\n  "
           << r128 << " = zext i32 " << m32 << " to i128";
        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }
};


} // namespace

bool AArch64NativeIntrinsicPass::run(Module& M) {
    AArch64NativeIntrinsicPassImpl impl(force_enable_);
    return impl.run(M);
}

} // namespace ultrs::passes

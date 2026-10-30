// UltraTears LLVM Pass Implementation: Pass 4 - FPCRFastFPLoweringPass (tools/lifter/pass_fpcr_fast_fp_lowering.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class FPCRFastFPLoweringPassImpl {
public:
    explicit FPCRFastFPLoweringPassImpl(bool enable_fast_fp = true) : enable_fast_fp_(enable_fast_fp) {}

    static const char* static_name() { return "fpcr-fast-fp-lowering-pass"; }
    const char* pass_name() const { return "fpcr-fast-fp-lowering-pass"; }

    bool run(Module& M) {
        if (!enable_fast_fp_) return false;
        bool changed = false;
        constexpr uint32_t FPCR_KEY = (3u << 14) | (3u << 11) | (4u << 7) | (4u << 3) | 0u;

        for (auto& fn : M.functions) {
            if (!fn->hle_symbol.empty()) continue;
            bool writes_fpcr = false;
            for (const auto& bb : fn->blocks) {
                for (const auto& inst : bb->insts) {
                    if (!inst.dead && inst.tag == OpTag::HorizonSysRegWrite && inst.sys_key == FPCR_KEY) {
                        writes_fpcr = true;
                        break;
                    }
                }
            }
            fn->modifies_fpcr = writes_fpcr;
            if (writes_fpcr) continue;

            for (auto& bb : fn->blocks) {
                std::vector<Instruction> out;
                out.reserve(bb->insts.size() * 2);

                for (auto& inst : bb->insts) {
                    if (inst.dead || inst.tag != OpTag::RawIR) {
                        out.push_back(std::move(inst));
                        continue;
                    }
                    int a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_fp_op_s(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3) == 4 &&
                        is_inlineable_scalar_fop(a0)) {
                        emit_inline_scalar_fop(*fn, out, false, a0, a1, a2, a3);
                        changed = true;
                        continue;
                    }
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_fp_op_d(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3) == 4 &&
                        is_inlineable_scalar_fop(a0)) {
                        emit_inline_scalar_fop(*fn, out, true, a0, a1, a2, a3);
                        changed = true;
                        continue;
                    }
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_fp_fma_s(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3, &a4, &a5) == 6) {
                        emit_inline_scalar_fma(*fn, out, false, a0, a1, a2, a3, a4, a5);
                        changed = true;
                        continue;
                    }
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_fp_fma_d(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3, &a4, &a5) == 6) {
                        emit_inline_scalar_fma(*fn, out, true, a0, a1, a2, a3, a4, a5);
                        changed = true;
                        continue;
                    }
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_fp_dp1(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3) == 4 &&
                        (a0 == 0 || a0 == 1) && a1 == 3) { // FSQRT
                        emit_inline_scalar_sqrt(*fn, out, a0 == 1, a2, a3);
                        changed = true;
                        continue;
                    }
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_vec_fop(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7) == 8 &&
                        is_inlineable_vec_fop(a0, a1, a2)) {
                        emit_inline_vec_fop(*fn, out, a0, a1, a2, a3, a4, a5, a6, a7);
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
    bool enable_fast_fp_;

    static bool is_inlineable_scalar_fop(int fop) {
        return fop == FpOp::FOP_ADD || fop == FpOp::FOP_SUB || fop == FpOp::FOP_MUL ||
               fop == FpOp::FOP_NMUL || fop == FpOp::FOP_DIV ||
               fop == FpOp::FOP_MAXNM || fop == FpOp::FOP_MINNM;
    }

    static bool is_inlineable_vec_fop(int sz, int fop, int Q) {
        if (sz == 1 && Q == 0) return false; // 64-bit f64 scalar uses @rt_fp_op_d
        return fop == FpOp::FOP_ADD || fop == FpOp::FOP_SUB || fop == FpOp::FOP_MUL ||
               fop == FpOp::FOP_DIV || fop == FpOp::FOP_SQRT || fop == FpOp::FOP_MADD ||
               fop == FpOp::FOP_MAXNM || fop == FpOp::FOP_MINNM;
    }

    static Instruction make_vreg_read(Function& fn, int vreg, std::string& out_reg) {
        out_reg = fn.next_reg("vr");
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
        std::string p = fn.next_reg("vw");
        Instruction inst;
        inst.tag = OpTag::CpuRegWrite;
        inst.arch_reg = 64 + vreg;
        inst.op0 = Value::reg(TypeKind::Int128, val_i128);
        inst.text = p + " = getelementptr inbounds i8, ptr %c, i64 " + std::to_string(off) + "\n  " +
                    "store i128 " + val_i128 + ", ptr " + p + ", align 16, !alias.scope !3, !noalias !4";
        return inst;
    }

    static void emit_inline_scalar_fop(Function& fn, std::vector<Instruction>& out,
                                       bool is_double, int fop, int d, int n, int m) {
        std::string vn128, vm128;
        out.push_back(make_vreg_read(fn, n, vn128));
        out.push_back(make_vreg_read(fn, m, vm128));

        const char* it = is_double ? "i64" : "i32";
        const char* ft = is_double ? "double" : "float";
        std::string ni = fn.next_reg("fni"), mi = fn.next_reg("fmi");
        std::string nf = fn.next_reg("fnf"), mf = fn.next_reg("fmf");
        std::string rf = fn.next_reg("frf"), ri = fn.next_reg("fri"), r128 = fn.next_reg("frq");

        std::ostringstream os;
        os << "; [FPCRFastFPLoweringPass] inline scalar FP op\n  "
           << ni << " = trunc i128 " << vn128 << " to " << it << "\n  "
           << mi << " = trunc i128 " << vm128 << " to " << it << "\n  "
           << nf << " = bitcast " << it << " " << ni << " to " << ft << "\n  "
           << mf << " = bitcast " << it << " " << mi << " to " << ft << "\n  ";
        if (fop == FpOp::FOP_ADD) {
            os << rf << " = fadd " << ft << " " << nf << ", " << mf;
        } else if (fop == FpOp::FOP_SUB) {
            os << rf << " = fsub " << ft << " " << nf << ", " << mf;
        } else if (fop == FpOp::FOP_MUL) {
            os << rf << " = fmul " << ft << " " << nf << ", " << mf;
        } else if (fop == FpOp::FOP_NMUL) {
            std::string prod = fn.next_reg("fmp");
            os << prod << " = fmul " << ft << " " << nf << ", " << mf << "\n  "
               << rf << " = fneg " << ft << " " << prod;
        } else if (fop == FpOp::FOP_DIV) {
            os << rf << " = fdiv " << ft << " " << nf << ", " << mf;
        } else if (fop == FpOp::FOP_MAXNM) {
            os << rf << " = call " << ft << " @llvm.maxnum." << (is_double ? "f64" : "f32")
               << "(" << ft << " " << nf << ", " << ft << " " << mf << ")";
        } else {
            os << rf << " = call " << ft << " @llvm.minnum." << (is_double ? "f64" : "f32")
               << "(" << ft << " " << nf << ", " << ft << " " << mf << ")";
        }
        os << "\n  " << ri << " = bitcast " << ft << " " << rf << " to " << it
           << "\n  " << r128 << " = zext " << it << " " << ri << " to i128";

        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }

    static void emit_inline_scalar_fma(Function& fn, std::vector<Instruction>& out,
                                       bool is_double, int na, int nn, int d, int n, int m, int a) {
        std::string vn128, vm128, va128;
        out.push_back(make_vreg_read(fn, n, vn128));
        out.push_back(make_vreg_read(fn, m, vm128));
        out.push_back(make_vreg_read(fn, a, va128));

        const char* it = is_double ? "i64" : "i32";
        const char* ft = is_double ? "double" : "float";
        std::string ni = fn.next_reg("fni"), mi = fn.next_reg("fmi"), ai = fn.next_reg("fai");
        std::string nf = fn.next_reg("fnf"), mf = fn.next_reg("fmf"), af = fn.next_reg("faf");
        std::string rf = fn.next_reg("frf"), ri = fn.next_reg("fri"), r128 = fn.next_reg("frq");

        std::ostringstream os;
        os << "; [FPCRFastFPLoweringPass] inline scalar FMA\n  "
           << ni << " = trunc i128 " << vn128 << " to " << it << "\n  "
           << mi << " = trunc i128 " << vm128 << " to " << it << "\n  "
           << ai << " = trunc i128 " << va128 << " to " << it << "\n  "
           << nf << " = bitcast " << it << " " << ni << " to " << ft << "\n  "
           << mf << " = bitcast " << it << " " << mi << " to " << ft << "\n  "
           << af << " = bitcast " << it << " " << ai << " to " << ft << "\n  ";
        std::string use_n = nf, use_a = af;
        if (nn) {
            use_n = fn.next_reg("fnn");
            os << use_n << " = fneg " << ft << " " << nf << "\n  ";
        }
        if (na) {
            use_a = fn.next_reg("fna");
            os << use_a << " = fneg " << ft << " " << af << "\n  ";
        }
        os << rf << " = call " << ft << " @llvm.fma." << (is_double ? "f64" : "f32")
           << "(" << ft << " " << use_n << ", " << ft << " " << mf << ", " << ft << " " << use_a << ")\n  "
           << ri << " = bitcast " << ft << " " << rf << " to " << it << "\n  "
           << r128 << " = zext " << it << " " << ri << " to i128";

        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }

    static void emit_inline_scalar_sqrt(Function& fn, std::vector<Instruction>& out,
                                        bool is_double, int d, int n) {
        std::string vn128;
        out.push_back(make_vreg_read(fn, n, vn128));
        const char* it = is_double ? "i64" : "i32";
        const char* ft = is_double ? "double" : "float";
        std::string ni = fn.next_reg("fni"), nf = fn.next_reg("fnf");
        std::string rf = fn.next_reg("frf"), ri = fn.next_reg("fri"), r128 = fn.next_reg("frq");
        std::ostringstream os;
        os << "; [FPCRFastFPLoweringPass] inline scalar FSQRT\n  "
           << ni << " = trunc i128 " << vn128 << " to " << it << "\n  "
           << nf << " = bitcast " << it << " " << ni << " to " << ft << "\n  "
           << rf << " = call " << ft << " @llvm.sqrt." << (is_double ? "f64" : "f32") << "(" << ft << " " << nf << ")\n  "
           << ri << " = bitcast " << ft << " " << rf << " to " << it << "\n  "
           << r128 << " = zext " << it << " " << ri << " to i128";
        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }

    static void emit_inline_vec_fop(Function& fn, std::vector<Instruction>& out,
                                    int sz, int fop, int Q, int d, int n, int m, int acc, int neg_n) {
        std::string vn128, vm128, va128;
        out.push_back(make_vreg_read(fn, n, vn128));
        if (fop != FpOp::FOP_SQRT) {
            out.push_back(make_vreg_read(fn, m, vm128));
        }
        if (fop == FpOp::FOP_MADD && acc >= 0) {
            out.push_back(make_vreg_read(fn, acc, va128));
        }

        const char* vt = sz ? "<2 x double>" : "<4 x float>";
        const char* isfx = sz ? "v2f64" : "v4f32";
        std::string nv = fn.next_reg("vnf"), mv = fn.next_reg("vmf"), av = fn.next_reg("vaf");
        std::string rv = fn.next_reg("vrf"), r128 = fn.next_reg("vrq");

        std::ostringstream os;
        os << "; [FPCRFastFPLoweringPass] inline vector FP op\n  "
           << nv << " = bitcast i128 " << vn128 << " to " << vt << "\n  ";
        if (fop != FpOp::FOP_SQRT) {
            os << mv << " = bitcast i128 " << vm128 << " to " << vt << "\n  ";
        }
        if (fop == FpOp::FOP_ADD) {
            os << rv << " = fadd " << vt << " " << nv << ", " << mv;
        } else if (fop == FpOp::FOP_SUB) {
            os << rv << " = fsub " << vt << " " << nv << ", " << mv;
        } else if (fop == FpOp::FOP_MUL) {
            os << rv << " = fmul " << vt << " " << nv << ", " << mv;
        } else if (fop == FpOp::FOP_DIV) {
            os << rv << " = fdiv " << vt << " " << nv << ", " << mv;
        } else if (fop == FpOp::FOP_SQRT) {
            os << rv << " = call " << vt << " @llvm.sqrt." << isfx << "(" << vt << " " << nv << ")";
        } else if (fop == FpOp::FOP_MAXNM) {
            os << rv << " = call " << vt << " @llvm.maxnum." << isfx << "(" << vt << " " << nv << ", " << vt << " " << mv << ")";
        } else if (fop == FpOp::FOP_MINNM) {
            os << rv << " = call " << vt << " @llvm.minnum." << isfx << "(" << vt << " " << nv << ", " << vt << " " << mv << ")";
        } else if (fop == FpOp::FOP_MADD) {
            os << av << " = bitcast i128 " << va128 << " to " << vt << "\n  ";
            std::string use_n = nv;
            if (neg_n) {
                use_n = fn.next_reg("vnn");
                os << use_n << " = fneg " << vt << " " << nv << "\n  ";
            }
            os << rv << " = call " << vt << " @llvm.fma." << isfx
               << "(" << vt << " " << use_n << ", " << vt << " " << mv << ", " << vt << " " << av << ")";
        }
        if (Q) {
            os << "\n  " << r128 << " = bitcast " << vt << " " << rv << " to i128";
        } else {
            std::string raw128 = fn.next_reg("vrw");
            std::string lo64 = fn.next_reg("vlo");
            os << "\n  " << raw128 << " = bitcast " << vt << " " << rv << " to i128\n  "
               << lo64 << " = trunc i128 " << raw128 << " to i64\n  "
               << r128 << " = zext i64 " << lo64 << " to i128";
        }
        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }
};


} // namespace

bool FPCRFastFPLoweringPass::run(Module& M) {
    FPCRFastFPLoweringPassImpl impl(enabled_);
    return impl.run(M);
}

} // namespace ultrs::passes

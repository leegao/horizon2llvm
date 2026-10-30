// UltraTears LLVM Pass Implementation: Pass 5 - SIMDPairwiseAndConversionInlinePass (tools/lifter/pass_simd_pairwise_conversion_inline.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class SIMDPairwiseAndConversionInlinePassImpl {
public:
    explicit SIMDPairwiseAndConversionInlinePassImpl(bool enable_fast_fp = true) : enable_fast_fp_(enable_fast_fp) {}

    static const char* static_name() { return "simd-pairwise-conversion-inline-pass"; }
    const char* pass_name() const { return "simd-pairwise-conversion-inline-pass"; }

    bool run(Module& M) {
        if (!enable_fast_fp_) return false;
        bool changed = false;

        for (auto& fn : M.functions) {
            if (!fn->hle_symbol.empty() || fn->modifies_fpcr) continue;

            for (auto& bb : fn->blocks) {
                std::vector<Instruction> out;
                out.reserve(bb->insts.size() * 2);

                for (auto& inst : bb->insts) {
                    if (inst.dead || inst.tag != OpTag::RawIR) {
                        out.push_back(std::move(inst));
                        continue;
                    }

                    int a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0, a8 = 0;

                    // 1. @rt_vec_fpair(ptr %c, i32 <sz>, i32 <fop>, i32 <Q>, i32 <d>, i32 <n>, i32 <m>)
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_vec_fpair(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3, &a4, &a5) == 6 &&
                        (a1 == FpOp::FOP_ADD || a1 == FpOp::FOP_MAXNM || a1 == FpOp::FOP_MINNM) &&
                        !(a0 == 1 && a2 == 0)) {
                        emit_vec_fpair(*fn, out, a0, a1, a2, a3, a4, a5);
                        changed = true;
                        continue;
                    }

                    // 2. @rt_scalar_fpair(ptr %c, i32 <sz>, i32 <fop>, i32 <d>, i32 <n>)
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_scalar_fpair(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3) == 4 &&
                        (a1 == FpOp::FOP_ADD || a1 == FpOp::FOP_MAXNM || a1 == FpOp::FOP_MINNM)) {
                        emit_scalar_fpair(*fn, out, a0, a1, a2, a3);
                        changed = true;
                        continue;
                    }

                    // 3. @rt_indexed_fp(ptr %c, i32 <fsz>, i32 <Q>, i32 <scalar>, i32 <opc>, i32 <U>, i32 <d>, i32 <n>, i32 <m>, i32 <idx>)
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_indexed_fp(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8) == 9 &&
                        ((a3 == 9 && a4 == 0) || a3 == 1 || a3 == 5)) {
                        emit_indexed_fp(*fn, out, a0, a1, a2, a3, a5, a6, a7, a8);
                        changed = true;
                        continue;
                    }

                    // 4. @rt_vec_fcmp(ptr %c, i32 <sz>, i32 <cop>, i32 <Q>, i32 <scalar>, i32 <d>, i32 <n>, i32 <m>, i32 <zero>, i32 <swap>)
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_vec_fcmp(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8) == 9 &&
                        a1 >= 0 && a1 <= 4) {
                        emit_vec_fcmp(*fn, out, a0, a1, a2, a3, a4, a5, a6, a7, a8);
                        changed = true;
                        continue;
                    }

                    // 5. @rt_fp_dp1(ptr %c, i32 <t>, i32 <opc>, i32 <d>, i32 <n>) -> FCVT S<->D and FRINT*
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_fp_dp1(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3) == 4 &&
                        (a0 == 0 || a0 == 1) &&
                        ((a0 == 0 && a1 == 5) || (a0 == 1 && a1 == 4) ||
                         a1 == 8 || a1 == 9 || a1 == 10 || a1 == 11 || a1 == 12 || a1 == 14 || a1 == 15)) {
                        emit_fp_dp1(*fn, out, a0, a1, a2, a3);
                        changed = true;
                        continue;
                    }

                    // 6. @rt_vec_frint(ptr %c, i32 <sz>, i32 <Q>, i32 <rm>, i32 <d>, i32 <n>)
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_vec_frint(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3, &a4) == 5 &&
                        a2 >= 0 && a2 <= 6 && !(a0 == 1 && a1 == 0)) {
                        emit_vec_frint(*fn, out, a0, a1, a2, a3, a4);
                        changed = true;
                        continue;
                    }

                    // 7. @rt_vec_fcvtl(ptr %c, i32 1, i32 <Q>, i32 <d>, i32 <n>) -> 2S to 2D
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_vec_fcvtl(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3) == 4 && a0 == 1) {
                        emit_vec_fcvtl(*fn, out, a1, a2, a3);
                        changed = true;
                        continue;
                    }

                    // 8. @rt_vec_fcvtn(ptr %c, i32 1, i32 <Q>, i32 0, i32 <d>, i32 <n>) -> 2D to 2S
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_vec_fcvtn(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3, &a4) == 5 && a0 == 1 && a2 == 0 && (a1 == 0 || a1 == 1)) {
                        emit_vec_fcvtn(*fn, out, a1, a3, a4);
                        changed = true;
                        continue;
                    }

                    // 9. @rt_fp_fromint(ptr %c, i32 <t>, i32 <d>, i64 <src>, i32 <ib>, i32 <is_u>, i32 <fb>)
                    if (inst.text.find("call void @rt_fp_fromint(") != std::string::npos) {
                        char src_op[64] = {};
                        int t = 0, d = 0, ib = 0, is_u = 0, fb = 0;
                        if (std::sscanf(inst.text.c_str(),
                                        "call void @rt_fp_fromint(ptr %%c, i32 %d, i32 %d, i64 %63[^,], i32 %d, i32 %d, i32 %d)",
                                        &t, &d, src_op, &ib, &is_u, &fb) == 6 &&
                            (t == 0 || t == 1) && (ib == 32 || ib == 64) && fb == 0) {
                            emit_fp_fromint(*fn, out, t, d, src_op, ib, is_u);
                            changed = true;
                            continue;
                        }
                    }

                    // 10. %r = call i64 @rt_fp_toint(ptr %c, i32 <t>, i32 <n>, i32 3, i32 0, i32 <ib>, i32 <is_u>)
                    if (inst.text.find("call i64 @rt_fp_toint(") != std::string::npos) {
                        char dst_reg[64] = {};
                        int t = 0, n = 0, rm = 0, fb = 0, ib = 0, is_u = 0;
                        if (std::sscanf(inst.text.c_str(),
                                        "%63s = call i64 @rt_fp_toint(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d)",
                                        dst_reg, &t, &n, &rm, &fb, &ib, &is_u) == 7 &&
                            (t == 0 || t == 1) && rm == 3 && fb == 0 && (ib == 32 || ib == 64)) {
                            emit_fp_toint(*fn, out, dst_reg, t, n, ib, is_u);
                            changed = true;
                            continue;
                        }
                    }

                    // 11. @rt_vec_fcvt_int(ptr %c, i32 <sz>, i32 <to_int>, i32 <Q>, i32 <scalar>, i32 <rm>, i32 0, i32 <U>, i32 <d>, i32 <n>)
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_vec_fcvt_int(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 0, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7) == 8 &&
                        (a1 == 0 || (a1 == 1 && a4 == 3)) && !(a3 == 0 && a0 == 1 && a2 == 0)) {
                        emit_vec_fcvt_int(*fn, out, a0, a1, a2, a3, a5, a6, a7);
                        changed = true;
                        continue;
                    }

                    // 12. @rt_sl_gen shift/narrow/widen integer vector ops
                    if (inst.text.find("call void @rt_sl_gen(") != std::string::npos) {
                        int op = 0, sz = 0, ne = 0, d = 0, szd = 0, offd = 0, keep = 0;
                        int n = 0, szn = 0, offn = 0, m = 0, szm = 0, offm = 0, bidx = 0;
                        unsigned long long bimm = 0;
                        if (std::sscanf(inst.text.c_str(),
                                        "call void @rt_sl_gen(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i64 %llu)",
                                        &op, &sz, &ne, &d, &szd, &offd, &keep, &n, &szn, &offn, &m, &szm, &offm, &bidx, &bimm) == 15) {
                            if (try_emit_inline_sl_gen(*fn, out, op, sz, ne, d, szd, offd, keep, n, szn, offn, m, szm, offm, bidx, bimm)) {
                                changed = true;
                                continue;
                            }
                        }
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

    static Instruction make_vreg_read(Function& fn, int vreg, std::string& out_reg) {
        out_reg = fn.next_reg("svr");
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
        std::string p = fn.next_reg("svw");
        Instruction inst;
        inst.tag = OpTag::CpuRegWrite;
        inst.arch_reg = 64 + vreg;
        inst.op0 = Value::reg(TypeKind::Int128, val_i128);
        inst.text = p + " = getelementptr inbounds i8, ptr %c, i64 " + std::to_string(off) + "\n  " +
                    "store i128 " + val_i128 + ", ptr " + p + ", align 16, !alias.scope !3, !noalias !4";
        return inst;
    }

    static void emit_vec_fpair(Function& fn, std::vector<Instruction>& out,
                               int sz, int fop, int Q, int d, int n, int m) {
        std::string vn128, vm128;
        out.push_back(make_vreg_read(fn, n, vn128));
        out.push_back(make_vreg_read(fn, m, vm128));

        const char* vt = sz ? "<2 x double>" : "<4 x float>";
        const char* isfx = sz ? "v2f64" : "v4f32";
        const char* ev_mask = sz ? "<2 x i32> <i32 0, i32 2>"
                            : (Q ? "<4 x i32> <i32 0, i32 2, i32 4, i32 6>"
                                 : "<4 x i32> <i32 0, i32 4, i32 0, i32 0>");
        const char* od_mask = sz ? "<2 x i32> <i32 1, i32 3>"
                            : (Q ? "<4 x i32> <i32 1, i32 3, i32 5, i32 7>"
                                 : "<4 x i32> <i32 1, i32 5, i32 0, i32 0>");

        std::string nv = fn.next_reg("vpn"), mv = fn.next_reg("vpm");
        std::string ev = fn.next_reg("vpev"), od = fn.next_reg("vpod");
        std::string rv = fn.next_reg("vpr"), r128 = fn.next_reg("vpq");

        std::ostringstream os;
        os << "; [SIMDPairwiseAndConversionInlinePass] inline vector pairwise FP\n  "
           << nv << " = bitcast i128 " << vn128 << " to " << vt << "\n  "
           << mv << " = bitcast i128 " << vm128 << " to " << vt << "\n  "
           << ev << " = shufflevector " << vt << " " << nv << ", " << vt << " " << mv << ", " << ev_mask << "\n  "
           << od << " = shufflevector " << vt << " " << nv << ", " << vt << " " << mv << ", " << od_mask << "\n  ";
        if (fop == FpOp::FOP_ADD) {
            os << rv << " = fadd " << vt << " " << ev << ", " << od;
        } else if (fop == FpOp::FOP_MAXNM) {
            os << rv << " = call " << vt << " @llvm.maxnum." << isfx << "(" << vt << " " << ev << ", " << vt << " " << od << ")";
        } else {
            os << rv << " = call " << vt << " @llvm.minnum." << isfx << "(" << vt << " " << ev << ", " << vt << " " << od << ")";
        }
        if (Q) {
            os << "\n  " << r128 << " = bitcast " << vt << " " << rv << " to i128";
        } else {
            std::string raw128 = fn.next_reg("vprw"), lo64 = fn.next_reg("vplo");
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

    static void emit_scalar_fpair(Function& fn, std::vector<Instruction>& out,
                                  int sz, int fop, int d, int n) {
        std::string vn128;
        out.push_back(make_vreg_read(fn, n, vn128));
        const char* vt = sz ? "<2 x double>" : "<4 x float>";
        const char* ft = sz ? "double" : "float";
        const char* it = sz ? "i64" : "i32";
        const char* sfx = sz ? "f64" : "f32";

        std::string nv = fn.next_reg("spn"), e0 = fn.next_reg("spe0"), e1 = fn.next_reg("spe1");
        std::string rf = fn.next_reg("sprf"), ri = fn.next_reg("spri"), r128 = fn.next_reg("sprq");
        std::ostringstream os;
        os << "; [SIMDPairwiseAndConversionInlinePass] inline scalar pairwise FP\n  "
           << nv << " = bitcast i128 " << vn128 << " to " << vt << "\n  "
           << e0 << " = extractelement " << vt << " " << nv << ", i32 0\n  "
           << e1 << " = extractelement " << vt << " " << nv << ", i32 1\n  ";
        if (fop == FpOp::FOP_ADD) {
            os << rf << " = fadd " << ft << " " << e0 << ", " << e1;
        } else if (fop == FpOp::FOP_MAXNM) {
            os << rf << " = call " << ft << " @llvm.maxnum." << sfx << "(" << ft << " " << e0 << ", " << ft << " " << e1 << ")";
        } else {
            os << rf << " = call " << ft << " @llvm.minnum." << sfx << "(" << ft << " " << e0 << ", " << ft << " " << e1 << ")";
        }
        os << "\n  " << ri << " = bitcast " << ft << " " << rf << " to " << it
           << "\n  " << r128 << " = zext " << it << " " << ri << " to i128";
        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }

    static void emit_indexed_fp(Function& fn, std::vector<Instruction>& out,
                                int fsz, int Q, int scalar, int opc, int d, int n, int m, int idx) {
        std::string vn128, vm128, vd128;
        if (opc == 1 || opc == 5) {
            out.push_back(make_vreg_read(fn, d, vd128));
        }
        out.push_back(make_vreg_read(fn, n, vn128));
        out.push_back(make_vreg_read(fn, m, vm128));

        const char* vt = fsz ? "<2 x double>" : "<4 x float>";
        const char* ft = fsz ? "double" : "float";
        const char* it = fsz ? "i64" : "i32";

        std::ostringstream os;
        os << "; [SIMDPairwiseAndConversionInlinePass] inline indexed FP mul/FMA\n  ";
        std::string r128 = fn.next_reg("ixq");

        if (scalar) {
            std::string nv = fn.next_reg("ixnv"), mv = fn.next_reg("ixmv");
            std::string nf = fn.next_reg("ixnf"), mf = fn.next_reg("ixmf");
            std::string rf = fn.next_reg("ixrf"), ri = fn.next_reg("ixri");
            os << nv << " = bitcast i128 " << vn128 << " to " << vt << "\n  "
               << mv << " = bitcast i128 " << vm128 << " to " << vt << "\n  "
               << nf << " = extractelement " << vt << " " << nv << ", i32 0\n  "
               << mf << " = extractelement " << vt << " " << mv << ", i32 " << idx << "\n  ";
            if (opc == 9) {
                os << rf << " = fmul " << ft << " " << nf << ", " << mf << "\n  ";
            } else {
                std::string dv = fn.next_reg("ixdv"), df = fn.next_reg("ixdf");
                os << dv << " = bitcast i128 " << vd128 << " to " << vt << "\n  "
                   << df << " = extractelement " << vt << " " << dv << ", i32 0\n  ";
                std::string use_n = nf;
                if (opc == 5) {
                    use_n = fn.next_reg("ixnn");
                    os << use_n << " = fneg " << ft << " " << nf << "\n  ";
                }
                os << rf << " = call " << ft << " @llvm.fma." << (fsz ? "f64" : "f32")
                   << "(" << ft << " " << use_n << ", " << ft << " " << mf << ", " << ft << " " << df << ")\n  ";
            }
            os << ri << " = bitcast " << ft << " " << rf << " to " << it << "\n  "
               << r128 << " = zext " << it << " " << ri << " to i128";
        } else {
            std::string nv = fn.next_reg("ixnv"), mv = fn.next_reg("ixmv");
            std::string splat = fn.next_reg("ixsp"), rv = fn.next_reg("ixrv");
            std::string mask_str = fsz ? ("<2 x i32> <i32 " + std::to_string(idx) + ", i32 " + std::to_string(idx) + ">")
                                       : ("<4 x i32> <i32 " + std::to_string(idx) + ", i32 " + std::to_string(idx) +
                                          ", i32 " + std::to_string(idx) + ", i32 " + std::to_string(idx) + ">");
            os << nv << " = bitcast i128 " << vn128 << " to " << vt << "\n  "
               << mv << " = bitcast i128 " << vm128 << " to " << vt << "\n  "
               << splat << " = shufflevector " << vt << " " << mv << ", " << vt << " " << mv << ", " << mask_str << "\n  ";
            if (opc == 9) {
                os << rv << " = fmul " << vt << " " << nv << ", " << splat;
            } else {
                std::string dv = fn.next_reg("ixdv");
                os << dv << " = bitcast i128 " << vd128 << " to " << vt << "\n  ";
                std::string use_n = nv;
                if (opc == 5) {
                    use_n = fn.next_reg("ixnn");
                    os << use_n << " = fneg " << vt << " " << nv << "\n  ";
                }
                os << rv << " = call " << vt << " @llvm.fma." << (fsz ? "v2f64" : "v4f32")
                   << "(" << vt << " " << use_n << ", " << vt << " " << splat << ", " << vt << " " << dv << ")";
            }
            if (Q) {
                os << "\n  " << r128 << " = bitcast " << vt << " " << rv << " to i128";
            } else {
                std::string raw128 = fn.next_reg("ixrw"), lo64 = fn.next_reg("ixlo");
                os << "\n  " << raw128 << " = bitcast " << vt << " " << rv << " to i128\n  "
                   << lo64 << " = trunc i128 " << raw128 << " to i64\n  "
                   << r128 << " = zext i64 " << lo64 << " to i128";
            }
        }
        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }

    static void emit_vec_fcmp(Function& fn, std::vector<Instruction>& out,
                              int sz, int cop, int Q, int scalar, int d, int n, int m, int zero, int swap) {
        std::string vn128, vm128;
        out.push_back(make_vreg_read(fn, n, vn128));
        if (!zero) {
            out.push_back(make_vreg_read(fn, m, vm128));
        }
        const char* pred = (cop == 0) ? "oeq" : (cop == 1 || cop == 3) ? "oge" : "ogt";
        std::string r128 = fn.next_reg("fcmq");
        std::ostringstream os;
        os << "; [SIMDPairwiseAndConversionInlinePass] inline vector/scalar FCMP\n  ";

        if (scalar) {
            const char* ft = sz ? "double" : "float";
            const char* it = sz ? "i64" : "i32";
            const char* sfx = sz ? "f64" : "f32";
            std::string ni = fn.next_reg("fcni"), nf = fn.next_reg("fcnf");
            os << ni << " = trunc i128 " << vn128 << " to " << it << "\n  "
               << nf << " = bitcast " << it << " " << ni << " to " << ft << "\n  ";
            std::string lhs = nf, rhs = "0.000000e+00";
            if (!zero) {
                std::string mi = fn.next_reg("fcmi"), mf = fn.next_reg("fcmf");
                os << mi << " = trunc i128 " << vm128 << " to " << it << "\n  "
                   << mf << " = bitcast " << it << " " << mi << " to " << ft << "\n  ";
                rhs = mf;
            }
            if (swap) std::swap(lhs, rhs);
            if (cop >= 3) {
                std::string al = fn.next_reg("fcal"), ar = fn.next_reg("fcar");
                os << al << " = call " << ft << " @llvm.fabs." << sfx << "(" << ft << " " << lhs << ")\n  "
                   << ar << " = call " << ft << " @llvm.fabs." << sfx << "(" << ft << " " << rhs << ")\n  ";
                lhs = al;
                rhs = ar;
            }
            std::string cmp = fn.next_reg("fcc"), sx = fn.next_reg("fcsx");
            os << cmp << " = fcmp " << pred << " " << ft << " " << lhs << ", " << rhs << "\n  "
               << sx << " = sext i1 " << cmp << " to " << it << "\n  "
               << r128 << " = zext " << it << " " << sx << " to i128";
        } else {
            const char* vt = sz ? "<2 x double>" : "<4 x float>";
            const char* ivt = sz ? "<2 x i64>" : "<4 x i32>";
            const char* bvt = sz ? "<2 x i1>" : "<4 x i1>";
            const char* sfx = sz ? "v2f64" : "v4f32";
            std::string nv = fn.next_reg("fcnv");
            os << nv << " = bitcast i128 " << vn128 << " to " << vt << "\n  ";
            std::string lhs = nv, rhs = "zeroinitializer";
            if (!zero) {
                std::string mv = fn.next_reg("fcmv");
                os << mv << " = bitcast i128 " << vm128 << " to " << vt << "\n  ";
                rhs = mv;
            }
            if (swap) std::swap(lhs, rhs);
            if (cop >= 3) {
                std::string al = fn.next_reg("fcal"), ar = fn.next_reg("fcar");
                os << al << " = call " << vt << " @llvm.fabs." << sfx << "(" << vt << " " << lhs << ")\n  "
                   << ar << " = call " << vt << " @llvm.fabs." << sfx << "(" << vt << " " << rhs << ")\n  ";
                lhs = al;
                rhs = ar;
            }
            std::string cmp = fn.next_reg("fcc"), sx = fn.next_reg("fcsx");
            os << cmp << " = fcmp " << pred << " " << vt << " " << lhs << ", " << rhs << "\n  "
               << sx << " = sext " << bvt << " " << cmp << " to " << ivt;
            if (Q) {
                os << "\n  " << r128 << " = bitcast " << ivt << " " << sx << " to i128";
            } else {
                std::string raw128 = fn.next_reg("fcrw"), lo64 = fn.next_reg("fclo");
                os << "\n  " << raw128 << " = bitcast " << ivt << " " << sx << " to i128\n  "
                   << lo64 << " = trunc i128 " << raw128 << " to i64\n  "
                   << r128 << " = zext i64 " << lo64 << " to i128";
            }
        }
        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }

    static const char* frint_intrinsic_base(int rm) {
        switch (rm) {
            case 0: return "rint";      // FRINTN (round to nearest ties to even under default FPCR)
            case 1: return "ceil";      // FRINTP (+inf)
            case 2: return "floor";     // FRINTM (-inf)
            case 3: return "trunc";     // FRINTZ (towards zero)
            case 4: return "round";     // FRINTA (ties away from zero)
            case 5: return "rint";      // FRINTX
            case 6: return "nearbyint"; // FRINTI
            default: return "rint";
        }
    }

    static void emit_fp_dp1(Function& fn, std::vector<Instruction>& out, int t, int opc, int d, int n) {
        std::string vn128;
        out.push_back(make_vreg_read(fn, n, vn128));
        std::string r128 = fn.next_reg("dp1q");
        std::ostringstream os;
        os << "; [SIMDPairwiseAndConversionInlinePass] inline scalar FCVT/FRINT\n  ";

        if (t == 0 && opc == 5) { // FCVT Dd, Sn (float -> double)
            std::string ni = fn.next_reg("fni"), nf = fn.next_reg("fnf");
            std::string df = fn.next_reg("fdf"), di = fn.next_reg("fdi");
            os << ni << " = trunc i128 " << vn128 << " to i32\n  "
               << nf << " = bitcast i32 " << ni << " to float\n  "
               << df << " = fpext float " << nf << " to double\n  "
               << di << " = bitcast double " << df << " to i64\n  "
               << r128 << " = zext i64 " << di << " to i128";
        } else if (t == 1 && opc == 4) { // FCVT Sd, Dn (double -> float)
            std::string ni = fn.next_reg("fni"), nf = fn.next_reg("fnf");
            std::string sf = fn.next_reg("fsf"), si = fn.next_reg("fsi");
            os << ni << " = trunc i128 " << vn128 << " to i64\n  "
               << nf << " = bitcast i64 " << ni << " to double\n  "
               << sf << " = fptrunc double " << nf << " to float\n  "
               << si << " = bitcast float " << sf << " to i32\n  "
               << r128 << " = zext i32 " << si << " to i128";
        } else { // FRINT*
            int rm = (opc == 8) ? 0 : (opc == 9) ? 1 : (opc == 10) ? 2 : (opc == 11) ? 3
                   : (opc == 12) ? 4 : (opc == 14) ? 5 : 6;
            const char* it = t ? "i64" : "i32";
            const char* ft = t ? "double" : "float";
            const char* sfx = t ? "f64" : "f32";
            std::string ni = fn.next_reg("fni"), nf = fn.next_reg("fnf");
            std::string rf = fn.next_reg("frf"), ri = fn.next_reg("fri");
            os << ni << " = trunc i128 " << vn128 << " to " << it << "\n  "
               << nf << " = bitcast " << it << " " << ni << " to " << ft << "\n  "
               << rf << " = call " << ft << " @llvm." << frint_intrinsic_base(rm) << "." << sfx << "(" << ft << " " << nf << ")\n  "
               << ri << " = bitcast " << ft << " " << rf << " to " << it << "\n  "
               << r128 << " = zext " << it << " " << ri << " to i128";
        }
        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }

    static void emit_vec_frint(Function& fn, std::vector<Instruction>& out, int sz, int Q, int rm, int d, int n) {
        std::string vn128;
        out.push_back(make_vreg_read(fn, n, vn128));
        const char* vt = sz ? "<2 x double>" : "<4 x float>";
        const char* sfx = sz ? "v2f64" : "v4f32";
        std::string nv = fn.next_reg("vrn"), rv = fn.next_reg("vrr"), r128 = fn.next_reg("vrq");
        std::ostringstream os;
        os << "; [SIMDPairwiseAndConversionInlinePass] inline vector FRINT\n  "
           << nv << " = bitcast i128 " << vn128 << " to " << vt << "\n  "
           << rv << " = call " << vt << " @llvm." << frint_intrinsic_base(rm) << "." << sfx << "(" << vt << " " << nv << ")";
        if (Q) {
            os << "\n  " << r128 << " = bitcast " << vt << " " << rv << " to i128";
        } else {
            std::string raw128 = fn.next_reg("vrrw"), lo64 = fn.next_reg("vrlo");
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

    static void emit_vec_fcvtl(Function& fn, std::vector<Instruction>& out, int Q, int d, int n) {
        std::string vn128;
        out.push_back(make_vreg_read(fn, n, vn128));
        std::string nv = fn.next_reg("cvln"), sub = fn.next_reg("cvls");
        std::string ext = fn.next_reg("cvle"), r128 = fn.next_reg("cvlq");
        const char* mask = Q ? "<2 x i32> <i32 2, i32 3>" : "<2 x i32> <i32 0, i32 1>";
        std::ostringstream os;
        os << "; [SIMDPairwiseAndConversionInlinePass] inline FCVTL (2S -> 2D)\n  "
           << nv << " = bitcast i128 " << vn128 << " to <4 x float>\n  "
           << sub << " = shufflevector <4 x float> " << nv << ", <4 x float> " << nv << ", " << mask << "\n  "
           << ext << " = fpext <2 x float> " << sub << " to <2 x double>\n  "
           << r128 << " = bitcast <2 x double> " << ext << " to i128";
        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }

    static void emit_vec_fcvtn(Function& fn, std::vector<Instruction>& out, int Q, int d, int n) {
        std::string vn128, vd128;
        if (Q) out.push_back(make_vreg_read(fn, d, vd128));
        out.push_back(make_vreg_read(fn, n, vn128));
        std::string nv = fn.next_reg("cvnn"), tr = fn.next_reg("cvnt"), n64 = fn.next_reg("cvn64");
        std::string r128 = fn.next_reg("cvnq");
        std::ostringstream os;
        os << "; [SIMDPairwiseAndConversionInlinePass] inline FCVTN (2D -> 2S)\n  "
           << nv << " = bitcast i128 " << vn128 << " to <2 x double>\n  "
           << tr << " = fptrunc <2 x double> " << nv << " to <2 x float>\n  "
           << n64 << " = bitcast <2 x float> " << tr << " to i64\n  ";
        if (!Q) {
            os << r128 << " = zext i64 " << n64 << " to i128";
        } else {
            std::string lo64 = fn.next_reg("cvnlo"), lo128 = fn.next_reg("cvnlq");
            std::string hi128 = fn.next_reg("cvnhi"), hish = fn.next_reg("cvnhs");
            os << lo64 << " = trunc i128 " << vd128 << " to i64\n  "
               << lo128 << " = zext i64 " << lo64 << " to i128\n  "
               << hi128 << " = zext i64 " << n64 << " to i128\n  "
               << hish << " = shl i128 " << hi128 << ", 64\n  "
               << r128 << " = or i128 " << lo128 << ", " << hish;
        }
        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }

    static void emit_fp_fromint(Function& fn, std::vector<Instruction>& out,
                                int t, int d, const std::string& src_op, int ib, int is_u) {
        const char* ft = t ? "double" : "float";
        const char* fit = t ? "i64" : "i32";
        const char* cop = is_u ? "uitofp" : "sitofp";
        std::string rf = fn.next_reg("itff"), ri = fn.next_reg("itfi"), r128 = fn.next_reg("itfq");
        std::ostringstream os;
        os << "; [SIMDPairwiseAndConversionInlinePass] inline scalar SCVTF/UCVTF\n  ";
        if (ib == 32) {
            std::string s32 = fn.next_reg("itf32");
            os << s32 << " = trunc i64 " << src_op << " to i32\n  "
               << rf << " = " << cop << " i32 " << s32 << " to " << ft << "\n  ";
        } else {
            os << rf << " = " << cop << " i64 " << src_op << " to " << ft << "\n  ";
        }
        os << ri << " = bitcast " << ft << " " << rf << " to " << fit << "\n  "
           << r128 << " = zext " << fit << " " << ri << " to i128";
        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }

    static void emit_fp_toint(Function& fn, std::vector<Instruction>& out,
                              const std::string& dst_reg, int t, int n, int ib, int is_u) {
        std::string vn128;
        out.push_back(make_vreg_read(fn, n, vn128));
        const char* ft = t ? "double" : "float";
        const char* fit = t ? "i64" : "i32";
        const char* fsfx = t ? "f64" : "f32";
        const char* intr = is_u ? "fptoui.sat" : "fptosi.sat";
        std::string ni = fn.next_reg("ftini"), nf = fn.next_reg("ftinf");
        std::ostringstream os;
        os << "; [SIMDPairwiseAndConversionInlinePass] inline scalar FCVTZS/FCVTZU\n  "
           << ni << " = trunc i128 " << vn128 << " to " << fit << "\n  "
           << nf << " = bitcast " << fit << " " << ni << " to " << ft << "\n  ";
        if (ib == 64) {
            os << dst_reg << " = call i64 @llvm." << intr << ".i64." << fsfx << "(" << ft << " " << nf << ")";
        } else {
            std::string s32 = fn.next_reg("ftis32");
            os << s32 << " = call i32 @llvm." << intr << ".i32." << fsfx << "(" << ft << " " << nf << ")\n  "
               << dst_reg << " = zext i32 " << s32 << " to i64";
        }
        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.dst_reg = dst_reg;
        calc.dst_type = TypeKind::Int64;
        calc.text = os.str();
        out.push_back(std::move(calc));
    }

    static void emit_vec_fcvt_int(Function& fn, std::vector<Instruction>& out,
                                  int sz, int to_int, int Q, int scalar, int U, int d, int n) {
        std::string vn128;
        out.push_back(make_vreg_read(fn, n, vn128));
        std::string r128 = fn.next_reg("vciq");
        std::ostringstream os;
        os << "; [SIMDPairwiseAndConversionInlinePass] inline vector/scalar FP<->Int conversion\n  ";

        if (scalar) {
            const char* ft = sz ? "double" : "float";
            const char* it = sz ? "i64" : "i32";
            const char* fsfx = sz ? "f64" : "f32";
            std::string ni = fn.next_reg("vcini");
            os << ni << " = trunc i128 " << vn128 << " to " << it << "\n  ";
            if (!to_int) {
                std::string rf = fn.next_reg("vcirf"), ri = fn.next_reg("vciri");
                os << rf << " = " << (U ? "uitofp " : "sitofp ") << it << " " << ni << " to " << ft << "\n  "
                   << ri << " = bitcast " << ft << " " << rf << " to " << it << "\n  "
                   << r128 << " = zext " << it << " " << ri << " to i128";
            } else {
                std::string nf = fn.next_reg("vcinf"), ri = fn.next_reg("vciri");
                const char* intr = U ? "fptoui.sat" : "fptosi.sat";
                os << nf << " = bitcast " << it << " " << ni << " to " << ft << "\n  "
                   << ri << " = call " << it << " @llvm." << intr << "." << it << "." << fsfx << "(" << ft << " " << nf << ")\n  "
                   << r128 << " = zext " << it << " " << ri << " to i128";
            }
        } else {
            const char* fvt = sz ? "<2 x double>" : "<4 x float>";
            const char* ivt = sz ? "<2 x i64>" : "<4 x i32>";
            std::string rv = fn.next_reg("vcirv");
            if (!to_int) {
                std::string nv = fn.next_reg("vcinv");
                os << nv << " = bitcast i128 " << vn128 << " to " << ivt << "\n  "
                   << rv << " = " << (U ? "uitofp " : "sitofp ") << ivt << " " << nv << " to " << fvt;
            } else {
                std::string nv = fn.next_reg("vcinv");
                const char* intr = U ? (sz ? "fptoui.sat.v2i64.v2f64" : "fptoui.sat.v4i32.v4f32")
                                     : (sz ? "fptosi.sat.v2i64.v2f64" : "fptosi.sat.v4i32.v4f32");
                os << nv << " = bitcast i128 " << vn128 << " to " << fvt << "\n  "
                   << rv << " = call " << ivt << " @llvm." << intr << "(" << fvt << " " << nv << ")";
            }
            const char* dst_vt = to_int ? ivt : fvt;
            if (Q) {
                os << "\n  " << r128 << " = bitcast " << dst_vt << " " << rv << " to i128";
            } else {
                std::string raw128 = fn.next_reg("vcirw"), lo64 = fn.next_reg("vcilo");
                os << "\n  " << raw128 << " = bitcast " << dst_vt << " " << rv << " to i128\n  "
                   << lo64 << " = trunc i128 " << raw128 << " to i64\n  "
                   << r128 << " = zext i64 " << lo64 << " to i128";
            }
        }
        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }

    static std::string make_splat_const(int sz, int lanes, uint64_t val) {
        const char* et = (sz == 0) ? "i8" : (sz == 1) ? "i16" : (sz == 2) ? "i32" : "i64";
        std::ostringstream os;
        os << "<";
        for (int i = 0; i < lanes; ++i) {
            if (i) os << ", ";
            os << et << " " << val;
        }
        os << ">";
        return os.str();
    }

    static bool try_emit_inline_sl_gen(Function& fn, std::vector<Instruction>& out,
                                       int op, int sz, int ne, int d, int szd, int offd, int keep,
                                       int n, int szn, int offn, int m, int szm, int offm, int bidx, uint64_t bimm) {
        bool is_u = (op & LaneOp::LOP_U) != 0;
        int base_op = op & ~LaneOp::LOP_U;

        // Case A: Same-width immediate vector shifts (LOP_SHLI, LOP_SHRI)
        if ((base_op == LaneOp::LOP_SHLI || base_op == LaneOp::LOP_SHRI) &&
            sz >= 0 && sz <= 3 && szd == sz && szn == sz && offd == 0 && keep == 0 && offn == 0 && m == -1 && bidx == -1) {
            int total_bytes = ne * (1 << sz);
            if (total_bytes != 8 && total_bytes != 16) return false;
            int Q = (total_bytes == 16) ? 1 : 0;
            unsigned es = 8u << sz;
            if (bimm == 0 || bimm >= es) return false;

            std::string vn128;
            out.push_back(make_vreg_read(fn, n, vn128));
            const char* vt = type_str(vec_int_type(static_cast<unsigned>(sz)));
            int full_lanes = 16 >> sz;
            std::string splat = make_splat_const(sz, full_lanes, bimm);
            const char* sop = (base_op == LaneOp::LOP_SHLI) ? "shl" : (is_u ? "lshr" : "ashr");

            std::string nv = fn.next_reg("shnv"), rv = fn.next_reg("shrv"), r128 = fn.next_reg("shrq");
            std::ostringstream os;
            os << "; [SIMDPairwiseAndConversionInlinePass] inline NEON vector immediate shift\n  "
               << nv << " = bitcast i128 " << vn128 << " to " << vt << "\n  "
               << rv << " = " << sop << " " << vt << " " << nv << ", " << splat;
            if (Q) {
                os << "\n  " << r128 << " = bitcast " << vt << " " << rv << " to i128";
            } else {
                std::string raw128 = fn.next_reg("shrw"), lo64 = fn.next_reg("shlo");
                os << "\n  " << raw128 << " = bitcast " << vt << " " << rv << " to i128\n  "
                   << lo64 << " = trunc i128 " << raw128 << " to i64\n  "
                   << r128 << " = zext i64 " << lo64 << " to i128";
            }
            Instruction calc;
            calc.tag = OpTag::RawIR;
            calc.text = os.str();
            out.push_back(std::move(calc));
            out.push_back(make_vreg_write(fn, d, r128));
            return true;
        }

        // Case B: Vector Narrowing (LOP_XTN, LOP_SHRN)
        if ((base_op == LaneOp::LOP_XTN || base_op == LaneOp::LOP_SHRN) &&
            sz >= 0 && sz <= 2 && ne == (8 >> sz) && szd == sz && szn == sz + 1 && offn == 0 &&
            (keep == 0 || keep == 1) && offd == keep * ne && m == -1 && bidx == -1) {
            std::string vn128, vd128;
            if (keep) out.push_back(make_vreg_read(fn, d, vd128));
            out.push_back(make_vreg_read(fn, n, vn128));

            const char* src_vt = (sz == 0) ? "<8 x i16>" : (sz == 1) ? "<4 x i32>" : "<2 x i64>";
            const char* dst_vt = (sz == 0) ? "<8 x i8>"  : (sz == 1) ? "<4 x i16>" : "<2 x i32>";
            std::string nv = fn.next_reg("xtnn"), tr = fn.next_reg("xtnt"), n64 = fn.next_reg("xtn64");
            std::string r128 = fn.next_reg("xtnq");
            std::ostringstream os;
            os << "; [SIMDPairwiseAndConversionInlinePass] inline NEON vector narrow (XTN/SHRN)\n  "
               << nv << " = bitcast i128 " << vn128 << " to " << src_vt << "\n  ";
            std::string to_trunc = nv;
            if (base_op == LaneOp::LOP_SHRN && bimm > 0) {
                std::string sh = fn.next_reg("xtns");
                os << sh << " = lshr " << src_vt << " " << nv << ", " << make_splat_const(sz + 1, ne, bimm) << "\n  ";
                to_trunc = sh;
            }
            os << tr << " = trunc " << src_vt << " " << to_trunc << " to " << dst_vt << "\n  "
               << n64 << " = bitcast " << dst_vt << " " << tr << " to i64\n  ";
            if (!keep) {
                os << r128 << " = zext i64 " << n64 << " to i128";
            } else {
                std::string lo64 = fn.next_reg("xtnlo"), lo128 = fn.next_reg("xtnlq");
                std::string hi128 = fn.next_reg("xtnhi"), hish = fn.next_reg("xtnhs");
                os << lo64 << " = trunc i128 " << vd128 << " to i64\n  "
                   << lo128 << " = zext i64 " << lo64 << " to i128\n  "
                   << hi128 << " = zext i64 " << n64 << " to i128\n  "
                   << hish << " = shl i128 " << hi128 << ", 64\n  "
                   << r128 << " = or i128 " << lo128 << ", " << hish;
            }
            Instruction calc;
            calc.tag = OpTag::RawIR;
            calc.text = os.str();
            out.push_back(std::move(calc));
            out.push_back(make_vreg_write(fn, d, r128));
            return true;
        }

        // Case C: Vector Widening Shifts & Arithmetic (LOP_SHLL, LOP_SHLLI, LOP_ADDL, LOP_SUBL, LOP_MULL)
        if ((base_op == LaneOp::LOP_SHLL || base_op == LaneOp::LOP_SHLLI ||
             base_op == LaneOp::LOP_ADDL || base_op == LaneOp::LOP_SUBL || base_op == LaneOp::LOP_MULL) &&
            sz >= 0 && sz <= 2 && ne == (8 >> sz) && szd == sz + 1 && offd == 0 && keep == 0 &&
            szn == sz && (offn == 0 || offn == ne) && bidx == -1) {
            bool has_m = (base_op == LaneOp::LOP_ADDL || base_op == LaneOp::LOP_SUBL || base_op == LaneOp::LOP_MULL);
            if (has_m && (m < 0 || szm != sz || offm != offn)) return false;

            std::string vn128, vm128;
            out.push_back(make_vreg_read(fn, n, vn128));
            if (has_m) out.push_back(make_vreg_read(fn, m, vm128));

            const char* half_vt = (sz == 0) ? "<8 x i8>"  : (sz == 1) ? "<4 x i16>" : "<2 x i32>";
            const char* wide_vt = (sz == 0) ? "<8 x i16>" : (sz == 1) ? "<4 x i32>" : "<2 x i64>";
            const char* ext_op  = (is_u || base_op == LaneOp::LOP_SHLL) ? "zext" : "sext";

            std::ostringstream os;
            os << "; [SIMDPairwiseAndConversionInlinePass] inline NEON widening op\n  ";
            auto extract_and_widen = [&](const std::string& v128) -> std::string {
                std::string h64 = fn.next_reg("wdh64"), hv = fn.next_reg("wdhv"), wv = fn.next_reg("wdwv");
                if (offn == 0) {
                    os << h64 << " = trunc i128 " << v128 << " to i64\n  ";
                } else {
                    std::string sh = fn.next_reg("wdsh");
                    os << sh << " = lshr i128 " << v128 << ", 64\n  "
                       << h64 << " = trunc i128 " << sh << " to i64\n  ";
                }
                os << hv << " = bitcast i64 " << h64 << " to " << half_vt << "\n  "
                   << wv << " = " << ext_op << " " << half_vt << " " << hv << " to " << wide_vt << "\n  ";
                return wv;
            };

            std::string wn = extract_and_widen(vn128);
            std::string rv = fn.next_reg("wdrv"), r128 = fn.next_reg("wdrq");
            if (base_op == LaneOp::LOP_SHLL || base_op == LaneOp::LOP_SHLLI) {
                uint64_t sh_amt = (base_op == LaneOp::LOP_SHLL) ? (8ULL << sz) : bimm;
                os << rv << " = shl " << wide_vt << " " << wn << ", " << make_splat_const(sz + 1, ne, sh_amt) << "\n  ";
            } else {
                std::string wm = extract_and_widen(vm128);
                const char* aop = (base_op == LaneOp::LOP_ADDL) ? "add" : (base_op == LaneOp::LOP_SUBL) ? "sub" : "mul";
                os << rv << " = " << aop << " " << wide_vt << " " << wn << ", " << wm << "\n  ";
            }
            os << r128 << " = bitcast " << wide_vt << " " << rv << " to i128";

            Instruction calc;
            calc.tag = OpTag::RawIR;
            calc.text = os.str();
            out.push_back(std::move(calc));
            out.push_back(make_vreg_write(fn, d, r128));
            return true;
        }

        return false;
    }
};


} // namespace

bool SIMDPairwiseAndConversionInlinePass::run(Module& M) {
    SIMDPairwiseAndConversionInlinePassImpl impl(enabled_);
    return impl.run(M);
}

} // namespace ultrs::passes

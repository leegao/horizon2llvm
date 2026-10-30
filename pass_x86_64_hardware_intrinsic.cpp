// UltraTears LLVM Pass Implementation: Pass 7 - X8664HardwareIntrinsicPass (tools/lifter/pass_x86_64_hardware_intrinsic.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class X8664HardwareIntrinsicPassImpl {
public:
    explicit X8664HardwareIntrinsicPassImpl(bool force_enable = false) : force_enable_(force_enable) {}

    static const char* static_name() { return "x86-64-hardware-intrinsic-pass"; }
    const char* pass_name() const { return "x86-64-hardware-intrinsic-pass"; }

    bool run(Module& M) {
        if (!force_enable_ && !M.is_x86_64_target()) return false;
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

                    // 1. Hardware CRC32C (`is_c == 1`) via @llvm.x86.sse42.crc32.*
                    if (inst.text.find("call i32 @rt_crc32(") != std::string::npos) {
                        char dst[64] = {}, acc[64] = {}, val[64] = {};
                        int bytes = 0, is_c = 0;
                        if (std::sscanf(inst.text.c_str(),
                                        "%63s = call i32 @rt_crc32(i32 %63[^,], i64 %63[^,], i32 %d, i32 %d)",
                                        dst, acc, val, &bytes, &is_c) == 5 &&
                            is_c == 1 && (bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8)) {
                            std::ostringstream os;
                            os << "; [X8664HardwareIntrinsicPass] hardware SSE4.2 CRC32C\n  ";
                            if (bytes == 1) {
                                std::string v8 = fn->next_reg("crc8");
                                os << v8 << " = trunc i64 " << val << " to i8\n  "
                                   << dst << " = call i32 @llvm.x86.sse42.crc32.32.8(i32 " << acc << ", i8 " << v8 << ")";
                            } else if (bytes == 2) {
                                std::string v16 = fn->next_reg("crc16");
                                os << v16 << " = trunc i64 " << val << " to i16\n  "
                                   << dst << " = call i32 @llvm.x86.sse42.crc32.32.16(i32 " << acc << ", i16 " << v16 << ")";
                            } else if (bytes == 4) {
                                std::string v32 = fn->next_reg("crc32");
                                os << v32 << " = trunc i64 " << val << " to i32\n  "
                                   << dst << " = call i32 @llvm.x86.sse42.crc32.32.32(i32 " << acc << ", i32 " << v32 << ")";
                            } else {
                                std::string acc64 = fn->next_reg("crca64"), res64 = fn->next_reg("crcr64");
                                os << acc64 << " = zext i32 " << acc << " to i64\n  "
                                   << res64 << " = call i64 @llvm.x86.sse42.crc32.64.64(i64 " << acc64 << ", i64 " << val << ")\n  "
                                   << dst << " = trunc i64 " << res64 << " to i32";
                            }
                            inst.text = os.str();
                            out.push_back(std::move(inst));
                            M.uses_x86_crc32 = true;
                            changed = true;
                            continue;
                        }
                    }

                    // 2. NEON vector CNT, CLZ, RBIT (@rt_sl_gen)
                    if (inst.text.find("call void @rt_sl_gen(") != std::string::npos) {
                        int op = 0, sz = 0, ne = 0, d = 0, szd = 0, offd = 0, keep = 0;
                        int n = 0, szn = 0, offn = 0, m = 0, szm = 0, offm = 0, bidx = 0;
                        unsigned long long bimm = 0;
                        if (std::sscanf(inst.text.c_str(),
                                        "call void @rt_sl_gen(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i64 %llu)",
                                        &op, &sz, &ne, &d, &szd, &offd, &keep, &n, &szn, &offn, &m, &szm, &offm, &bidx, &bimm) == 15 &&
                            (op == LaneOp::LOP_CNT || op == LaneOp::LOP_RBIT || op == LaneOp::LOP_CLZ) &&
                            sz <= 2 && szd == sz && szn == sz && offd == 0 && keep == 0 && offn == 0) {
                            int total_bytes = ne * (1 << sz);
                            if (total_bytes == 8 || total_bytes == 16) {
                                emit_vec_bitop(*fn, out, op, sz, total_bytes == 16, d, n);
                                changed = true;
                                continue;
                            }
                        }
                    }

                    // 3. NEON across-vector reductions (@rt_sl_across: ADDV, SADDLV, UADDLV, SMAXV, UMAXV, SMINV, UMINV)
                    int a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0;
                    if (std::sscanf(inst.text.c_str(),
                                    "call void @rt_sl_across(ptr %%c, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d, i32 %d)",
                                    &a0, &a1, &a2, &a3, &a4, &a5) == 6 &&
                        a0 >= 0 && a0 <= 3 && a3 >= 0 && a3 <= 2 && !(a3 == 2 && a2 == 0)) {
                        emit_across_reduction(*fn, out, a0, a1, a2, a3, a4, a5);
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
        out_reg = fn.next_reg("xvr");
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
        std::string p = fn.next_reg("xvw");
        Instruction inst;
        inst.tag = OpTag::CpuRegWrite;
        inst.arch_reg = 64 + vreg;
        inst.op0 = Value::reg(TypeKind::Int128, val_i128);
        inst.text = p + " = getelementptr inbounds i8, ptr %c, i64 " + std::to_string(off) + "\n  " +
                    "store i128 " + val_i128 + ", ptr " + p + ", align 16, !alias.scope !3, !noalias !4";
        return inst;
    }

    static void emit_vec_bitop(Function& fn, std::vector<Instruction>& out, int op, int sz, bool Q, int d, int n) {
        std::string vn128;
        out.push_back(make_vreg_read(fn, n, vn128));
        const char* vt = (sz == 0) ? "<16 x i8>" : (sz == 1) ? "<8 x i16>" : "<4 x i32>";
        const char* sfx = (sz == 0) ? "v16i8" : (sz == 1) ? "v8i16" : "v4i32";
        std::string nv = fn.next_reg("xbn"), rv = fn.next_reg("xbr"), r128 = fn.next_reg("xbq");
        std::ostringstream os;
        os << "; [X8664HardwareIntrinsicPass] inline vector CNT/CLZ/RBIT\n  "
           << nv << " = bitcast i128 " << vn128 << " to " << vt << "\n  ";
        if (op == LaneOp::LOP_CNT) {
            os << rv << " = call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> " << nv << ")";
        } else if (op == LaneOp::LOP_RBIT) {
            os << rv << " = call <16 x i8> @llvm.bitreverse.v16i8(<16 x i8> " << nv << ")";
        } else {
            os << rv << " = call " << vt << " @llvm.ctlz." << sfx << "(" << vt << " " << nv << ", i1 false)";
        }
        if (Q) {
            os << "\n  " << r128 << " = bitcast " << vt << " " << rv << " to i128";
        } else {
            std::string raw128 = fn.next_reg("xbrw"), lo64 = fn.next_reg("xblo");
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

    static void emit_across_reduction(Function& fn, std::vector<Instruction>& out,
                                      int op, int U, int Q, int sz, int d, int n) {
        std::string vn128;
        out.push_back(make_vreg_read(fn, n, vn128));
        std::string r128 = fn.next_reg("xrq");
        std::ostringstream os;
        os << "; [X8664HardwareIntrinsicPass] inline across-vector reduction\n  ";

        const char* src_vt = Q ? ((sz == 0) ? "<16 x i8>" : (sz == 1) ? "<8 x i16>" : "<4 x i32>")
                               : ((sz == 0) ? "<8 x i8>"  : "<4 x i16>");
        const char* src_sfx = Q ? ((sz == 0) ? "v16i8" : (sz == 1) ? "v8i16" : "v4i32")
                                : ((sz == 0) ? "v8i8"  : "v4i16");
        const char* et = (sz == 0) ? "i8" : (sz == 1) ? "i16" : "i32";

        std::string nv = fn.next_reg("xrn");
        if (Q) {
            os << nv << " = bitcast i128 " << vn128 << " to " << src_vt << "\n  ";
        } else {
            std::string lo64 = fn.next_reg("xrlo");
            os << lo64 << " = trunc i128 " << vn128 << " to i64\n  "
               << nv << " = bitcast i64 " << lo64 << " to " << src_vt << "\n  ";
        }

        if (op == 1) { // SADDLV / UADDLV (widening sum across lanes into sz + 1)
            const char* wide_vt = Q ? ((sz == 0) ? "<16 x i16>" : (sz == 1) ? "<8 x i32>" : "<4 x i64>")
                                    : ((sz == 0) ? "<8 x i16>"  : "<4 x i32>");
            const char* wide_sfx = Q ? ((sz == 0) ? "v16i16" : (sz == 1) ? "v8i32" : "v4i64")
                                     : ((sz == 0) ? "v8i16"  : "v4i32");
            const char* wet = (sz == 0) ? "i16" : (sz == 1) ? "i32" : "i64";
            std::string wv = fn.next_reg("xrwv"), red = fn.next_reg("xrred");
            os << wv << " = " << (U ? "zext " : "sext ") << src_vt << " " << nv << " to " << wide_vt << "\n  "
               << red << " = call " << wet << " @llvm.vector.reduce.add." << wide_sfx << "(" << wide_vt << " " << wv << ")\n  "
               << r128 << " = zext " << wet << " " << red << " to i128";
        } else { // ADDV (op==0), SMAXV/UMAXV (op==2), SMINV/UMINV (op==3)
            const char* rkind = (op == 0) ? "add"
                              : (op == 2) ? (U ? "umax" : "smax")
                                          : (U ? "umin" : "smin");
            std::string red = fn.next_reg("xrred");
            os << red << " = call " << et << " @llvm.vector.reduce." << rkind << "." << src_sfx
               << "(" << src_vt << " " << nv << ")\n  "
               << r128 << " = zext " << et << " " << red << " to i128";
        }

        Instruction calc;
        calc.tag = OpTag::RawIR;
        calc.text = os.str();
        out.push_back(std::move(calc));
        out.push_back(make_vreg_write(fn, d, r128));
    }
};


} // namespace

bool X8664HardwareIntrinsicPass::run(Module& M) {
    X8664HardwareIntrinsicPassImpl impl(force_enable_);
    return impl.run(M);
}

} // namespace ultrs::passes

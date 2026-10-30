// UltraTears LLVM Pass Implementation: Pass 15 - FCMPPredicateFusionPass (tools/lifter/pass_fcmp_predicate_fusion.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class FCMPPredicateFusionPassImpl {
public:
    explicit FCMPPredicateFusionPassImpl(bool enabled = true) : enabled_(enabled) {}

    static const char* static_name() { return "fcmp-predicate-fusion-pass"; }
    const char* pass_name() const { return "fcmp-predicate-fusion-pass"; }

    bool run(Module& M) {
        if (!enabled_) return false;
        bool changed = false;
        for (auto& fn : M.functions) {
            if (!fn->hle_symbol.empty()) continue;

            for (auto& bb : fn->blocks) {
                std::vector<Instruction> out;
                out.reserve(bb->insts.size() * 2);

                for (size_t i = 0; i < bb->insts.size(); ++i) {
                    Instruction& inst = bb->insts[i];
                    if (inst.dead || inst.tag != OpTag::RawIR) {
                        out.push_back(std::move(inst));
                        continue;
                    }

                    char dst[64] = {};
                    int n = 0, m = 0, zero = 0;
                    bool is_s = (std::sscanf(inst.text.c_str(),
                                             "%63s = call i64 @rt_fp_cmp_s(ptr %%c, i32 %d, i32 %d, i32 %d)",
                                             dst, &n, &m, &zero) == 4);
                    bool is_d = (!is_s && std::sscanf(inst.text.c_str(),
                                                      "%63s = call i64 @rt_fp_cmp_d(ptr %%c, i32 %d, i32 %d, i32 %d)",
                                                      dst, &n, &m, &zero) == 4);
                    if (!is_s && !is_d) {
                        out.push_back(std::move(inst));
                        continue;
                    }

                    std::string vn128, vm128;
                    out.push_back(make_vreg_read(*fn, n, vn128));
                    if (!zero) {
                        out.push_back(make_vreg_read(*fn, m, vm128));
                    }

                    const char* it = is_d ? "i64" : "i32";
                    const char* ft = is_d ? "double" : "float";
                    TypeKind ftk = is_d ? TypeKind::Double : TypeKind::Float;
                    std::string ni = fn->next_reg("fcni"), nf = fn->next_reg("fcnf");
                    std::string mf = "0.000000e+00";

                    std::ostringstream extract_os;
                    extract_os << "; [FCMPPredicateFusionPass] devirtualized @rt_fp_cmp_" << (is_d ? "d" : "s") << "\n  "
                               << ni << " = trunc i128 " << vn128 << " to " << it << "\n  "
                               << nf << " = bitcast " << it << " " << ni << " to " << ft;
                    if (!zero) {
                        std::string mi = fn->next_reg("fcmi");
                        mf = fn->next_reg("fcmf");
                        extract_os << "\n  " << mi << " = trunc i128 " << vm128 << " to " << it
                                   << "\n  " << mf << " = bitcast " << it << " " << mi << " to " << ft;
                    }

                    std::string feq = fn->next_reg("fceq"), flt = fn->next_reg("fclt"), fgt = fn->next_reg("fcgt");
                    std::string sgt = fn->next_reg("fcsgt"), slt = fn->next_reg("fcslt");
                    std::ostringstream nzcv_os;
                    nzcv_os << feq << " = fcmp oeq " << ft << " " << nf << ", " << mf << "\n  "
                            << flt << " = fcmp olt " << ft << " " << nf << ", " << mf << "\n  "
                            << fgt << " = fcmp ogt " << ft << " " << nf << ", " << mf << "\n  "
                            << sgt << " = select i1 " << fgt << ", i64 536870912, i64 805306368\n  "
                            << slt << " = select i1 " << flt << ", i64 2147483648, i64 " << sgt << "\n  "
                            << dst << " = select i1 " << feq << ", i64 1610612736, i64 " << slt;

                    if (i + 1 < bb->insts.size() && !bb->insts[i + 1].dead &&
                        bb->insts[i + 1].tag == OpTag::FlagWrite && bb->insts[i + 1].op0.repr == dst) {
                        Instruction ext_inst;
                        ext_inst.tag = OpTag::RawIR;
                        ext_inst.text = extract_os.str();
                        out.push_back(std::move(ext_inst));

                        Instruction& fw = bb->insts[i + 1];
                        fw.cmp_kind = is_d ? 5 : 4;
                        fw.cmp_sf = is_d ? 1 : 0;
                        fw.op1 = Value::reg(ftk, nf);
                        fw.op2 = Value::reg(ftk, mf);
                        fw.prefix_ir = nzcv_os.str();
                        fw.text = fw.prefix_ir + "\n  " + fw.text;
                        out.push_back(std::move(fw));
                        ++i; // consumed i+1 FlagWrite
                    } else {
                        Instruction full_inst;
                        full_inst.tag = OpTag::RawIR;
                        full_inst.dst_reg = dst;
                        full_inst.dst_type = TypeKind::Int64;
                        full_inst.text = extract_os.str() + "\n  " + nzcv_os.str();
                        out.push_back(std::move(full_inst));
                    }
                    changed = true;
                }
                bb->insts = std::move(out);
            }
        }
        return changed;
    }

private:
    bool enabled_;

    static Instruction make_vreg_read(Function& fn, int vreg, std::string& out_reg) {
        out_reg = fn.next_reg("fcvr");
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
};


} // namespace

bool FCMPPredicateFusionPass::run(Module& M) {
    FCMPPredicateFusionPassImpl impl(enabled_);
    return impl.run(M);
}

} // namespace ultrs::passes

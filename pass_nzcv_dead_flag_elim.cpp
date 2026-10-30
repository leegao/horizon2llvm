// UltraTears LLVM Pass Implementation: Pass 17 - NZCVDeadFlagElimPass (tools/lifter/pass_nzcv_dead_flag_elim.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class NZCVDeadFlagElimPassImpl {
public:
    explicit NZCVDeadFlagElimPassImpl(bool enabled = true) : enabled_(enabled) {}

    static const char* static_name() { return "nzcv-dead-flag-elim-pass"; }
    const char* pass_name() const { return "nzcv-dead-flag-elim-pass"; }

    bool run(Module& M) {
        if (!enabled_) return false;
        bool changed = false;
        for (auto& fn : M.functions) {
            for (auto& bb : fn->blocks) {
                bool nzcv_overwritten = false;
                for (auto it = bb->insts.rbegin(); it != bb->insts.rend(); ++it) {
                    if (it->dead) continue;
                    if (it->tag == OpTag::FlagWrite) {
                        if (nzcv_overwritten) {
                            it->dead = true;
                            changed = true;
                        } else {
                            nzcv_overwritten = true;
                        }
                    } else if (it->tag == OpTag::CpuRegRead && it->arch_reg == 32) {
                        nzcv_overwritten = false;
                    } else if (it->tag == OpTag::DirectCall || it->tag == OpTag::IndirectCall ||
                               it->tag == OpTag::HorizonSvc) {
                        nzcv_overwritten = false;
                    } else if (it->text.find("@rt_") != std::string::npos) {
                        nzcv_overwritten = false;
                    }
                }
            }
        }
        return changed;
    }

private:
    bool enabled_;
};


} // namespace

bool NZCVDeadFlagElimPass::run(Module& M) {
    NZCVDeadFlagElimPassImpl impl(enabled_);
    return impl.run(M);
}

} // namespace ultrs::passes

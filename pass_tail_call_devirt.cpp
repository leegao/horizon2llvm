// UltraTears LLVM Pass Implementation: Pass 19 - TailCallDevirtualizationPass (tools/lifter/pass_tail_call_devirt.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class TailCallDevirtualizationPassImpl {
public:
    explicit TailCallDevirtualizationPassImpl(bool enabled = true) : enabled_(enabled) {}

    static const char* static_name() { return "tail-call-devirt-pass"; }
    const char* pass_name() const { return "tail-call-devirt-pass"; }

    bool run(Module& M) {
        if (!enabled_) return false;
        bool changed = false;
        for (auto& fn : M.functions) {
            if (!fn->hle_symbol.empty() || fn->is_inline_clone) continue;
            for (auto& bb : fn->blocks) {
                int ret_idx = -1;
                int call_idx = -1;
                for (int i = static_cast<int>(bb->insts.size()) - 1; i >= 0; --i) {
                    const Instruction& inst = bb->insts[static_cast<size_t>(i)];
                    if (inst.dead || inst.tag == OpTag::Comment || inst.tag == OpTag::GuestInsnMarker) continue;
                    if (ret_idx < 0) {
                        if (inst.tag == OpTag::Return && inst.text == "ret void") {
                            ret_idx = i;
                            continue;
                        }
                        break;
                    } else {
                        if (inst.tag == OpTag::DirectCall && inst.branch_target != fn->entry_pc &&
                            inst.text.find("call void @f_") != std::string::npos) {
                            call_idx = i;
                        }
                        break;
                    }
                }

                if (ret_idx >= 0 && call_idx >= 0) {
                    // Check that the immediately preceding active instruction is NOT a live X30 write
                    bool has_live_x30_write = false;
                    for (int k = call_idx - 1; k >= 0; --k) {
                        const Instruction& prev = bb->insts[static_cast<size_t>(k)];
                        if (prev.dead || prev.tag == OpTag::Comment || prev.tag == OpTag::GuestInsnMarker) continue;
                        if (prev.tag == OpTag::CpuRegWrite && prev.arch_reg == 30) {
                            has_live_x30_write = true;
                        }
                        break;
                    }

                    if (!has_live_x30_write) {
                        Instruction& call_inst = bb->insts[static_cast<size_t>(call_idx)];
                        std::string callee = fn_symbol(call_inst.branch_target);
                        call_inst.tag = OpTag::Return;
                        call_inst.text = "; [TailCallDevirtualizationPass] devirtualized BL + RET -> musttail call\n  "
                                         "musttail call void @" + callee + "(ptr %c)\n  ret void";
                        bb->insts[static_cast<size_t>(ret_idx)].dead = true;
                        changed = true;
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

bool TailCallDevirtualizationPass::run(Module& M) {
    TailCallDevirtualizationPassImpl impl(enabled_);
    return impl.run(M);
}

} // namespace ultrs::passes

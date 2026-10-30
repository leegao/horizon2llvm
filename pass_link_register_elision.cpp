// UltraTears LLVM Pass Implementation: Pass 18 - LinkRegisterElisionPass (tools/lifter/pass_link_register_elision.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class LinkRegisterElisionPassImpl {
public:
    explicit LinkRegisterElisionPassImpl(bool enabled = true) : enabled_(enabled) {}

    static const char* static_name() { return "link-register-elision-pass"; }
    const char* pass_name() const { return "link-register-elision-pass"; }

    bool run(Module& M) {
        if (!enabled_) return false;
        bool changed = false;

        // Determine which module-local functions never read X30 (arch_reg == 30)
        std::unordered_set<uint64_t> x30_free_callees;
        for (const auto& fn : M.functions) {
            if (!fn->hle_symbol.empty()) continue;
            bool reads_x30_or_calls_unknown = false;
            for (const auto& bb : fn->blocks) {
                for (const auto& inst : bb->insts) {
                    if (inst.dead) continue;
                    if (inst.tag == OpTag::CpuRegRead && inst.arch_reg == 30) {
                        reads_x30_or_calls_unknown = true;
                        break;
                    }
                    if (inst.tag == OpTag::IndirectCall || inst.tag == OpTag::DirectCall ||
                        inst.tag == OpTag::HorizonSvc ||
                        inst.text.find("musttail call") != std::string::npos) {
                        reads_x30_or_calls_unknown = true;
                        break;
                    }
                }
                if (reads_x30_or_calls_unknown) break;
            }
            if (!reads_x30_or_calls_unknown) {
                x30_free_callees.insert(fn->entry_pc);
            }
        }

        if (x30_free_callees.empty()) return false;

        for (auto& fn : M.functions) {
            for (auto& bb : fn->blocks) {
                for (size_t i = 1; i < bb->insts.size(); ++i) {
                    Instruction& call_inst = bb->insts[i];
                    if (call_inst.dead || call_inst.tag != OpTag::DirectCall) continue;
                    if (!x30_free_callees.count(call_inst.branch_target)) continue;

                    // Look back 1 instruction for the `X30 = pc + 4` CpuRegWrite(30)
                    Instruction& prev = bb->insts[i - 1];
                    if (!prev.dead && prev.tag == OpTag::CpuRegWrite && prev.arch_reg == 30) {
                        prev.dead = true;
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

bool LinkRegisterElisionPass::run(Module& M) {
    LinkRegisterElisionPassImpl impl(enabled_);
    return impl.run(M);
}

} // namespace ultrs::passes

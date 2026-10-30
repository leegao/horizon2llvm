// UltraTears LLVM Pass Implementation: Pass 2 - ProfileGuidedEntryHookElimPass (tools/lifter/pass_profile_entry_hook_elim.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class ProfileGuidedEntryHookElimPassImpl {
public:
    explicit ProfileGuidedEntryHookElimPassImpl(size_t max_leaf_insns = 16, bool force_fnprof = false)
        : max_leaf_insns_(max_leaf_insns), force_fnprof_(force_fnprof) {}

    static const char* static_name() { return "profile-entry-hook-elim-pass"; }
    const char* pass_name() const { return "profile-entry-hook-elim-pass"; }

    bool run(Module& M) {
        if (force_fnprof_) return false;
        bool changed = false;
        for (auto& fn : M.functions) {
            if (!fn->hle_symbol.empty() || fn->blocks.empty()) continue;
            size_t insn_count = (fn->end_pc > fn->entry_pc) ? ((fn->end_pc - fn->entry_pc) / 4) : fn->raw_words.size();
            if (insn_count == 0 || insn_count > max_leaf_insns_) continue;

            bool has_call_or_svc = false;
            for (const auto& bb : fn->blocks) {
                if (bb->name == "prof_hit") continue;
                for (const auto& inst : bb->insts) {
                    if (inst.dead) continue;
                    if (inst.tag == OpTag::DirectCall || inst.tag == OpTag::IndirectCall ||
                        inst.tag == OpTag::HorizonSvc) {
                        has_call_or_svc = true;
                        break;
                    }
                }
                if (has_call_or_svc) break;
            }
            if (has_call_or_svc) continue;

            BasicBlock* entry_bb = fn->get_block("entry");
            BasicBlock* prof_hit = fn->get_block("prof_hit");
            BasicBlock* prof_cont = fn->get_block("prof_cont");
            if (!entry_bb || !prof_hit || !prof_cont) continue;

            std::string first_label = block_label(fn->entry_pc);
            entry_bb->insts.clear();
            Instruction br;
            br.tag = OpTag::UncondBranch;
            br.target_label = first_label;
            br.branch_target = fn->entry_pc;
            br.text = "; [ProfileGuidedEntryHookElimPass] elided @g_fnprof hook on leaf function\n  br label %" + first_label;
            entry_bb->insts.push_back(std::move(br));

            fn->block_map.erase("prof_hit");
            fn->block_map.erase("prof_cont");
            fn->blocks.erase(
                std::remove_if(fn->blocks.begin(), fn->blocks.end(),
                               [](const std::unique_ptr<BasicBlock>& b) {
                                   return b->name == "prof_hit" || b->name == "prof_cont";
                               }),
                fn->blocks.end());
            changed = true;
        }
        return changed;
    }

private:
    size_t max_leaf_insns_;
    bool force_fnprof_;
};


} // namespace

bool ProfileGuidedEntryHookElimPass::run(Module& M) {
    ProfileGuidedEntryHookElimPassImpl impl(max_leaf_insns_, force_fnprof_);
    return impl.run(M);
}

} // namespace ultrs::passes

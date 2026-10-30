// UltraTears LLVM Pass Implementation: Pass 22 - CrossChunkLeafInlinePass (tools/lifter/pass_cross_chunk_leaf_inline.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class CrossChunkLeafInlinePassImpl {
public:
    explicit CrossChunkLeafInlinePassImpl(const std::unordered_map<uint64_t, LeafFunctionSlice>* global_leaves = nullptr,
                                      uint64_t text_lo = 0,
                                      uint64_t text_hi = 0,
                                      size_t max_leaf_words = 12,
                                      bool enabled = true)
        : global_leaves_(global_leaves),
          text_lo_(text_lo),
          text_hi_(text_hi),
          max_leaf_words_(max_leaf_words),
          enabled_(enabled) {}

    static const char* static_name() { return "cross-chunk-leaf-inline-pass"; }
    const char* pass_name() const { return "cross-chunk-leaf-inline-pass"; }

    static std::unordered_map<uint64_t, LeafFunctionSlice> collect_nso_leaves(
        const std::vector<uint64_t>& order,
        uint64_t text_lo,
        uint64_t text_hi,
        const std::vector<uint32_t>& words,
        const std::unordered_map<uint64_t, std::string>& hle_map,
        size_t max_leaf_words = 12) {
        std::unordered_map<uint64_t, LeafFunctionSlice> leaves;
        for (size_t i = 0; i < order.size(); ++i) {
            uint64_t a = order[i];
            uint64_t end = (i + 1 < order.size()) ? order[i + 1] : text_hi;
            size_t nwords = (end - a) / 4;
            if (nwords == 0 || nwords > max_leaf_words || hle_map.count(a)) continue;

            bool pure_leaf = true;
            for (size_t k = 0; k < nwords; ++k) {
                uint32_t w = words[(a - text_lo) / 4 + k];
                if (k + 1 == nwords) {
                    if (w != 0xD65F03C0u) pure_leaf = false;
                } else {
                    uint32_t op0 = (w >> 25) & 0xFu;
                    if ((op0 & 0b1110u) == 0b1010u) {
                        pure_leaf = false;
                        break;
                    }
                }
            }
            if (pure_leaf) {
                LeafFunctionSlice slice;
                slice.entry_pc = a;
                slice.end_pc = end;
                slice.words.assign(words.begin() + (a - text_lo) / 4,
                                   words.begin() + (end - text_lo) / 4);
                leaves.emplace(a, std::move(slice));
            }
        }
        return leaves;
    }

    bool run(Module& M) {
        if (!enabled_) return false;
        bool changed = false;
        std::unordered_set<uint64_t> needed_external_leaves;
        std::unordered_set<uint64_t> local_leaf_pcs;

        // 1. Identify pure leaf functions already inside M
        for (const auto& fn : M.functions) {
            if (!fn->hle_symbol.empty() || fn->is_inline_clone) continue;
            size_t nwords = (fn->end_pc > fn->entry_pc) ? ((fn->end_pc - fn->entry_pc) / 4) : fn->raw_words.size();
            if (nwords == 0 || nwords > max_leaf_words_) continue;
            bool has_call = false;
            for (const auto& bb : fn->blocks) {
                for (const auto& inst : bb->insts) {
                    if (inst.dead) continue;
                    if (inst.tag == OpTag::DirectCall || inst.tag == OpTag::IndirectCall ||
                        inst.tag == OpTag::HorizonSvc) {
                        has_call = true;
                        break;
                    }
                }
            }
            if (!has_call) local_leaf_pcs.insert(fn->entry_pc);
        }

        // 2. Rewrite DirectCalls to local or global leaf functions to call `@f_<pc>_inline`
        std::unordered_set<uint64_t> local_clones_needed;
        for (auto& fn : M.functions) {
            if (fn->is_inline_clone) continue;
            for (auto& bb : fn->blocks) {
                for (auto& inst : bb->insts) {
                    if (inst.dead || inst.tag != OpTag::DirectCall) continue;
                    uint64_t target = inst.branch_target;
                    if (target == fn->entry_pc) continue;

                    if (local_leaf_pcs.count(target)) {
                        std::string inl_name = fn_symbol(target) + "_inline";
                        inst.text = "; [CrossChunkLeafInlinePass] inlined intra-chunk leaf call\n  call void @" +
                                    inl_name + "(ptr %c)";
                        local_clones_needed.insert(target);
                        changed = true;
                    } else if (global_leaves_ && global_leaves_->count(target)) {
                        std::string inl_name = fn_symbol(target) + "_inline";
                        inst.text = "; [CrossChunkLeafInlinePass] inlined cross-chunk leaf call\n  call void @" +
                                    inl_name + "(ptr %c)";
                        needed_external_leaves.insert(target);
                        changed = true;
                    }
                }
            }
        }

        // 3. Emit `internal alwaysinline` clones for local leaves
        for (uint64_t pc : local_clones_needed) {
            Function* orig = M.fn_map.at(fn_symbol(pc));
            std::string inl_name = orig->name + "_inline";
            if (M.fn_map.count(inl_name)) continue;
            Function* clone = M.create_function(inl_name, orig->entry_pc, orig->end_pc);
            clone->linkage = "internal";
            clone->is_inline_clone = true;
            for (const auto& bb : orig->blocks) {
                BasicBlock* cbb = clone->create_block(bb->name, bb->start_pc);
                cbb->insts = bb->insts;
            }
        }

        // 4. Lift and optimize `internal alwaysinline` definitions for cross-chunk leaves
        if (!needed_external_leaves.empty() && global_leaves_) {
            Module temp_mod(M.name + "_leaves", M.target_triple);
            uint64_t lo = text_lo_ ? text_lo_ : 0x800000000ULL;
            uint64_t hi = text_hi_ ? text_hi_ : 0x810000000ULL;
            lifter::AArch64ToLLVMLifter leaf_lifter(lo, hi);

            for (uint64_t pc : needed_external_leaves) {
                const auto& slice = global_leaves_->at(pc);
                std::string inl_name = fn_symbol(pc) + "_inline";
                if (M.fn_map.count(inl_name)) continue;
                Function* lfn = temp_mod.create_function(inl_name, slice.entry_pc, slice.end_pc);
                lfn->linkage = "internal";
                lfn->is_inline_clone = true;
                lfn->label_pcs.insert(slice.entry_pc);
                leaf_lifter.lift_function(temp_mod, lfn, slice.words, slice.entry_pc);
            }

            ProfileGuidedEntryHookElimPass(max_leaf_words_, false).run(temp_mod);
            FPCRFastFPLoweringPass(true).run(temp_mod);
            SIMDPairwiseAndConversionInlinePass(true).run(temp_mod);
            X8664HardwareIntrinsicPass().run(temp_mod);
            HorizonLoweringPass().run(temp_mod);
            GuestStackAndAlignmentPass().run(temp_mod);
            AArch64IdiomDeconstructPass().run(temp_mod);
            FCMPPredicateFusionPass().run(temp_mod);
            NZCVPredicateFusionPass().run(temp_mod);
            NZCVDeadFlagElimPass().run(temp_mod);
            CpuStatePromotionPass().run(temp_mod);

            if (temp_mod.uses_x86_crc32) M.uses_x86_crc32 = true;
            if (temp_mod.uses_x86_pause) M.uses_x86_pause = true;
            if (temp_mod.uses_aarch64_intrinsics) M.uses_aarch64_intrinsics = true;

            for (auto& lfn : temp_mod.functions) {
                for (auto& bb : lfn->blocks) {
                    for (auto& inst : bb->insts) inst.llvm_inst = nullptr;
                    bb->llvm_bb = nullptr;
                }
                lfn->ssa_values.clear();
                lfn->synthetic_ssa_vals.clear();
                lfn->llvm_fn = nullptr;
                lfn->parent_mod = &M;
                M.fn_map[lfn->name] = lfn.get();
                M.functions.push_back(std::move(lfn));
            }
        }

        return changed;
    }

private:
    const std::unordered_map<uint64_t, LeafFunctionSlice>* global_leaves_;
    uint64_t text_lo_;
    uint64_t text_hi_;
    size_t max_leaf_words_;
    bool enabled_;
};


} // namespace

std::unordered_map<uint64_t, LeafFunctionSlice> CrossChunkLeafInlinePass::collect_nso_leaves(
    const std::vector<uint64_t>& order,
    uint64_t text_lo,
    uint64_t text_hi,
    const std::vector<uint32_t>& words,
    const std::unordered_map<uint64_t, std::string>& hle_map,
    size_t max_leaf_words) {
    return CrossChunkLeafInlinePassImpl::collect_nso_leaves(order, text_lo, text_hi, words, hle_map, max_leaf_words);
}

bool CrossChunkLeafInlinePass::run(Module& M) {
    CrossChunkLeafInlinePassImpl impl(global_leaves_, text_lo_, text_hi_, max_leaf_words_, enabled_);
    return impl.run(M);
}

} // namespace ultrs::passes

// UltraTears LLVM Pass Implementation: Pass 20 - AAPCS64ABIContractPass (tools/lifter/pass_aapcs64_abi_contract.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class AAPCS64ABIContractPassImpl {
public:
    explicit AAPCS64ABIContractPassImpl(bool enabled = true, bool allow_external_scratch_elim = false)
        : enabled_(enabled), allow_external_scratch_elim_(allow_external_scratch_elim) {}

    static const char* static_name() { return "aapcs64-abi-contract-pass"; }
    const char* pass_name() const { return "aapcs64-abi-contract-pass"; }

    bool run(Module& M) {
        if (!enabled_) return false;
        bool changed = false;

        // 1. Compute direct register read/write sets per function
        for (auto& fn : M.functions) {
            fn->summary_reads.clear();
            fn->summary_writes.clear();
            fn->has_unknown_calls = !fn->hle_symbol.empty();

            for (const auto& bb : fn->blocks) {
                for (const auto& inst : bb->insts) {
                    if (inst.dead) continue;
                    if (inst.tag == OpTag::CpuRegRead && inst.arch_reg >= 0) {
                        fn->summary_reads.insert(inst.arch_reg);
                    } else if ((inst.tag == OpTag::CpuRegWrite || inst.tag == OpTag::FlagWrite) && inst.arch_reg >= 0) {
                        fn->summary_writes.insert(inst.arch_reg);
                    } else if (inst.tag == OpTag::IndirectCall || inst.tag == OpTag::HorizonSvc) {
                        fn->has_unknown_calls = true;
                    } else if (inst.tag == OpTag::DirectCall) {
                        if (inst.branch_target != 0) fn->direct_callee_pcs.insert(inst.branch_target);
                        std::string cname = fn_symbol(inst.branch_target);
                        if (!M.fn_map.count(cname)) {
                            fn->has_unknown_calls = true;
                        }
                    } else if (inst.text.find("musttail call") != std::string::npos) {
                        if (inst.branch_target != 0) {
                            fn->direct_callee_pcs.insert(inst.branch_target);
                            std::string cname = fn_symbol(inst.branch_target);
                            if (!M.fn_map.count(cname)) {
                                fn->has_unknown_calls = true;
                            }
                        } else {
                            fn->has_unknown_calls = true;
                        }
                    } else if (inst.text.find("call ") != std::string::npos &&
                               inst.text.find("(ptr %c") != std::string::npos) {
                        // Runtime SIMD/FP helpers (@rt_*) read/write V0..V31 (64..95) and NZCV (32) via `ptr %c`
                        fn->summary_reads.insert(32);
                        fn->summary_writes.insert(32);
                        for (int vr = 64; vr <= 95; ++vr) {
                            fn->summary_reads.insert(vr);
                            fn->summary_writes.insert(vr);
                        }
                    }
                }
            }
        }

        // 2. Propagate summaries across intra-module DirectCalls to fixpoint
        bool prop = true;
        int iter = 0;
        while (prop && iter++ < 1024) {
            prop = false;
            for (auto& fn : M.functions) {
                for (uint64_t callee_pc : fn->direct_callee_pcs) {
                    auto it = M.fn_map.find(fn_symbol(callee_pc));
                    if (it == M.fn_map.end()) {
                        if (!fn->has_unknown_calls) { fn->has_unknown_calls = true; prop = true; }
                        continue;
                    }
                    Function* callee = it->second;
                    if (callee->has_unknown_calls && !fn->has_unknown_calls) {
                        fn->has_unknown_calls = true;
                        prop = true;
                    }
                    for (int r : callee->summary_reads) {
                        if (fn->summary_reads.insert(r).second) prop = true;
                    }
                    for (int r : callee->summary_writes) {
                        if (fn->summary_writes.insert(r).second) prop = true;
                    }
                }
            }
        }

        // 3. Eliminate dead writes to AAPCS64 caller-saved scratch registers (X9..X15, NZCV)
        //    immediately before a DirectCall when the callee is known not to read them and the caller
        //    overwrites them after the call before reading them.
        for (auto& fn : M.functions) {
            for (auto& bb : fn->blocks) {
                for (size_t i = 0; i < bb->insts.size(); ++i) {
                    Instruction& inst = bb->insts[i];
                    if (inst.dead || inst.tag != OpTag::DirectCall) continue;

                    const Function* callee = nullptr;
                    auto cit = M.fn_map.find(fn_symbol(inst.branch_target));
                    if (cit != M.fn_map.end() && !cit->second->has_unknown_calls) {
                        callee = cit->second;
                    }

                    std::unordered_set<int> killed_after_call;
                    std::unordered_set<int> read_after_call;
                    for (size_t j = i + 1; j < bb->insts.size(); ++j) {
                        const Instruction& post = bb->insts[j];
                        if (post.dead) continue;
                        if (post.tag == OpTag::DirectCall || post.tag == OpTag::IndirectCall ||
                            post.tag == OpTag::HorizonSvc || post.text.find("(ptr %c") != std::string::npos) {
                            break;
                        }
                        if (post.tag == OpTag::CpuRegRead && post.arch_reg >= 0) {
                            if (!killed_after_call.count(post.arch_reg)) {
                                read_after_call.insert(post.arch_reg);
                            }
                        } else if ((post.tag == OpTag::CpuRegWrite || post.tag == OpTag::FlagWrite) && post.arch_reg >= 0) {
                            if (!read_after_call.count(post.arch_reg)) {
                                killed_after_call.insert(post.arch_reg);
                            }
                        }
                    }

                    std::unordered_set<int> read_before_call;
                    for (int k = static_cast<int>(i) - 1; k >= 0; --k) {
                        Instruction& pre = bb->insts[static_cast<size_t>(k)];
                        if (pre.dead) continue;
                        if (pre.tag == OpTag::DirectCall || pre.tag == OpTag::IndirectCall ||
                            pre.tag == OpTag::HorizonSvc || pre.text.find("(ptr %c") != std::string::npos) {
                            break;
                        }
                        if (pre.tag == OpTag::CpuRegRead && pre.arch_reg >= 0) {
                            read_before_call.insert(pre.arch_reg);
                        } else if ((pre.tag == OpTag::CpuRegWrite || pre.tag == OpTag::FlagWrite) && pre.arch_reg >= 0) {
                            int r = pre.arch_reg;
                            bool is_aapcs_scratch = (r >= 9 && r <= 15) || (r == 32) || (r >= 80 && r <= 95);
                            bool callee_ignores = callee ? (!callee->summary_reads.count(r))
                                                         : (allow_external_scratch_elim_ && is_aapcs_scratch);
                            if (!read_before_call.count(r) && callee_ignores && killed_after_call.count(r)) {
                                pre.dead = true;
                                changed = true;
                            } else {
                                read_before_call.insert(r);
                            }
                        }
                    }
                }
            }
        }
        return changed;
    }

private:
    bool enabled_;
    bool allow_external_scratch_elim_;
};


} // namespace

bool AAPCS64ABIContractPass::run(Module& M) {
    AAPCS64ABIContractPassImpl impl(enabled_, allow_external_scratch_elim_);
    return impl.run(M);
}

} // namespace ultrs::passes

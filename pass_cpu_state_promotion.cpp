// UltraTears LLVM Pass Implementation: Pass 21 - CpuStatePromotionPass (tools/lifter/pass_cpu_state_promotion.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class CpuStatePromotionPassImpl {
public:
    explicit CpuStatePromotionPassImpl(bool enable_entry_kill_pruning = true,
                                   bool enable_interproc_pruning = true)
        : enable_entry_kill_pruning_(enable_entry_kill_pruning),
          enable_interproc_pruning_(enable_interproc_pruning) {}

    static const char* static_name() { return "cpu-state-promotion-pass"; }
    const char* pass_name() const { return "cpu-state-promotion-pass"; }

    bool run(Module& M) {
        bool changed = false;
        for (auto& fn : M.functions) {
            if (!fn->hle_symbol.empty() || fn->blocks.empty()) continue;

            // 1. Intra-block dead store elimination before alloca promotion
            for (auto& bb : fn->blocks) {
                std::unordered_set<int> overwritten_regs;
                for (auto it = bb->insts.rbegin(); it != bb->insts.rend(); ++it) {
                    if (it->dead) continue;
                    if (it->tag == OpTag::DirectCall || it->tag == OpTag::IndirectCall ||
                        it->tag == OpTag::HorizonSvc || it->text.find("(ptr %c") != std::string::npos) {
                        overwritten_regs.clear();
                        continue;
                    }
                    if ((it->tag == OpTag::CpuRegWrite || it->tag == OpTag::FlagWrite) && it->arch_reg >= 0) {
                        if (overwritten_regs.count(it->arch_reg)) {
                            it->dead = true;
                            changed = true;
                        } else {
                            overwritten_regs.insert(it->arch_reg);
                        }
                    } else if (it->tag == OpTag::CpuRegRead && it->arch_reg >= 0) {
                        overwritten_regs.erase(it->arch_reg);
                    }
                }
            }

            // 2. Collect touched architectural registers (read vs written) and check for calls
            std::set<int> read_regs, written_regs, all_regs;
            for (const auto& bb : fn->blocks) {
                for (const auto& inst : bb->insts) {
                    if (inst.dead) continue;
                    if (inst.tag == OpTag::CpuRegRead && inst.arch_reg >= 0) {
                        read_regs.insert(inst.arch_reg);
                        all_regs.insert(inst.arch_reg);
                    } else if ((inst.tag == OpTag::CpuRegWrite || inst.tag == OpTag::FlagWrite) && inst.arch_reg >= 0) {
                        written_regs.insert(inst.arch_reg);
                        all_regs.insert(inst.arch_reg);
                    }
                }
            }
            if (all_regs.empty()) continue;

            // 2b. Upward-Exposed Entry Reload Pruning:
            // Identify registers unconditionally overwritten in the entry block before any read or call.
            // NOTE: Write-only registers that are only conditionally written in downstream blocks MUST still
            // be initialized from `Cpu* %c` on entry so untaken paths do not flush uninitialized alloca/`undef`
            // back to `Cpu* %c` at `ret void`!
            std::unordered_set<int> killed_on_entry;
            std::unordered_set<int> read_on_entry;
            auto scan_entry_kills = [&](const BasicBlock* b) {
                for (const auto& inst : b->insts) {
                    if (inst.dead) continue;
                    if (inst.tag == OpTag::DirectCall || inst.tag == OpTag::IndirectCall ||
                        inst.tag == OpTag::HorizonSvc ||
                        (inst.text.find("call ") != std::string::npos && inst.text.find("(ptr %c") != std::string::npos)) {
                        break;
                    }
                    if (inst.tag == OpTag::CpuRegRead && inst.arch_reg >= 0) {
                        if (!killed_on_entry.count(inst.arch_reg)) {
                            read_on_entry.insert(inst.arch_reg);
                        }
                    } else if ((inst.tag == OpTag::CpuRegWrite || inst.tag == OpTag::FlagWrite) && inst.arch_reg >= 0) {
                        if (!read_on_entry.count(inst.arch_reg)) {
                            killed_on_entry.insert(inst.arch_reg);
                        }
                    }
                }
            };

            BasicBlock* entry_bb = fn->blocks[0].get();
            if (enable_entry_kill_pruning_) {
                scan_entry_kills(entry_bb);
                // If entry_bb unconditionally branches to blocks[1] (`L_<entry_pc>`) and blocks[1] has no other predecessors:
                if (fn->blocks.size() >= 2 && entry_bb->name == "entry" && !fn->has_all_labels) {
                    std::string first_lbl = block_label(fn->entry_pc);
                    if (fn->blocks[1]->name == first_lbl) {
                        int pred_cnt = 0;
                        for (const auto& [br_pc, jti] : fn->jump_tables) {
                            (void)br_pc;
                            if (jti.is_all_labels) pred_cnt += 2;
                            else for (uint64_t t : jti.targets) if (t == fn->entry_pc) ++pred_cnt;
                        }
                        for (const auto& bb : fn->blocks) {
                            for (const auto& inst : bb->insts) {
                                if (inst.dead) continue;
                                if ((inst.tag == OpTag::UncondBranch || inst.tag == OpTag::CondBranch) &&
                                    (inst.target_label == first_lbl || inst.false_label == first_lbl)) {
                                    ++pred_cnt;
                                }
                            }
                        }
                        if (pred_cnt == 1) {
                            scan_entry_kills(fn->blocks[1].get());
                        }
                    }
                }
            }

            // 3. Emit entry-block `alloca`s and upward-exposed initial loads from `%c`
            std::vector<Instruction> prolog;
            PassLLVMBuilder entry_PB(*fn, entry_bb);
            for (int r : all_regs) {
                bool is_vec = (r >= 64);
                std::string loc = local_alloca_name(r);
                Instruction al;
                al.tag = OpTag::RawIR;
                al.text = entry_PB.emit_alloca(is_vec ? TypeKind::Int128 : TypeKind::Int64, is_vec ? 16 : 8, loc);
                prolog.push_back(std::move(al));
            }
            for (int r : all_regs) {
                // Skip initial load ONLY if `r` is unconditionally overwritten in the entry block before any read/call/branch
                if (enable_entry_kill_pruning_ && killed_on_entry.count(r)) continue;
                prolog.push_back(make_reload_from_cpu(*fn, r));
            }
            entry_bb->insts.insert(entry_bb->insts.begin(), prolog.begin(), prolog.end());

            // 4. Rewrite all CpuRegRead / CpuRegWrite / FlagWrite to access `%loc_r`,
            //    and insert targeted, liveness-pruned flushes/reloads around calls and exits.
            for (auto& bb : fn->blocks) {
                std::vector<Instruction> new_insts;
                new_insts.reserve(bb->insts.size() * 2);
                std::unordered_set<int> synced_in_cpu; // Registers guaranteed to match `Cpu* %c` after a call in `bb`

                // Check whether `bb` terminates in a function exit (`Return`) rather than branching to another block
                bool block_ends_in_return = false;
                for (auto it = bb->insts.rbegin(); it != bb->insts.rend(); ++it) {
                    if (it->dead || it->tag == OpTag::Comment || it->tag == OpTag::GuestInsnMarker) continue;
                    if (it->tag == OpTag::Return || it->text == "ret void" ||
                        it->text.find("musttail call") != std::string::npos) {
                        block_ends_in_return = true;
                    }
                    break;
                }

                for (size_t idx = 0; idx < bb->insts.size(); ++idx) {
                    Instruction& inst = bb->insts[idx];
                    if (inst.dead) continue;

                    if (inst.tag == OpTag::CpuRegRead && inst.arch_reg >= 0) {
                        bool is_vec = (inst.arch_reg >= 64);
                        std::string loc = local_alloca_name(inst.arch_reg);
                        inst.text = inst.dst_reg + " = load " + (is_vec ? "i128" : "i64") + ", ptr " + loc +
                                    ", align " + (is_vec ? "16" : "8");
                        new_insts.push_back(std::move(inst));
                        changed = true;
                        continue;
                    }

                    if ((inst.tag == OpTag::CpuRegWrite || inst.tag == OpTag::FlagWrite) && inst.arch_reg >= 0) {
                        synced_in_cpu.erase(inst.arch_reg);
                        bool is_vec = (inst.arch_reg >= 64);
                        std::string loc = local_alloca_name(inst.arch_reg);
                        std::string store_stmt = "store " + std::string(is_vec ? "i128 " : "i64 ") + inst.op0.repr +
                                                 ", ptr " + loc + ", align " + (is_vec ? "16" : "8");
                        inst.text = inst.prefix_ir.empty() ? store_stmt : (inst.prefix_ir + "\n  " + store_stmt);
                        new_insts.push_back(std::move(inst));
                        changed = true;
                        continue;
                    }

                    // Full guest call (`DirectCall`, `IndirectCall`)
                    if (inst.tag == OpTag::DirectCall || inst.tag == OpTag::IndirectCall) {
                        const Function* known_callee = nullptr;
                        if (enable_interproc_pruning_ && inst.tag == OpTag::DirectCall) {
                            auto cit = M.fn_map.find(fn_symbol(inst.branch_target));
                            if (cit != M.fn_map.end() && !cit->second->has_unknown_calls &&
                                (!cit->second->summary_reads.empty() || !cit->second->summary_writes.empty())) {
                                known_callee = cit->second;
                            }
                        }

                        std::unordered_set<int> overwritten_after;
                        std::unordered_set<int> read_after;
                        bool has_later_call_in_bb = false;
                        for (size_t j = idx + 1; j < bb->insts.size(); ++j) {
                            const Instruction& post = bb->insts[j];
                            if (post.dead) continue;
                            if (post.tag == OpTag::DirectCall || post.tag == OpTag::IndirectCall ||
                                post.tag == OpTag::HorizonSvc ||
                                (post.text.find("call ") != std::string::npos && post.text.find("(ptr %c") != std::string::npos)) {
                                has_later_call_in_bb = true;
                                break;
                            }
                            if (post.tag == OpTag::CpuRegRead && post.arch_reg >= 0) {
                                if (!overwritten_after.count(post.arch_reg)) {
                                    read_after.insert(post.arch_reg);
                                }
                            } else if ((post.tag == OpTag::CpuRegWrite || post.tag == OpTag::FlagWrite) && post.arch_reg >= 0) {
                                if (!read_after.count(post.arch_reg)) {
                                    overwritten_after.insert(post.arch_reg);
                                }
                            }
                        }

                        std::unordered_set<int> flushed_at_this_call;
                        for (int r : written_regs) {
                            if (synced_in_cpu.count(r)) {
                                flushed_at_this_call.insert(r);
                                continue;
                            }
                            if (known_callee && !known_callee->summary_reads.count(r) &&
                                !known_callee->summary_writes.count(r)) {
                                continue;
                            }
                            new_insts.push_back(make_flush_to_cpu(*fn, r));
                            flushed_at_this_call.insert(r);
                        }
                        new_insts.push_back(std::move(inst));

                        synced_in_cpu.clear();
                        bool is_final_call_before_ret = block_ends_in_return && !has_later_call_in_bb;
                        for (int r : all_regs) {
                            if (overwritten_after.count(r)) continue;
                            if (known_callee && !known_callee->summary_writes.count(r)) {
                                if (flushed_at_this_call.count(r)) synced_in_cpu.insert(r);
                                continue;
                            }
                            if (is_final_call_before_ret && !read_after.count(r) &&
                                (!written_regs.count(r) || flushed_at_this_call.count(r))) {
                                // `c` already holds the callee's updated value of `r`, and `r` is not read before `ret void`!
                                synced_in_cpu.insert(r);
                                continue;
                            }
                            new_insts.push_back(make_reload_from_cpu(*fn, r));
                            if (flushed_at_this_call.count(r) || !written_regs.count(r)) {
                                synced_in_cpu.insert(r);
                            }
                        }
                        changed = true;
                        continue;
                    }

                    // SIMD/FP runtime helper call (`call ... @rt_*(ptr %c, ...)`):
                    // Only touches V0..V31 (64..95) and NZCV (32); GPRs X0..X30 & SP stay in SSA!
                    if (inst.text.find("call ") != std::string::npos && inst.text.find("(ptr %c") != std::string::npos &&
                        inst.text.find("musttail ") == std::string::npos && inst.text.find("@rt_udf") == std::string::npos &&
                        inst.text.find("@rt_brk") == std::string::npos) {
                        for (int r : written_regs) {
                            if (r >= 32 && !synced_in_cpu.count(r)) {
                                new_insts.push_back(make_flush_to_cpu(*fn, r));
                            }
                        }
                        new_insts.push_back(std::move(inst));
                        for (int r : all_regs) {
                            if (r >= 32) {
                                new_insts.push_back(make_reload_from_cpu(*fn, r));
                                synced_in_cpu.insert(r);
                            }
                        }
                        changed = true;
                        continue;
                    }

                    // Function return or tail-call exit
                    if (inst.tag == OpTag::Return || inst.tag == OpTag::HorizonUdf || inst.tag == OpTag::HorizonBrk ||
                        inst.text.find("musttail call") != std::string::npos || inst.text == "ret void") {
                        for (int r : written_regs) {
                            if (synced_in_cpu.count(r)) continue;
                            new_insts.push_back(make_flush_to_cpu(*fn, r));
                        }
                        new_insts.push_back(std::move(inst));
                        changed = true;
                        continue;
                    }

                    new_insts.push_back(std::move(inst));
                }
                bb->insts = std::move(new_insts);
            }
        }
        return changed;
    }

private:
    bool enable_entry_kill_pruning_;
    bool enable_interproc_pruning_;

    static std::string local_alloca_name(int r) {
        if (r < 31) return "%loc_x" + std::to_string(r);
        if (r == 31) return "%loc_sp";
        if (r == 32) return "%loc_nzcv";
        return "%loc_v" + std::to_string(r - 64);
    }

    static int cpu_byte_offset(int r) {
        if (r < 31) return r * 8;
        if (r == 31) return 256;
        if (r == 32) return 272;
        return 512 + (r - 64) * 16;
    }

    static Instruction make_reload_from_cpu(Function& fn, int r) {
        bool is_vec = (r >= 64);
        TypeKind tk = is_vec ? TypeKind::Int128 : TypeKind::Int64;
        unsigned al = is_vec ? 16 : 8;
        std::string p = fn.next_reg("ldp");
        std::string v = fn.next_reg("ldv");
        std::string loc = local_alloca_name(r);
        int off = cpu_byte_offset(r);

        PassLLVMBuilder PB(fn);
        Instruction inst;
        inst.tag = OpTag::RawIR;
        inst.text = PB.emit_cpu_gep_load(off, tk, al, p, v) + "\n  " +
                    PB.emit_local_store(tk, v, loc, al);
        return inst;
    }

    static Instruction make_flush_to_cpu(Function& fn, int r) {
        bool is_vec = (r >= 64);
        TypeKind tk = is_vec ? TypeKind::Int128 : TypeKind::Int64;
        unsigned al = is_vec ? 16 : 8;
        std::string v = fn.next_reg("stv");
        std::string p = fn.next_reg("stp");
        std::string loc = local_alloca_name(r);
        int off = cpu_byte_offset(r);

        PassLLVMBuilder PB(fn);
        Instruction inst;
        inst.tag = OpTag::RawIR;
        inst.text = PB.emit_local_load(tk, loc, al, v) + "\n  " +
                    PB.emit_cpu_gep_store(off, tk, v, al, p);
        return inst;
    }
};


} // namespace

bool CpuStatePromotionPass::run(Module& M) {
    CpuStatePromotionPassImpl impl(enable_entry_kill_pruning_, enable_interproc_pruning_);
    return impl.run(M);
}

} // namespace ultrs::passes

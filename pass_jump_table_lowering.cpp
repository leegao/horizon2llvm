// UltraTears LLVM Pass Implementation: Pass 13 - JumpTableLoweringPass (tools/lifter/pass_jump_table_lowering.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class JumpTableLoweringPassImpl {
public:
    static const char* static_name() { return "jump-table-lowering-pass"; }
    const char* pass_name() const { return "jump-table-lowering-pass"; }

    bool run(Module& M) {
        bool changed = false;
        for (auto& fn : M.functions) {
            bool needs_dispatch_bb = fn->has_all_labels;
            size_t orig_block_count = fn->blocks.size();

            for (size_t bi = 0; bi < orig_block_count; ++bi) {
                BasicBlock* bb = fn->blocks[bi].get();
                for (size_t ii = 0; ii < bb->insts.size(); ++ii) {
                    Instruction& inst = bb->insts[ii];
                    if (inst.dead || inst.tag != OpTag::IndirectJump) continue;

                    uint64_t br_pc = inst.guest_pc;
                    std::string target_repr = inst.op0.repr;
                    auto jt_it = fn->jump_tables.find(br_pc);

                    if (jt_it != fn->jump_tables.end() && jt_it->second.is_all_labels) {
                        std::string p = fn->next_reg("pcp");
                        inst.text = p + " = getelementptr inbounds i8, ptr %c, i64 264\n  "
                                    "store i64 " + target_repr + ", ptr " + p + ", align 8, !alias.scope !3, !noalias !4\n  "
                                    "br label %L_dispatch";
                        needs_dispatch_bb = true;
                        changed = true;
                    } else if (jt_it != fn->jump_tables.end() && !jt_it->second.targets.empty()) {
                        std::string def_name = block_label(br_pc) + "_jt_def";
                        BasicBlock* def_bb = fn->create_block(def_name, br_pc);

                        PassLLVMBuilder PB(*fn, bb);
                        std::vector<std::pair<uint64_t, std::string>> cases;
                        cases.reserve(jt_it->second.targets.size());
                        for (uint64_t t : jt_it->second.targets) {
                            cases.emplace_back(t, block_label(t));
                        }
                        inst.text = PB.emit_switch(target_repr, def_bb->name, cases);

                        IRBuilder defB(fn.get(), def_bb);
                        defB.write_cpu_field_i64(264, inst.op0);
                        if (fn->has_all_labels) {
                            defB.emit_raw("br label %L_dispatch", OpTag::UncondBranch);
                            needs_dispatch_bb = true;
                        } else {
                            defB.emit_raw("musttail call void @rt_dispatch(ptr %c)\n  ret void", OpTag::Return);
                        }
                        changed = true;
                    } else {
                        std::string p = fn->next_reg("pcp");
                        inst.text = p + " = getelementptr inbounds i8, ptr %c, i64 264\n  "
                                    "store i64 " + target_repr + ", ptr " + p + ", align 8, !alias.scope !3, !noalias !4\n  "
                                    "musttail call void @rt_dispatch(ptr %c)\n  ret void";
                        inst.tag = OpTag::Return;
                        changed = true;
                    }
                }
            }

            if (needs_dispatch_bb && !fn->get_block("L_dispatch")) {
                BasicBlock* disp_bb = fn->create_block("L_dispatch", fn->end_pc);
                BasicBlock* disp_def = fn->create_block("L_dispatch_ext", fn->end_pc);

                IRBuilder B(fn.get(), disp_bb);
                Value pc_val = B.read_cpu_field_i64(264);
                PassLLVMBuilder PB(*fn, disp_bb);
                std::vector<std::pair<uint64_t, std::string>> cases;
                for (uint64_t t = fn->entry_pc; t < fn->end_pc; t += 4) {
                    cases.emplace_back(t, block_label(t));
                }
                B.emit_raw(PB.emit_switch(pc_val.repr, disp_def->name, cases), OpTag::IndirectJump);

                B.set_insert_point(disp_def);
                B.emit_raw("musttail call void @rt_dispatch(ptr %c)\n  ret void", OpTag::Return);
                changed = true;
            }
        }
        return changed;
    }
};


} // namespace

bool JumpTableLoweringPass::run(Module& M) {
    JumpTableLoweringPassImpl impl;
    return impl.run(M);
}

} // namespace ultrs::passes

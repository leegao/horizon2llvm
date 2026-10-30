// UltraTears LLVM Pass Implementation: Pass 1 - UltraTearsPatchPass (tools/lifter/pass_ultratears_patch.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class UltraTearsPatchPassImpl {
public:
    // Built-in TOTK version patch descriptors for `drs_off`:
    //   - TOTK 1.4.3: f_800584300 (+0xe0 = 0x8005843e0, word 0x360042e8 `tbz w8, #0, 0x800584c3c`)
    //   - TOTK 1.4.2: f_8005e9b00 (+0x210 = 0x8005e9d10, word 0x360041a8 `tbz w8, #0, 0x8005ea544`)
    static constexpr uint64_t DRS_PC_143     = 0x8005843e0ULL;
    static constexpr uint32_t DRS_WORD_143   = 0x360042e8u;
    static constexpr uint64_t DRS_TARGET_143 = 0x800584c3cULL;

    static constexpr uint64_t DRS_PC_142     = 0x8005e9d10ULL;
    static constexpr uint32_t DRS_WORD_142   = 0x360041a8u;
    static constexpr uint64_t DRS_TARGET_142 = 0x8005ea544ULL;

    explicit UltraTearsPatchPassImpl(std::vector<PatchRule> extra_rules = {}, bool include_default_drs = true) {
        if (include_default_drs) {
            PatchRule drs143;
            drs143.guest_pc = DRS_PC_143;
            drs143.expected_word = DRS_WORD_143;
            drs143.name = "drs_off";
            drs143.kind = PatchRule::Kind::DrsOff;
            drs143.target_pc = DRS_TARGET_143;
            default_drs_rules_.push_back(drs143);

            PatchRule drs142;
            drs142.guest_pc = DRS_PC_142;
            drs142.expected_word = DRS_WORD_142;
            drs142.name = "drs_off";
            drs142.kind = PatchRule::Kind::DrsOff;
            drs142.target_pc = DRS_TARGET_142;
            default_drs_rules_.push_back(drs142);
        }

        for (auto& r : extra_rules) {
            rules_[r.guest_pc] = std::move(r);
        }
    }

    static const char* static_name() { return "ultratears-patch-pass"; }
    const char* pass_name() const { return "ultratears-patch-pass"; }

    bool run(Module& M) {
        bool changed = false;
        for (auto& fn : M.functions) {
            for (const auto& rule : default_drs_rules_) {
                if (rule.guest_pc < fn->entry_pc || rule.guest_pc >= fn->end_pc) continue;
                auto it = fn->raw_words.find(rule.guest_pc);
                if (it != fn->raw_words.end() && it->second == rule.expected_word) {
                    if (apply_patch_in_function(*fn, rule)) {
                        changed = true;
                    }
                }
            }
            for (const auto& [pc, rule] : rules_) {
                if (pc < fn->entry_pc || pc >= fn->end_pc) continue;
                verify_expected_word(*fn, pc, rule.expected_word, rule.name);
                if (apply_patch_in_function(*fn, rule)) {
                    changed = true;
                }
            }
        }
        return changed;
    }

private:
    std::vector<PatchRule> default_drs_rules_;
    std::map<uint64_t, PatchRule> rules_;

    static void verify_expected_word(const Function& fn, uint64_t addr, uint32_t expected, const std::string& what) {
        auto it = fn.raw_words.find(addr);
        if (it == fn.raw_words.end()) {
            throw std::runtime_error("patch " + what + ": guest address not found in function " + fn.name);
        }
        uint32_t got = it->second;
        if (got != expected) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "patch %s at 0x%" PRIx64 ": expected 0x%08x, found 0x%08x (not TOTK 1.4.3?)",
                          what.c_str(), addr, expected, got);
            throw std::runtime_error(buf);
        }
    }

    static bool apply_patch_in_function(Function& fn, const PatchRule& rule) {
        for (auto& bb : fn.blocks) {
            for (size_t i = 0; i < bb->insts.size(); ++i) {
                if (bb->insts[i].tag != OpTag::GuestInsnMarker || bb->insts[i].guest_pc != rule.guest_pc) {
                    continue;
                }
                size_t end_i = i + 1;
                while (end_i < bb->insts.size() && bb->insts[end_i].tag != OpTag::GuestInsnMarker) {
                    ++end_i;
                }

                if (rule.kind == PatchRule::Kind::DrsOff) {
                    std::string fall_label = block_label(rule.guest_pc + 4) +
                                             (fn.label_pcs.count(rule.guest_pc + 4) ? "" : "_ft");
                    std::string taken_label = block_label(rule.target_pc);

                    for (size_t k = i + 1; k < end_i; ++k) {
                        bb->insts[k].dead = true;
                    }

                    std::string x8_val  = fn.next_reg("x8v");
                    std::string x8_ptr  = x8_val + ".p";
                    std::string drs_reg = fn.next_reg("drs_on");
                    std::string drs_cmp = fn.next_reg("drs_off");
                    std::string bit0    = fn.next_reg("b0");
                    std::string tbz_cmp = fn.next_reg("tbz");
                    std::string or_cond = fn.next_reg("drs_br");

                    PassLLVMBuilder PB(fn, bb.get());
                    auto& RB = PB.raw();
                    Instruction read_x8;
                    read_x8.tag = OpTag::CpuRegRead;
                    read_x8.arch_reg = 8;
                    read_x8.dst_reg = x8_val;
                    read_x8.dst_type = TypeKind::Int64;
                    read_x8.text = PB.emit_cpu_gep_load(64, TypeKind::Int64, 8, x8_ptr, x8_val);

                    char comment_buf[96];
                    std::snprintf(comment_buf, sizeof(comment_buf),
                                  "; [UltraTearsPatchPass] drs_off at 0x%" PRIx64, rule.guest_pc);

                    llvm::FunctionType* drs_fty = llvm::FunctionType::get(RB.getInt32Ty(), {}, false);
                    llvm::FunctionCallee drs_callee = PB.mod()->getOrInsertFunction("totk_drs_enabled", drs_fty);
                    auto* v_drs = RB.CreateCall(drs_callee, {}, strip_ssa_pct(drs_reg));
                    auto* v_dcmp = RB.CreateICmpEQ(v_drs, RB.getInt32(0), strip_ssa_pct(drs_cmp));
                    auto* v_b0 = RB.CreateAnd(PB.val(TypeKind::Int64, x8_val), RB.getInt64(1), strip_ssa_pct(bit0));
                    auto* v_tcmp = RB.CreateICmpEQ(v_b0, RB.getInt64(0), strip_ssa_pct(tbz_cmp));
                    auto* v_or = RB.CreateOr(v_dcmp, v_tcmp, strip_ssa_pct(or_cond));
                    auto* v_br = RB.CreateCondBr(v_or, PB.get_or_create_llvm_bb(taken_label), PB.get_or_create_llvm_bb(fall_label));

                    Instruction patch_ir;
                    patch_ir.tag = OpTag::CondBranch;
                    patch_ir.guest_pc = rule.guest_pc;
                    patch_ir.branch_target = rule.target_pc;
                    patch_ir.target_label = taken_label;
                    patch_ir.false_label = fall_label;
                    patch_ir.llvm_inst = v_br;
                    patch_ir.text = std::string(comment_buf) + "\n  " +
                                    IRBuilder::format_insts({v_drs, v_dcmp, v_b0, v_tcmp, v_or, v_br});

                    bb->insts.insert(bb->insts.begin() + static_cast<ptrdiff_t>(end_i), std::move(patch_ir));
                    bb->insts.insert(bb->insts.begin() + static_cast<ptrdiff_t>(end_i), std::move(read_x8));
                    return true;
                } else if (rule.kind == PatchRule::Kind::Nop) {
                    for (size_t k = i + 1; k < end_i; ++k) {
                        bb->insts[k].dead = true;
                    }
                    return true;
                } else if (rule.kind == PatchRule::Kind::BranchTarget) {
                    for (size_t k = i + 1; k < end_i; ++k) {
                        bb->insts[k].dead = true;
                    }
                    Instruction br;
                    br.tag = OpTag::UncondBranch;
                    br.guest_pc = rule.guest_pc;
                    br.branch_target = rule.target_pc;
                    br.target_label = block_label(rule.target_pc);
                    br.text = "br label %" + br.target_label;
                    bb->insts.insert(bb->insts.begin() + static_cast<ptrdiff_t>(end_i), std::move(br));
                    return true;
                } else if (rule.kind == PatchRule::Kind::WriteRegImm) {
                    for (size_t k = i + 1; k < end_i; ++k) {
                        bb->insts[k].dead = true;
                    }
                    std::string wp = fn.next_reg("pwp");
                    int off = (rule.reg_idx == 31) ? 256 : (rule.reg_idx * 8);
                    Instruction wr;
                    wr.tag = OpTag::CpuRegWrite;
                    wr.arch_reg = rule.reg_idx;
                    wr.op0 = Value::imm_u64(rule.imm_val);
                    wr.text = wp + " = getelementptr inbounds i8, ptr %c, i64 " + std::to_string(off) + "\n  " +
                              "store i64 " + std::to_string(rule.imm_val) + ", ptr " + wp +
                              ", align 8, !alias.scope !3, !noalias !4";
                    bb->insts.insert(bb->insts.begin() + static_cast<ptrdiff_t>(end_i), std::move(wr));
                    return true;
                }
            }
        }
        return false;
    }
};


} // namespace

bool UltraTearsPatchPass::run(Module& M) {
    UltraTearsPatchPassImpl impl(extra_rules_, include_default_drs_);
    return impl.run(M);
}

} // namespace ultrs::passes

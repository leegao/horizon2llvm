// UltraTears LLVM Pass Implementation: Pass 3 - NsoRodataConstPropAndDevirtPass (tools/lifter/pass_nso_rodata_constprop_devirt.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class NsoRodataConstPropAndDevirtPassImpl {
public:
    NsoRodataConstPropAndDevirtPassImpl(const uint8_t* image_data = nullptr,
                                    size_t image_size = 0,
                                    uint64_t image_base = 0,
                                    uint64_t rodata_lo = 0,
                                    uint64_t rodata_hi = 0,
                                    uint64_t text_lo = 0,
                                    uint64_t text_hi = 0,
                                    bool enabled = true,
                                    const std::unordered_set<uint64_t>* unrelocated_rodata_addrs = nullptr)
        : image_data_(image_data),
          image_size_(image_size),
          image_base_(image_base),
          rodata_lo_(rodata_lo),
          rodata_hi_(rodata_hi),
          text_lo_(text_lo),
          text_hi_(text_hi),
          enabled_(enabled),
          unrelocated_rodata_addrs_(unrelocated_rodata_addrs) {}

    static const char* static_name() { return "nso-rodata-constprop-devirt-pass"; }
    const char* pass_name() const { return "nso-rodata-constprop-devirt-pass"; }

    bool run(Module& M) {
        if (!enabled_) return false;
        bool changed = false;
        for (auto& fn : M.functions) {
            if (!fn->hle_symbol.empty()) continue;

            // Count incoming CFG edges per block (including JumpTable / IndirectJump edges, which have not
            // yet been lowered to `switch` in Pass 3!) so jump-table targets never falsely inherit single-pred gpr_consts.
            std::unordered_map<std::string, int> pred_count;
            std::unordered_map<std::string, std::string> single_pred;
            if (fn->has_all_labels) {
                for (const auto& bb : fn->blocks) pred_count[bb->name] += 2;
            }
            for (const auto& [br_pc, jti] : fn->jump_tables) {
                (void)br_pc;
                if (jti.is_all_labels) {
                    for (const auto& bb : fn->blocks) pred_count[bb->name] += 2;
                } else {
                    for (uint64_t t : jti.targets) {
                        pred_count[block_label(t)]++;
                    }
                }
            }
            for (const auto& bb : fn->blocks) {
                for (auto it = bb->insts.rbegin(); it != bb->insts.rend(); ++it) {
                    if (it->dead || it->tag == OpTag::Comment || it->tag == OpTag::GuestInsnMarker) continue;
                    if (it->tag == OpTag::UncondBranch && !it->target_label.empty()) {
                        pred_count[it->target_label]++;
                        single_pred[it->target_label] = bb->name;
                    } else if (it->tag == OpTag::CondBranch) {
                        if (!it->target_label.empty()) {
                            pred_count[it->target_label]++;
                            single_pred[it->target_label] = bb->name;
                        }
                        if (!it->false_label.empty()) {
                            pred_count[it->false_label]++;
                            single_pred[it->false_label] = bb->name;
                        }
                    } else if (it->tag == OpTag::IndirectJump && !fn->jump_tables.count(it->guest_pc)) {
                        // Unresolved indirect jump in the function: be conservative about cross-block inheritance
                        for (const auto& b2 : fn->blocks) pred_count[b2->name] += 2;
                    }
                    break;
                }
            }

            std::unordered_map<std::string, std::unordered_map<int, uint64_t>> out_gpr_consts;

            for (auto& bb : fn->blocks) {
                std::unordered_map<std::string, uint64_t> ssa_consts;
                std::unordered_map<int, uint64_t> gpr_consts;

                auto pit = pred_count.find(bb->name);
                if (pit != pred_count.end() && pit->second == 1) {
                    auto sit = out_gpr_consts.find(single_pred[bb->name]);
                    if (sit != out_gpr_consts.end()) {
                        gpr_consts = sit->second;
                    }
                }

                auto resolve_val = [&](const Value& v, uint64_t& out) -> bool {
                    if (v.is_const) {
                        out = v.const_u64;
                        return true;
                    }
                    auto it = ssa_consts.find(v.repr);
                    if (it != ssa_consts.end()) {
                        out = it->second;
                        return true;
                    }
                    return false;
                };

                for (size_t i = 0; i < bb->insts.size(); ++i) {
                    Instruction& inst = bb->insts[i];
                    if (inst.dead) continue;

                    if (inst.tag == OpTag::DirectCall || inst.tag == OpTag::IndirectCall ||
                        inst.tag == OpTag::HorizonSvc || inst.text.find("(ptr %c") != std::string::npos) {
                        // Try to devirtualize IndirectCall BEFORE clearing gpr_consts
                        if (inst.tag == OpTag::IndirectCall && !inst.op0.repr.empty()) {
                            uint64_t target_pc = 0;
                            if (resolve_val(inst.op0, target_pc) &&
                                target_pc >= text_lo_ && target_pc < text_hi_ && (target_pc % 4) == 0) {
                                std::string callee = fn_symbol(target_pc);
                                M.external_guest_fns.insert(callee);
                                fn->direct_callee_pcs.insert(target_pc);
                                inst.tag = OpTag::DirectCall;
                                inst.branch_target = target_pc;
                                inst.text = "; [NsoRodataConstPropAndDevirtPass] devirtualized BLR -> @" + callee +
                                            "\n  call void @" + callee + "(ptr %c)";
                                changed = true;
                            }
                        }
                        gpr_consts.clear();
                        continue;
                    }

                    if (inst.tag == OpTag::CpuRegRead && inst.arch_reg >= 0 && inst.arch_reg < 31) {
                        auto it = gpr_consts.find(inst.arch_reg);
                        if (it != gpr_consts.end()) {
                            // 32-bit Wn reads truncate the 64-bit architectural register to 32 bits
                            uint64_t cval = (inst.dst_type == TypeKind::Int32)
                                          ? static_cast<uint64_t>(static_cast<uint32_t>(it->second))
                                          : it->second;
                            ssa_consts[inst.dst_reg] = cval;
                        }
                        continue;
                    }

                    if (inst.tag == OpTag::CpuRegWrite && inst.arch_reg >= 0 && inst.arch_reg < 31) {
                        uint64_t cval = 0;
                        if (resolve_val(inst.op0, cval)) {
                            gpr_consts[inst.arch_reg] = cval;
                            inst.op0 = Value::imm_u64(cval);
                        } else {
                            gpr_consts.erase(inst.arch_reg);
                        }
                        continue;
                    }

                    if (inst.tag == OpTag::RawIR && !inst.dst_reg.empty()) {
                        uint64_t a = 0, b = 0;
                        if (inst.text.find(" = add ") != std::string::npos &&
                            resolve_val(inst.op0, a) && resolve_val(inst.op1, b)) {
                            uint64_t res = (inst.dst_type == TypeKind::Int32) ? static_cast<uint32_t>(a + b) : (a + b);
                            ssa_consts[inst.dst_reg] = res;
                        } else if (inst.text.find(" = sub ") != std::string::npos &&
                                   resolve_val(inst.op0, a) && resolve_val(inst.op1, b)) {
                            uint64_t res = (inst.dst_type == TypeKind::Int32) ? static_cast<uint32_t>(a - b) : (a - b);
                            ssa_consts[inst.dst_reg] = res;
                        } else if (inst.text.find(" = and ") != std::string::npos &&
                                   resolve_val(inst.op0, a) && resolve_val(inst.op1, b)) {
                            uint64_t res = (inst.dst_type == TypeKind::Int32) ? static_cast<uint32_t>(a & b) : (a & b);
                            ssa_consts[inst.dst_reg] = res;
                        } else if (inst.text.find(" = or ") != std::string::npos &&
                                   resolve_val(inst.op0, a) && resolve_val(inst.op1, b)) {
                            uint64_t res = (inst.dst_type == TypeKind::Int32) ? static_cast<uint32_t>(a | b) : (a | b);
                            ssa_consts[inst.dst_reg] = res;
                        } else if (inst.text.find(" = xor ") != std::string::npos &&
                                   resolve_val(inst.op0, a) && resolve_val(inst.op1, b)) {
                            uint64_t res = (inst.dst_type == TypeKind::Int32) ? static_cast<uint32_t>(a ^ b) : (a ^ b);
                            ssa_consts[inst.dst_reg] = res;
                        } else if (inst.text.find(" = shl ") != std::string::npos &&
                                   resolve_val(inst.op0, a) && resolve_val(inst.op1, b) && b < 64) {
                            uint64_t res = (inst.dst_type == TypeKind::Int32) ? static_cast<uint32_t>(a << b) : (a << b);
                            ssa_consts[inst.dst_reg] = res;
                        } else if (inst.text.find(" = lshr ") != std::string::npos &&
                                   resolve_val(inst.op0, a) && resolve_val(inst.op1, b) && b < 64) {
                            uint64_t res = (inst.dst_type == TypeKind::Int32) ? (static_cast<uint32_t>(a) >> b) : (a >> b);
                            ssa_consts[inst.dst_reg] = res;
                        } else if (inst.text.find(" = zext ") != std::string::npos && resolve_val(inst.op0, a)) {
                            ssa_consts[inst.dst_reg] = a;
                        } else if (inst.text.find(" = trunc ") != std::string::npos && resolve_val(inst.op0, a)) {
                            if (inst.dst_type == TypeKind::Int32) ssa_consts[inst.dst_reg] = static_cast<uint32_t>(a);
                            else if (inst.dst_type == TypeKind::Int16) ssa_consts[inst.dst_reg] = static_cast<uint16_t>(a);
                            else if (inst.dst_type == TypeKind::Int8) ssa_consts[inst.dst_reg] = static_cast<uint8_t>(a);
                        }
                        continue;
                    }

                    if (inst.tag == OpTag::GuestMemLoad && image_data_ != nullptr) {
                        uint64_t addr = 0;
                        size_t width = (inst.dst_type == TypeKind::Int8)  ? 1
                                     : (inst.dst_type == TypeKind::Int16) ? 2
                                     : (inst.dst_type == TypeKind::Int32) ? 4
                                     : (inst.dst_type == TypeKind::Int64) ? 8 : 0;
                        bool has_unreloc = false;
                        if (unrelocated_rodata_addrs_ && width > 0 && resolve_val(inst.op0, addr)) {
                            uint64_t aligned_addr = addr & ~7ULL;
                            if (unrelocated_rodata_addrs_->count(addr) || unrelocated_rodata_addrs_->count(aligned_addr)) {
                                has_unreloc = true;
                            }
                        }
                        if (!has_unreloc && width > 0 && resolve_val(inst.op0, addr) &&
                            addr >= rodata_lo_ && addr + width <= rodata_hi_ &&
                            addr >= image_base_ && (addr - image_base_) + width <= image_size_) {
                            uint64_t val = 0;
                            std::memcpy(&val, image_data_ + (addr - image_base_), width);
                            ssa_consts[inst.dst_reg] = val;
                            char cbuf[128];
                            std::snprintf(cbuf, sizeof(cbuf),
                                          "; [NsoRodataConstPropAndDevirtPass] folded .rodata load at 0x%" PRIx64 " -> 0x%" PRIx64,
                                          addr, val);
                            inst.tag = OpTag::RawIR;
                            inst.text = std::string(cbuf) + "\n  " + inst.dst_reg + " = add " +
                                        type_str(inst.dst_type) + " 0, " + std::to_string(val);
                            changed = true;
                        }
                        continue;
                    }

                    if (inst.tag == OpTag::IndirectJump) {
                        uint64_t target_pc = 0;
                        if (resolve_val(inst.op0, target_pc) &&
                            target_pc >= fn->entry_pc && target_pc < fn->end_pc &&
                            fn->label_pcs.count(target_pc)) {
                            inst.tag = OpTag::UncondBranch;
                            inst.branch_target = target_pc;
                            inst.target_label = block_label(target_pc);
                            inst.text = "; [NsoRodataConstPropAndDevirtPass] resolved indirect jump -> %" +
                                        inst.target_label + "\n  br label %" + inst.target_label;
                            changed = true;
                        }
                    }
                }

                out_gpr_consts[bb->name] = std::move(gpr_consts);
            }
        }
        return changed;
    }

private:
    const uint8_t* image_data_;
    size_t image_size_;
    uint64_t image_base_;
    uint64_t rodata_lo_;
    uint64_t rodata_hi_;
    uint64_t text_lo_;
    uint64_t text_hi_;
    bool enabled_;
    const std::unordered_set<uint64_t>* unrelocated_rodata_addrs_;
};


} // namespace

bool NsoRodataConstPropAndDevirtPass::run(Module& M) {
    NsoRodataConstPropAndDevirtPassImpl impl(
        image_data_, image_size_, image_base_, rodata_lo_, rodata_hi_,
        text_lo_, text_hi_, enabled_, unrelocated_rodata_addrs_);
    return impl.run(M);
}

} // namespace ultrs::passes

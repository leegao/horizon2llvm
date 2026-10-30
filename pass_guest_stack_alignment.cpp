// UltraTears LLVM Pass Implementation: Pass 11 - GuestStackAndAlignmentPass (tools/lifter/pass_guest_stack_alignment.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class GuestStackAndAlignmentPassImpl {
public:
    explicit GuestStackAndAlignmentPassImpl(bool enabled = true) : enabled_(enabled) {}

    static const char* static_name() { return "guest-stack-alignment-pass"; }
    const char* pass_name() const { return "guest-stack-alignment-pass"; }

    bool run(Module& M) {
        if (!enabled_) return false;
        bool changed = false;
        for (auto& fn : M.functions) {
            if (!fn->hle_symbol.empty()) continue;
            for (auto& bb : fn->blocks) {
                std::unordered_map<std::string, int> ssa_align;
                std::unordered_map<int, int> gpr_align;
                std::unordered_set<std::string> ssa_stack;
                std::unordered_set<int> gpr_stack;

                gpr_align[31] = 16; // AAPCS64 invariant: SP is 16-byte aligned
                gpr_stack.insert(31);

                auto get_align = [&](const Value& v) -> int {
                    if (v.is_const) {
                        if (v.const_u64 == 0) return 16;
                        uint64_t lsb = v.const_u64 & (~v.const_u64 + 1ULL);
                        return static_cast<int>(std::min<uint64_t>(16, lsb));
                    }
                    auto it = ssa_align.find(v.repr);
                    return (it != ssa_align.end()) ? it->second : 1;
                };

                for (auto& inst : bb->insts) {
                    if (inst.dead) continue;

                    if (inst.tag == OpTag::DirectCall || inst.tag == OpTag::IndirectCall ||
                        inst.tag == OpTag::HorizonSvc) {
                        gpr_align.clear();
                        gpr_stack.clear();
                        gpr_align[31] = 16;
                        gpr_stack.insert(31);
                        continue;
                    }

                    if (inst.tag == OpTag::CpuRegRead && inst.arch_reg >= 0 && inst.arch_reg <= 31) {
                        auto it = gpr_align.find(inst.arch_reg);
                        if (it != gpr_align.end()) ssa_align[inst.dst_reg] = it->second;
                        if (gpr_stack.count(inst.arch_reg)) ssa_stack.insert(inst.dst_reg);
                        continue;
                    }

                    if (inst.tag == OpTag::CpuRegWrite && inst.arch_reg >= 0 && inst.arch_reg <= 31) {
                        int a = get_align(inst.op0);
                        if (inst.arch_reg == 31) a = std::max(a, 16);
                        gpr_align[inst.arch_reg] = a;
                        if (ssa_stack.count(inst.op0.repr) || inst.arch_reg == 31) {
                            gpr_stack.insert(inst.arch_reg);
                        } else {
                            gpr_stack.erase(inst.arch_reg);
                        }
                        continue;
                    }

                    if (inst.tag == OpTag::RawIR && !inst.dst_reg.empty()) {
                        if (inst.text.find(" = add ") != std::string::npos ||
                            inst.text.find(" = sub ") != std::string::npos) {
                            int a0 = get_align(inst.op0);
                            int a1 = get_align(inst.op1);
                            ssa_align[inst.dst_reg] = std::gcd(a0, a1);
                            if (ssa_stack.count(inst.op0.repr)) {
                                ssa_stack.insert(inst.dst_reg);
                            }
                        }
                        continue;
                    }

                    if (inst.tag == OpTag::GuestMemLoad || inst.tag == OpTag::GuestMemStore) {
                        TypeKind mt = (inst.tag == OpTag::GuestMemLoad) ? inst.dst_type : inst.op1.type;
                        int natural_align = (mt == TypeKind::Int8)   ? 1
                                          : (mt == TypeKind::Int16)  ? 2
                                          : (mt == TypeKind::Int32)  ? 4
                                          : (mt == TypeKind::Int64)  ? 8
                                          : (mt == TypeKind::Int128) ? 16 : 1;
                        int proven = std::min(natural_align, get_align(inst.op0));
                        bool on_stack = ssa_stack.count(inst.op0.repr) > 0;
                        inst.is_stack_access = on_stack;

                        if (proven > 1) {
                            inst.align_bytes = proven;
                            auto pos = inst.text.find(", align 1");
                            if (pos != std::string::npos) {
                                inst.text.replace(pos, 9, ", align " + std::to_string(proven));
                                changed = true;
                            }
                        }
                        if (on_stack) {
                            auto scope_pos = inst.text.find("!alias.scope !4");
                            if (scope_pos != std::string::npos) {
                                inst.text.replace(scope_pos, 15, "!alias.scope !6");
                                changed = true;
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
};


} // namespace

bool GuestStackAndAlignmentPass::run(Module& M) {
    GuestStackAndAlignmentPassImpl impl(enabled_);
    return impl.run(M);
}

} // namespace ultrs::passes

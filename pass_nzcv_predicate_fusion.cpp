// UltraTears LLVM Pass Implementation: Pass 16 - NZCVPredicateFusionPass (tools/lifter/pass_nzcv_predicate_fusion.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class NZCVPredicateFusionPassImpl {
public:
    explicit NZCVPredicateFusionPassImpl(bool enabled = true, bool enable_interblock_dse = true)
        : enabled_(enabled), enable_interblock_dse_(enable_interblock_dse) {}

    static const char* static_name() { return "nzcv-predicate-fusion-pass"; }
    const char* pass_name() const { return "nzcv-predicate-fusion-pass"; }

    bool run(Module& M) {
        if (!enabled_) return false;
        bool changed = false;
        for (auto& fn : M.functions) {
            if (!fn->hle_symbol.empty() || fn->blocks.empty()) continue;

            // Phase 1: Intra-block FlagWrite + CondEval -> direct `icmp` / `fcmp` fusion
            for (auto& bb : fn->blocks) {
                int active_fw_idx = -1;
                std::unordered_map<std::string, size_t> nzcv_read_inst_idx;

                for (size_t i = 0; i < bb->insts.size(); ++i) {
                    Instruction& inst = bb->insts[i];
                    if (inst.dead) continue;

                    if (inst.tag == OpTag::DirectCall || inst.tag == OpTag::IndirectCall ||
                        inst.tag == OpTag::HorizonSvc || inst.text.find("@rt_") != std::string::npos) {
                        active_fw_idx = -1;
                        continue;
                    }

                    if (inst.tag == OpTag::FlagWrite) {
                        active_fw_idx = (inst.cmp_kind >= 1 && inst.cmp_kind <= 5) ? static_cast<int>(i) : -1;
                        continue;
                    }

                    if (inst.tag == OpTag::CpuRegRead && inst.arch_reg == 32) {
                        nzcv_read_inst_idx[inst.dst_reg] = i;
                        continue;
                    }

                    if (inst.cond_code >= 0 && inst.cond_code < 14 && !inst.dst_reg.empty() && active_fw_idx >= 0) {
                        const Instruction& fw = bb->insts[static_cast<size_t>(active_fw_idx)];
                        std::string fused_ir = try_fuse_condition(fw, inst.cond_code, inst.dst_reg);
                        if (!fused_ir.empty()) {
                            inst.text = std::string("; [NZCVPredicateFusionPass] fused NZCV condition -> direct ") + (fw.cmp_kind >= 4 ? "fcmp\n  " : "icmp\n  ") + fused_ir;
                            auto r_it = nzcv_read_inst_idx.find(inst.op0.repr);
                            if (r_it != nzcv_read_inst_idx.end()) {
                                bb->insts[r_it->second].dead = true;
                            }
                            changed = true;
                        }
                    }
                }
            }

            // Phase 2: Inter-block backward liveness for NZCV (`arch_reg == 32`)
            if (!enable_interblock_dse_) continue;
            std::unordered_map<std::string, bool> kills_nzcv_on_entry;
            for (const auto& bb : fn->blocks) {
                bool kills = false;
                for (const auto& inst : bb->insts) {
                    if (inst.dead) continue;
                    if (inst.tag == OpTag::CpuRegRead && inst.arch_reg == 32) {
                        kills = false;
                        break;
                    }
                    if (inst.tag == OpTag::DirectCall || inst.tag == OpTag::IndirectCall ||
                        inst.tag == OpTag::HorizonSvc || inst.text.find("@rt_") != std::string::npos) {
                        kills = false;
                        break;
                    }
                    if (inst.tag == OpTag::FlagWrite) {
                        kills = true;
                        break;
                    }
                }
                kills_nzcv_on_entry[bb->name] = kills;
            }

            for (auto& bb : fn->blocks) {
                std::vector<std::string> succs;
                for (auto it = bb->insts.rbegin(); it != bb->insts.rend(); ++it) {
                    if (it->dead || it->tag == OpTag::Comment || it->tag == OpTag::GuestInsnMarker) continue;
                    if (it->tag == OpTag::UncondBranch && !it->target_label.empty()) {
                        succs.push_back(it->target_label);
                    } else if (it->tag == OpTag::CondBranch) {
                        if (!it->target_label.empty()) succs.push_back(it->target_label);
                        if (!it->false_label.empty()) succs.push_back(it->false_label);
                    }
                    break;
                }
                if (succs.empty()) continue;

                bool all_succs_kill = true;
                for (const auto& s : succs) {
                    auto it = kills_nzcv_on_entry.find(s);
                    if (it == kills_nzcv_on_entry.end() || !it->second) {
                        all_succs_kill = false;
                        break;
                    }
                }
                if (!all_succs_kill) continue;

                for (auto it = bb->insts.rbegin(); it != bb->insts.rend(); ++it) {
                    if (it->dead) continue;
                    if (it->tag == OpTag::CpuRegRead && it->arch_reg == 32) break;
                    if (it->tag == OpTag::DirectCall || it->tag == OpTag::IndirectCall ||
                        it->tag == OpTag::HorizonSvc || it->text.find("@rt_") != std::string::npos) {
                        break;
                    }
                    if (it->tag == OpTag::FlagWrite) {
                        it->dead = true;
                        changed = true;
                        break;
                    }
                }
            }
        }
        return changed;
    }

private:
    bool enabled_;
    bool enable_interblock_dse_;

    static std::string try_fuse_condition(const Instruction& fw, int cond, const std::string& out_reg) {
        const char* ty = fw.cmp_sf ? "i64" : "i32";
        const std::string& a = fw.op1.repr;
        const std::string& b = fw.op2.repr;
        const std::string& r = fw.dst_reg;
        if (fw.cmp_kind == 1) { // SUBS / CMP (a - b)
            const char* pred = nullptr;
            switch (cond) {
                case 0:  pred = "eq";  break; // EQ
                case 1:  pred = "ne";  break; // NE
                case 2:  pred = "uge"; break; // CS / HS
                case 3:  pred = "ult"; break; // CC / LO
                case 4:  return out_reg + " = icmp slt " + ty + " " + r + ", 0"; // MI
                case 5:  return out_reg + " = icmp sge " + ty + " " + r + ", 0"; // PL
                case 8:  pred = "ugt"; break; // HI
                case 9:  pred = "ule"; break; // LS
                case 10: pred = "sge"; break; // GE
                case 11: pred = "slt"; break; // LT
                case 12: pred = "sgt"; break; // GT
                case 13: pred = "sle"; break; // LE
                default: return "";
            }
            if (pred) {
                return out_reg + " = icmp " + pred + " " + ty + " " + a + ", " + b;
            }
        } else if (fw.cmp_kind == 2) { // ADDS / CMN (r = a + b)
            switch (cond) {
                case 0: return out_reg + " = icmp eq "  + ty + " " + r + ", 0";
                case 1: return out_reg + " = icmp ne "  + ty + " " + r + ", 0";
                case 2: return out_reg + " = icmp ult " + ty + " " + r + ", " + a;
                case 3: return out_reg + " = icmp uge " + ty + " " + r + ", " + a;
                case 4: return out_reg + " = icmp slt " + ty + " " + r + ", 0";
                case 5: return out_reg + " = icmp sge " + ty + " " + r + ", 0";
                default: return "";
            }
        } else if (fw.cmp_kind == 3) { // ANDS / TST (r = a & b, C=0, V=0)
            switch (cond) {
                case 0:  return out_reg + " = icmp eq "  + ty + " " + r + ", 0";
                case 1:  return out_reg + " = icmp ne "  + ty + " " + r + ", 0";
                case 4:
                case 11: return out_reg + " = icmp slt " + ty + " " + r + ", 0";
                case 5:
                case 10: return out_reg + " = icmp sge " + ty + " " + r + ", 0";
                case 12: return out_reg + " = icmp sgt " + ty + " " + r + ", 0";
                case 13: return out_reg + " = icmp sle " + ty + " " + r + ", 0";
                default: return "";
            }
        } else if (fw.cmp_kind == 4 || fw.cmp_kind == 5) { // FCMP_S / FCMP_D
            const char* fty = (fw.cmp_kind == 5) ? "double" : "float";
            const char* fpred = nullptr;
            switch (cond) {
                case 0:  fpred = "oeq"; break; // EQ (Z==1)
                case 1:  fpred = "une"; break; // NE (Z==0)
                case 2:  fpred = "uge"; break; // CS/HS (C==1)
                case 3:  fpred = "olt"; break; // CC/LO (C==0)
                case 4:  fpred = "olt"; break; // MI (N==1)
                case 5:  fpred = "uge"; break; // PL (N==0)
                case 6:  fpred = "uno"; break; // VS (V==1)
                case 7:  fpred = "ord"; break; // VC (V==0)
                case 8:  fpred = "ugt"; break; // HI (C==1 && Z==0)
                case 9:  fpred = "ole"; break; // LS (C==0 || Z==1)
                case 10: fpred = "oge"; break; // GE (N==V)
                case 11: fpred = "ult"; break; // LT (N!=V)
                case 12: fpred = "ogt"; break; // GT (Z==0 && N==V)
                case 13: fpred = "ule"; break; // LE (Z==1 || N!=V)
                default: return "";
            }
            if (fpred) {
                return out_reg + " = fcmp " + fpred + " " + fty + " " + a + ", " + b;
            }
        }
        return "";
    }
};


} // namespace

bool NZCVPredicateFusionPass::run(Module& M) {
    NZCVPredicateFusionPassImpl impl(enabled_, enable_interblock_dse_);
    return impl.run(M);
}

} // namespace ultrs::passes

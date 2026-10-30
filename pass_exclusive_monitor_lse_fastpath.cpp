// UltraTears LLVM Pass Implementation: Pass 8 - ExclusiveMonitorLSEFastPathPass (tools/lifter/pass_exclusive_monitor_lse_fastpath.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class ExclusiveMonitorLSEFastPathPassImpl {
public:
    explicit ExclusiveMonitorLSEFastPathPassImpl(bool enabled = true) : enabled_(enabled) {}

    static const char* static_name() { return "exclusive-monitor-lse-fastpath-pass"; }
    const char* pass_name() const { return "exclusive-monitor-lse-fastpath-pass"; }

    bool run(Module& M) {
        if (!enabled_) return false;
        bool changed = false;
        for (auto& fn : M.functions) {
            if (!fn->hle_symbol.empty()) continue;
            for (auto& bb : fn->blocks) {
                int ldx_idx = -1;
                std::string ldx_val_reg;
                std::string ldx_addr_reg;
                int excl_addr_store_idx = -1;
                int excl_val_store_idx = -1;

                for (size_t i = 0; i < bb->insts.size(); ++i) {
                    Instruction& inst = bb->insts[i];
                    if (inst.dead) continue;

                    if (inst.tag == OpTag::DirectCall || inst.tag == OpTag::IndirectCall ||
                        inst.tag == OpTag::HorizonSvc || inst.tag == OpTag::HorizonBarrier ||
                        inst.tag == OpTag::GuestMemStore) {
                        ldx_idx = -1;
                        continue;
                    }

                    if (inst.tag == OpTag::RawIR && inst.text.find(" = load atomic ") != std::string::npos) {
                        auto eq_pos = inst.text.find(" = load atomic ");
                        ldx_val_reg = inst.text.substr(0, eq_pos);
                        ldx_idx = static_cast<int>(i);
                        excl_addr_store_idx = -1;
                        excl_val_store_idx = -1;
                        if (i + 3 < bb->insts.size() &&
                            bb->insts[i + 1].text.find("i64 296") != std::string::npos &&
                            bb->insts[i + 2].text.find("i64 304") != std::string::npos) {
                            excl_addr_store_idx = static_cast<int>(i + 1);
                            excl_val_store_idx = static_cast<int>(i + 2);
                        } else if (i + 4 < bb->insts.size() &&
                                   bb->insts[i + 2].text.find("i64 296") != std::string::npos &&
                                   bb->insts[i + 3].text.find("i64 304") != std::string::npos) {
                            excl_addr_store_idx = static_cast<int>(i + 2);
                            excl_val_store_idx = static_cast<int>(i + 3);
                        }
                        continue;
                    }

                    if (ldx_idx >= 0 && excl_addr_store_idx >= 0 && excl_val_store_idx >= 0 &&
                        inst.tag == OpTag::RawIR && inst.text.find(" = cmpxchg ptr ") != std::string::npos &&
                        i >= 4) {
                        int stxr_excl_start = -1;
                        for (int k = static_cast<int>(i) - 1; k >= std::max(0, static_cast<int>(i) - 8); --k) {
                            if (bb->insts[k].text.find("i64 296") != std::string::npos &&
                                bb->insts[k].text.find("load i64") != std::string::npos &&
                                k + 3 < static_cast<int>(i) &&
                                bb->insts[k + 1].text.find("i64 304") != std::string::npos &&
                                bb->insts[k + 2].text.find("i64 296") != std::string::npos &&
                                bb->insts[k + 3].text.find(" = icmp eq i64 ") != std::string::npos) {
                                stxr_excl_start = k;
                                break;
                            }
                        }
                        if (stxr_excl_start >= 0) {
                            const std::string& st_txt = bb->insts[excl_val_store_idx].text;
                            const std::string& ld_txt = bb->insts[stxr_excl_start + 1].text;
                            auto st_pos = st_txt.find("store i64 ");
                            auto ld_nl  = ld_txt.find("\n  ");
                            if (st_pos != std::string::npos && ld_nl != std::string::npos) {
                                size_t v_start = st_pos + 10;
                                size_t v_end = st_txt.find(",", v_start);
                                std::string v64_reg = st_txt.substr(v_start, v_end - v_start);
                                size_t fv_start = ld_nl + 3;
                                size_t fv_end = ld_txt.find(" = load i64", fv_start);
                                std::string fv_reg = ld_txt.substr(fv_start, fv_end - fv_start);

                                bb->insts[excl_addr_store_idx].dead = true;
                                bb->insts[excl_val_store_idx].dead = true;
                                bb->insts[stxr_excl_start].dead = true;
                                bb->insts[stxr_excl_start + 1].text = fv_reg + " = add i64 0, " + v64_reg;
                                bb->insts[stxr_excl_start + 2].dead = true;

                                std::string match_reg = bb->insts[stxr_excl_start + 3].dst_reg;
                                if (!match_reg.empty()) {
                                    bb->insts[stxr_excl_start + 3].text =
                                        "; [ExclusiveMonitorLSEFastPathPass] elided software exclusive monitor\n  " +
                                        match_reg + " = icmp eq i64 0, 0";
                                    changed = true;
                                }
                            }
                            ldx_idx = -1;
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

bool ExclusiveMonitorLSEFastPathPass::run(Module& M) {
    ExclusiveMonitorLSEFastPathPassImpl impl(enabled_);
    return impl.run(M);
}

} // namespace ultrs::passes

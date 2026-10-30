// UltraTears LLVM Pass Implementation: Pass 14 - ShadowDispatchTablePass (tools/lifter/pass_shadow_dispatch_table.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class ShadowDispatchTablePassImpl {
public:
    explicit ShadowDispatchTablePassImpl(bool enabled = true) : enabled_(enabled) {}

    static const char* static_name() { return "shadow-dispatch-table-pass"; }
    const char* pass_name() const { return "shadow-dispatch-table-pass"; }

    bool run(Module& M) {
        if (!enabled_) return false;
        bool changed = false;
        for (auto& fn : M.functions) {
            if (!fn->hle_symbol.empty()) continue;
            for (auto& bb : fn->blocks) {
                for (auto& inst : bb->insts) {
                    if (inst.dead || inst.tag != OpTag::IndirectCall) continue;
                    if (inst.op0.repr.empty() || inst.text.find("@rt_dispatch") == std::string::npos) continue;

                    std::string t_repr = inst.op0.repr;
                    std::string shr = fn->next_reg("ic_shr");
                    std::string idx = fn->next_reg("ic_idx");
                    std::string pcp = fn->next_reg("ic_pcp");
                    std::string pc  = fn->next_reg("ic_pc");
                    std::string fnp = fn->next_reg("ic_fnp");
                    std::string fptr = fn->next_reg("ic_fn");
                    std::string hit = fn->next_reg("ic_hit");
                    std::string cal = fn->next_reg("ic_cal");

                    PassLLVMBuilder PB(*fn, bb.get());
                    auto& RB = PB.raw();
                    llvm::Value* tv = PB.val(TypeKind::Int64, t_repr);
                    auto* v_shr = RB.CreateLShr(tv, RB.getInt64(2), strip_ssa_pct(shr));
                    auto* v_idx = RB.CreateAnd(v_shr, RB.getInt64(65535), strip_ssa_pct(idx));
                    auto* arr_pc_ty = llvm::ArrayType::get(RB.getInt64Ty(), 65536);
                    auto* arr_fn_ty = llvm::ArrayType::get(PB.type(TypeKind::Pointer), 65536);
                    auto* v_pcp = RB.CreateInBoundsGEP(arr_pc_ty, PB.val(TypeKind::Pointer, "@g_icache_pc"),
                                                       {RB.getInt64(0), v_idx}, strip_ssa_pct(pcp));
                    auto* v_pc  = RB.CreateAlignedLoad(RB.getInt64Ty(), v_pcp, llvm::Align(8), strip_ssa_pct(pc));
                    auto* v_fnp = RB.CreateInBoundsGEP(arr_fn_ty, PB.val(TypeKind::Pointer, "@g_icache_fn"),
                                                       {RB.getInt64(0), v_idx}, strip_ssa_pct(fnp));
                    auto* v_fptr = RB.CreateAlignedLoad(PB.type(TypeKind::Pointer), v_fnp, llvm::Align(8), strip_ssa_pct(fptr));
                    auto* v_hit = RB.CreateICmpEQ(v_pc, tv, strip_ssa_pct(hit));
                    auto* v_cal = RB.CreateSelect(v_hit, v_fptr, PB.val(TypeKind::Pointer, "@rt_dispatch"), strip_ssa_pct(cal));
                    auto* disp_fty = llvm::FunctionType::get(RB.getVoidTy(), {PB.type(TypeKind::Pointer)}, false);
                    auto* v_call = RB.CreateCall(disp_fty, v_cal, {PB.val(TypeKind::Pointer, "%c")});
                    inst.llvm_inst = v_call;
                    inst.text = "; [ShadowDispatchTablePass] inline O(1) shadow dispatch lookup\n  " +
                                IRBuilder::format_insts({v_shr, v_idx, v_pcp, v_pc, v_fnp, v_fptr, v_hit, v_cal, v_call});
                    changed = true;
                }
            }
        }
        return changed;
    }

private:
    bool enabled_;
};


} // namespace

bool ShadowDispatchTablePass::run(Module& M) {
    ShadowDispatchTablePassImpl impl(enabled_);
    return impl.run(M);
}

} // namespace ultrs::passes

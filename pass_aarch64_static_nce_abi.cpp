// UltraTears LLVM Pass Implementation: Pass 23 - AArch64StaticNCEABIPass (tools/lifter/pass_aarch64_static_nce_abi.cpp)
//
// Uses LLVM 21 C++ APIs (`llvm::IRBuilder<llvm::NoFolder>`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`) via `PassLLVMBuilder`.

#include "passes.hpp"
#include "passes_llvm_utils.hpp"

namespace ultrs::passes {

namespace {

class AArch64StaticNCEABIPassImpl {
public:
    explicit AArch64StaticNCEABIPassImpl(bool enabled = false) : enabled_(enabled) {}

    static const char* static_name() { return "aarch64-static-nce-abi-pass"; }
    const char* pass_name() const { return "aarch64-static-nce-abi-pass"; }

    bool run(Module& M) {
        if (!enabled_) return false;
        bool changed = false;
        for (const auto& fn : M.functions) {
            if (!fn->hle_symbol.empty() || fn->is_inline_clone) continue;
            PassLLVMBuilder PB(const_cast<Function&>(*fn));
            auto& RB = PB.raw();
            std::ostringstream nce;
            nce << "; [AArch64StaticNCEABIPass] Register-pinned Static-NCE entry point for @" << fn->name << "\n"
                << "define dso_local { i64, i64 } @" << fn->name << "_nce("
                << "ptr noalias noundef nonnull align 16 dereferenceable(1024) %c, "
                << "i64 %x0_in, i64 %x1_in, i64 %x2_in, i64 %x3_in, "
                << "i64 %x4_in, i64 %x5_in, i64 %x6_in, i64 %x7_in, i64 %sp_in) nounwind {\n"
                << "entry:\n";
            for (int r = 0; r < 8; ++r) {
                std::string pin = "%p_in" + std::to_string(r);
                std::string xin = "%x" + std::to_string(r) + "_in";
                nce << "  " << PB.emit_cpu_gep_store(r * 8, TypeKind::Int64, xin, 8, pin) << "\n";
            }
            nce << "  " << PB.emit_cpu_gep_store(256, TypeKind::Int64, "%sp_in", 8, "%p_sp") << "\n";
            auto* fty = llvm::FunctionType::get(RB.getVoidTy(), {PB.type(TypeKind::Pointer)}, false);
            llvm::FunctionCallee callee = PB.mod()->getOrInsertFunction(fn->name, fty);
            auto* v_call = RB.CreateCall(callee, {PB.val(TypeKind::Pointer, "%c")});
            auto* v_r0 = RB.CreateAlignedLoad(RB.getInt64Ty(), PB.val(TypeKind::Pointer, "%p_in0"), llvm::Align(8), "r0");
            auto* v_r1 = RB.CreateAlignedLoad(RB.getInt64Ty(), PB.val(TypeKind::Pointer, "%p_in1"), llvm::Align(8), "r1");
            auto* ret_struct_ty = llvm::StructType::get(PB.ctx(), {RB.getInt64Ty(), RB.getInt64Ty()});
            auto* v_ret0 = RB.CreateInsertValue(llvm::PoisonValue::get(ret_struct_ty), v_r0, {0}, "ret0");
            auto* v_ret1 = RB.CreateInsertValue(v_ret0, v_r1, {1}, "ret1");
            auto* v_ret  = RB.CreateRet(v_ret1);
            nce << "  " << format_llvm_inst(v_call) << "\n"
                << "  " << format_llvm_inst(v_r0) << ", !alias.scope !3, !noalias !4\n"
                << "  " << format_llvm_inst(v_r1) << ", !alias.scope !3, !noalias !4\n"
                << "  " << format_llvm_inst(v_ret0) << "\n"
                << "  " << format_llvm_inst(v_ret1) << "\n"
                << "  " << format_llvm_inst(v_ret) << "\n"
                << "}";
            M.extra_raw_functions.push_back(nce.str());
            changed = true;
        }
        return changed;
    }

private:
    bool enabled_;
};


} // namespace

bool AArch64StaticNCEABIPass::run(Module& M) {
    AArch64StaticNCEABIPassImpl impl(enabled_);
    return impl.run(M);
}

} // namespace ultrs::passes

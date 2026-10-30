// Native LLVM New Pass Manager Plugin (tools/lifter_llvm/llvm_pass_plugin.cpp)
//
// When compiled with full LLVM development headers (-DULTRS_HAVE_LLVM_HEADERS=1), registers the
// UltraTears passes (`ultratears-patch`, `horizon-lowering`, `nzcv-dce`, `cpu-sroa`) directly with
// `llvm::PassBuilder` so they can also be invoked via `opt -load-pass-plugin` or `clang -fpass-plugin`.

#if defined(ULTRS_HAVE_LLVM_HEADERS)
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

namespace {

struct UltraTearsNativePatchPass : public llvm::PassInfoMixin<UltraTearsNativePatchPass> {
    llvm::PreservedAnalyses run(llvm::Module& M, llvm::ModuleAnalysisManager&) {
        bool changed = false;
        // Locate @f_800584300 (or any function containing block L_8005843e0) and apply drs_off
        llvm::FunctionCallee drs_fn = M.getOrInsertFunction(
            "totk_drs_enabled", llvm::FunctionType::get(llvm::Type::getInt32Ty(M.getContext()), false));
        for (llvm::Function& F : M) {
            if (F.isDeclaration()) continue;
            llvm::BasicBlock* target_bb = nullptr;
            llvm::BasicBlock* disabled_bb = nullptr;
            for (llvm::BasicBlock& BB : F) {
                if (BB.getName() == "L_8005843e0") target_bb = &BB;
                if (BB.getName() == "L_800584c3c") disabled_bb = &BB;
            }
            if (target_bb && disabled_bb) {
                if (auto* br = llvm::dyn_cast<llvm::BranchInst>(target_bb->getTerminator())) {
                    if (br->isConditional()) {
                        llvm::IRBuilder<> B(br);
                        llvm::Value* drs_on = B.CreateCall(drs_fn);
                        llvm::Value* drs_off = B.CreateICmpEQ(drs_on, B.getInt32(0));
                        llvm::Value* new_cond = B.CreateOr(drs_off, br->getCondition());
                        br->setCondition(new_cond);
                        changed = true;
                    }
                }
            }
        }
        return changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
    }
};

struct UltraTearsNativeNoAliasPass : public llvm::PassInfoMixin<UltraTearsNativeNoAliasPass> {
    llvm::PreservedAnalyses run(llvm::Function& F, llvm::FunctionAnalysisManager&) {
        if (F.isDeclaration() || F.arg_empty()) return llvm::PreservedAnalyses::all();
        llvm::Argument* cpu_arg = F.getArg(0);
        if (!cpu_arg->getType()->isPointerTy()) return llvm::PreservedAnalyses::all();
        cpu_arg->addAttr(llvm::Attribute::NoAlias);
        cpu_arg->addAttr(llvm::Attribute::NonNull);
        cpu_arg->addAttr(llvm::Attribute::getWithAlignment(F.getContext(), llvm::Align(16)));
        cpu_arg->addAttr(llvm::Attribute::getWithDereferenceableBytes(F.getContext(), 1024));
        return llvm::PreservedAnalyses::all();
    }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "UltraTearsLLVMPasses", "v0.2.0",
        [](llvm::PassBuilder& PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::ModulePassManager& MPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                    if (Name == "ultratears-patch") {
                        MPM.addPass(UltraTearsNativePatchPass());
                        return true;
                    }
                    return false;
                });
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager& FPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                    if (Name == "ultratears-noalias") {
                        FPM.addPass(UltraTearsNativeNoAliasPass());
                        return true;
                    }
                    return false;
                });
        }};
}
#endif

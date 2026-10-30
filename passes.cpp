// UltraTears LLVM Pass Pipeline Orchestration (tools/lifter/passes.cpp)
//
// Implements `PassPipelineConfig`, `build_ultratears_pipeline`, `build_raw_lowering_pipeline`,
// and `serialize_unlowered_reference_ir`.

#include "passes.hpp"

namespace ultrs::passes {

void PassPipelineConfig::disable_high_risk_passes() {
    enable_rodata_constprop = false;
    enable_lse_fastpath = false;
    enable_nzcv_interblock_dse = false;
    enable_link_reg_elision = false;
    enable_tail_call_devirt = false;
    enable_aapcs64_abi_contract = false;
    allow_external_scratch_elim = false;
    enable_cpu_interproc_pruning = false;
    enable_cross_chunk_leaf_inline = false;
    enable_nce_abi = false;
}

void PassPipelineConfig::enable_all_passes() {
    enable_fast_fp = true;
    enable_simd_inline = true;
    enable_shadow_dispatch = true;
    enable_x86_64_intrinsics = true;
    enable_tls_spinwait = true;
    enable_stack_alignment = true;
    enable_idiom_deconstruct = true;
    enable_fcmp_fusion = true;
    enable_nzcv_fusion = true;
    enable_nzcv_dead_flag_elim = true;
    enable_cpu_entry_kill_pruning = true;
    enable_rodata_constprop = true;
    enable_lse_fastpath = true;
    enable_nzcv_interblock_dse = true;
    enable_link_reg_elision = true;
    enable_tail_call_devirt = true;
    enable_aapcs64_abi_contract = true;
    enable_cpu_interproc_pruning = true;
    enable_cross_chunk_leaf_inline = true;
}

void PassPipelineConfig::disable_all_opt_passes() {
    disable_high_risk_passes();
    enable_fast_fp = false;
    enable_simd_inline = false;
    enable_shadow_dispatch = false;
    enable_x86_64_intrinsics = false;
    enable_tls_spinwait = false;
    enable_stack_alignment = false;
    enable_idiom_deconstruct = false;
    enable_fcmp_fusion = false;
    enable_nzcv_fusion = false;
    enable_nzcv_dead_flag_elim = false;
    enable_cpu_entry_kill_pruning = false;
}

bool PassPipelineConfig::set_pass_enabled(const std::string& name, bool en) {
    if (name == "high-risk" || name == "high_risk") {
        if (!en) disable_high_risk_passes();
        else {
            enable_rodata_constprop = true;
            enable_lse_fastpath = true;
            enable_nzcv_interblock_dse = true;
            enable_link_reg_elision = true;
            enable_tail_call_devirt = true;
            enable_aapcs64_abi_contract = true;
            enable_cpu_interproc_pruning = true;
            enable_cross_chunk_leaf_inline = true;
        }
        return true;
    }
    if (name == "all-opt" || name == "all_opt" || name == "opt") {
        if (en) enable_all_passes();
        else disable_all_opt_passes();
        return true;
    }
    if (name == "rodata-constprop" || name == NsoRodataConstPropAndDevirtPass::static_name() || name == "3") {
        enable_rodata_constprop = en; return true;
    }
    if (name == "fast-fp" || name == FPCRFastFPLoweringPass::static_name() || name == "4") {
        enable_fast_fp = en; return true;
    }
    if (name == "simd-inline" || name == SIMDPairwiseAndConversionInlinePass::static_name() || name == "5") {
        enable_simd_inline = en; return true;
    }
    if (name == "aarch64-intrinsics" || name == AArch64NativeIntrinsicPass::static_name() || name == "6") {
        force_aarch64_intrinsics = en; return true;
    }
    if (name == "x86-64-intrinsics" || name == X8664HardwareIntrinsicPass::static_name() || name == "7") {
        enable_x86_64_intrinsics = en; return true;
    }
    if (name == "lse-fastpath" || name == ExclusiveMonitorLSEFastPathPass::static_name() || name == "8") {
        enable_lse_fastpath = en; return true;
    }
    if (name == "tls-spinwait" || name == HorizonTLSAndSpinWaitPass::static_name() || name == "10") {
        enable_tls_spinwait = en; return true;
    }
    if (name == "stack-alignment" || name == GuestStackAndAlignmentPass::static_name() || name == "11") {
        enable_stack_alignment = en; return true;
    }
    if (name == "idiom-deconstruct" || name == AArch64IdiomDeconstructPass::static_name() || name == "12") {
        enable_idiom_deconstruct = en; return true;
    }
    if (name == "shadow-dispatch" || name == ShadowDispatchTablePass::static_name() || name == "14") {
        enable_shadow_dispatch = en; return true;
    }
    if (name == "fcmp-fusion" || name == FCMPPredicateFusionPass::static_name() || name == "15") {
        enable_fcmp_fusion = en; return true;
    }
    if (name == "nzcv-fusion" || name == NZCVPredicateFusionPass::static_name() || name == "16") {
        enable_nzcv_fusion = en; return true;
    }
    if (name == "nzcv-interblock-dse") {
        enable_nzcv_interblock_dse = en; return true;
    }
    if (name == "nzcv-dead-flag-elim" || name == NZCVDeadFlagElimPass::static_name() || name == "17") {
        enable_nzcv_dead_flag_elim = en; return true;
    }
    if (name == "link-reg-elision" || name == LinkRegisterElisionPass::static_name() || name == "18") {
        enable_link_reg_elision = en; return true;
    }
    if (name == "tail-call-devirt" || name == TailCallDevirtualizationPass::static_name() || name == "19") {
        enable_tail_call_devirt = en; return true;
    }
    if (name == "aapcs64-abi" || name == AAPCS64ABIContractPass::static_name() || name == "20") {
        enable_aapcs64_abi_contract = en; return true;
    }
    if (name == "cpu-entry-kill-pruning") {
        enable_cpu_entry_kill_pruning = en; return true;
    }
    if (name == "cpu-interproc-pruning") {
        enable_cpu_interproc_pruning = en; return true;
    }
    if (name == "cross-chunk-leaf-inline" || name == CrossChunkLeafInlinePass::static_name() || name == "22") {
        enable_cross_chunk_leaf_inline = en; return true;
    }
    if (name == "static-nce-abi" || name == AArch64StaticNCEABIPass::static_name() || name == "23") {
        enable_nce_abi = en; return true;
    }
    return false;
}

void build_ultratears_pipeline(ModulePassManager& MPM, const PassPipelineConfig& cfg) {
    MPM.addPass(std::make_unique<UltraTearsPatchPass>(cfg.extra_patches, cfg.include_default_drs));
    MPM.addPass(std::make_unique<ProfileGuidedEntryHookElimPass>(16, cfg.force_fnprof));
    MPM.addPass(std::make_unique<NsoRodataConstPropAndDevirtPass>(
        cfg.image_data, cfg.image_size, cfg.image_base,
        cfg.rodata_lo, cfg.rodata_hi, cfg.text_lo, cfg.text_hi,
        cfg.enable_rodata_constprop, cfg.unrelocated_rodata_addrs));
    MPM.addPass(std::make_unique<FPCRFastFPLoweringPass>(cfg.enable_fast_fp));
    MPM.addPass(std::make_unique<SIMDPairwiseAndConversionInlinePass>(cfg.enable_fast_fp && cfg.enable_simd_inline));
    MPM.addPass(std::make_unique<AArch64NativeIntrinsicPass>(cfg.force_aarch64_intrinsics));
    if (cfg.enable_x86_64_intrinsics || cfg.force_x86_64_intrinsics) {
        MPM.addPass(std::make_unique<X8664HardwareIntrinsicPass>(cfg.force_x86_64_intrinsics));
    }
    MPM.addPass(std::make_unique<ExclusiveMonitorLSEFastPathPass>(cfg.enable_lse_fastpath));
    MPM.addPass(std::make_unique<HorizonLoweringPass>());
    MPM.addPass(std::make_unique<HorizonTLSAndSpinWaitPass>(cfg.enable_tls_spinwait));
    MPM.addPass(std::make_unique<GuestStackAndAlignmentPass>(cfg.enable_stack_alignment));
    MPM.addPass(std::make_unique<AArch64IdiomDeconstructPass>(cfg.enable_idiom_deconstruct));
    MPM.addPass(std::make_unique<JumpTableLoweringPass>());
    MPM.addPass(std::make_unique<ShadowDispatchTablePass>(cfg.enable_shadow_dispatch));
    MPM.addPass(std::make_unique<FCMPPredicateFusionPass>(cfg.enable_fcmp_fusion));
    MPM.addPass(std::make_unique<NZCVPredicateFusionPass>(cfg.enable_nzcv_fusion, cfg.enable_nzcv_interblock_dse));
    MPM.addPass(std::make_unique<NZCVDeadFlagElimPass>(cfg.enable_nzcv_dead_flag_elim));
    MPM.addPass(std::make_unique<LinkRegisterElisionPass>(cfg.enable_link_reg_elision));
    MPM.addPass(std::make_unique<TailCallDevirtualizationPass>(cfg.enable_tail_call_devirt));
    MPM.addPass(std::make_unique<AAPCS64ABIContractPass>(cfg.enable_aapcs64_abi_contract, cfg.allow_external_scratch_elim));
    MPM.addPass(std::make_unique<CpuStatePromotionPass>(cfg.enable_cpu_entry_kill_pruning, cfg.enable_cpu_interproc_pruning));
    MPM.addPass(std::make_unique<CrossChunkLeafInlinePass>(
        cfg.global_leaves, cfg.text_lo, cfg.text_hi, 12, cfg.enable_cross_chunk_leaf_inline));
    MPM.addPass(std::make_unique<AArch64StaticNCEABIPass>(cfg.enable_nce_abi));
}

void build_raw_lowering_pipeline(ModulePassManager& MPM, const PassPipelineConfig& cfg) {
    MPM.addPass(std::make_unique<UltraTearsPatchPass>(cfg.extra_patches, cfg.include_default_drs));
    MPM.addPass(std::make_unique<HorizonLoweringPass>());
    MPM.addPass(std::make_unique<JumpTableLoweringPass>());
}

std::string serialize_unlowered_reference_ir(const Module& M, const PassPipelineConfig& cfg) {
    Module copy_mod(M.name + "_raw", M.target_triple);
    copy_mod.external_guest_fns = M.external_guest_fns;
    copy_mod.external_hle_fns = M.external_hle_fns;
    copy_mod.global_defs = M.global_defs;
    copy_mod.needs_noalias_md = M.needs_noalias_md;
    for (const auto& fn : M.functions) {
        Function* cfn = copy_mod.create_function(fn->name, fn->entry_pc, fn->end_pc);
        cfn->linkage = fn->linkage;
        cfn->hle_symbol = fn->hle_symbol;
        cfn->has_all_labels = fn->has_all_labels;
        cfn->jump_tables = fn->jump_tables;
        cfn->label_pcs = fn->label_pcs;
        cfn->direct_callee_pcs = fn->direct_callee_pcs;
        cfn->raw_words = fn->raw_words;
        cfn->next_ssa_id = fn->next_ssa_id;
        for (const auto& bb : fn->blocks) {
            BasicBlock* cbb = cfn->create_block(bb->name, bb->start_pc);
            cbb->insts = bb->insts;
        }
    }
    ModulePassManager raw_mpm;
    build_raw_lowering_pipeline(raw_mpm, cfg);
    raw_mpm.run(copy_mod);
    return copy_mod.serialize();
}

} // namespace ultrs::passes

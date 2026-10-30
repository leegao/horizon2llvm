// UltraTears LLVM IR Lowering, Optimization, Switch-Specific & Static-NCE Passes Header (tools/lifter/passes.hpp)
//
// Declares the 23 modular LLVM IR passes (implemented in tools/lifter/pass_*.cpp and tools/lifter/passes.cpp):
//   1.  UltraTearsPatchPass                  (pass_ultratears_patch.cpp)
//   2.  ProfileGuidedEntryHookElimPass       (pass_profile_entry_hook_elim.cpp)
//   3.  NsoRodataConstPropAndDevirtPass      (pass_nso_rodata_constprop_devirt.cpp)
//   4.  FPCRFastFPLoweringPass               (pass_fpcr_fast_fp_lowering.cpp)
//   5.  SIMDPairwiseAndConversionInlinePass  (pass_simd_pairwise_conversion_inline.cpp)
//   6.  AArch64NativeIntrinsicPass           (pass_aarch64_native_intrinsic.cpp)
//   7.  X8664HardwareIntrinsicPass           (pass_x86_64_hardware_intrinsic.cpp)
//   8.  ExclusiveMonitorLSEFastPathPass      (pass_exclusive_monitor_lse_fastpath.cpp)
//   9.  HorizonLoweringPass                  (pass_horizon_lowering.cpp)
//   10. HorizonTLSAndSpinWaitPass            (pass_horizon_tls_spinwait.cpp)
//   11. GuestStackAndAlignmentPass           (pass_guest_stack_alignment.cpp)
//   12. AArch64IdiomDeconstructPass          (pass_aarch64_idiom_deconstruct.cpp)
//   13. JumpTableLoweringPass                (pass_jump_table_lowering.cpp)
//   14. ShadowDispatchTablePass              (pass_shadow_dispatch_table.cpp)
//   15. FCMPPredicateFusionPass              (pass_fcmp_predicate_fusion.cpp)
//   16. NZCVPredicateFusionPass              (pass_nzcv_predicate_fusion.cpp)
//   17. NZCVDeadFlagElimPass                 (pass_nzcv_dead_flag_elim.cpp)
//   18. LinkRegisterElisionPass              (pass_link_register_elision.cpp)
//   19. TailCallDevirtualizationPass         (pass_tail_call_devirt.cpp)
//   20. AAPCS64ABIContractPass               (pass_aapcs64_abi_contract.cpp)
//   21. CpuStatePromotionPass                (pass_cpu_state_promotion.cpp)
//   22. CrossChunkLeafInlinePass             (pass_cross_chunk_leaf_inline.cpp)
//   23. AArch64StaticNCEABIPass              (pass_aarch64_static_nce_abi.cpp)
#pragma once

#include <numeric>
#include "aarch64_lifter.hpp"
#include "llvm_ir.hpp"

namespace ultrs::passes {

using namespace ultrs::llvm_ir;
using ultrs::lifter::block_label;
using ultrs::lifter::fn_symbol;
using ultrs::lifter::FpOp;
using ultrs::lifter::LaneOp;

struct PatchRule {
    uint64_t guest_pc = 0;
    uint32_t expected_word = 0;
    std::string name;
    enum class Kind {
        DrsOff,       // Dynamic Resolution Off -> branch to target_pc if !totk_drs_enabled()
        Nop,          // Replace instruction at guest_pc with a no-op
        BranchTarget, // Force unconditional branch to target_pc
        WriteRegImm,  // Set X[reg_idx] = imm_val
    } kind = Kind::Nop;
    uint64_t target_pc = 0;
    int reg_idx = 0;
    uint64_t imm_val = 0;
};

struct LeafFunctionSlice {
    uint64_t entry_pc = 0;
    uint64_t end_pc = 0;
    std::vector<uint32_t> words;
};

// 1. UltraTearsPatchPass
class UltraTearsPatchPass final : public ModulePass, public PassInfoMixin<UltraTearsPatchPass> {
public:
    static constexpr uint64_t DRS_PC_143     = 0x8005843e0ULL;
    static constexpr uint32_t DRS_WORD_143   = 0x360042e8u;
    static constexpr uint64_t DRS_TARGET_143 = 0x800584c3cULL;

    static constexpr uint64_t DRS_PC_142     = 0x8005e9d10ULL;
    static constexpr uint32_t DRS_WORD_142   = 0x360041a8u;
    static constexpr uint64_t DRS_TARGET_142 = 0x8005ea544ULL;

    explicit UltraTearsPatchPass(std::vector<PatchRule> extra_rules = {}, bool include_default_drs = true)
        : extra_rules_(std::move(extra_rules)), include_default_drs_(include_default_drs) {}

    static const char* static_name() { return "ultratears-patch-pass"; }
    const char* pass_name() const override { return "ultratears-patch-pass"; }
    bool run(Module& M) override;

private:
    std::vector<PatchRule> extra_rules_;
    bool include_default_drs_;
};

// 2. ProfileGuidedEntryHookElimPass
class ProfileGuidedEntryHookElimPass final : public ModulePass, public PassInfoMixin<ProfileGuidedEntryHookElimPass> {
public:
    explicit ProfileGuidedEntryHookElimPass(size_t max_leaf_insns = 16, bool force_fnprof = false)
        : max_leaf_insns_(max_leaf_insns), force_fnprof_(force_fnprof) {}

    static const char* static_name() { return "profile-entry-hook-elim-pass"; }
    const char* pass_name() const override { return "profile-entry-hook-elim-pass"; }
    bool run(Module& M) override;

private:
    size_t max_leaf_insns_;
    bool force_fnprof_;
};

// 3. NsoRodataConstPropAndDevirtPass
class NsoRodataConstPropAndDevirtPass final : public ModulePass, public PassInfoMixin<NsoRodataConstPropAndDevirtPass> {
public:
    NsoRodataConstPropAndDevirtPass(const uint8_t* image_data = nullptr,
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
    const char* pass_name() const override { return "nso-rodata-constprop-devirt-pass"; }
    bool run(Module& M) override;

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

// 4. FPCRFastFPLoweringPass
class FPCRFastFPLoweringPass final : public ModulePass, public PassInfoMixin<FPCRFastFPLoweringPass> {
public:
    explicit FPCRFastFPLoweringPass(bool enabled = true) : enabled_(enabled) {}
    static const char* static_name() { return "fpcr-fast-fp-lowering-pass"; }
    const char* pass_name() const override { return "fpcr-fast-fp-lowering-pass"; }
    bool run(Module& M) override;
private:
    bool enabled_;
};

// 5. SIMDPairwiseAndConversionInlinePass
class SIMDPairwiseAndConversionInlinePass final : public ModulePass, public PassInfoMixin<SIMDPairwiseAndConversionInlinePass> {
public:
    explicit SIMDPairwiseAndConversionInlinePass(bool enabled = true) : enabled_(enabled) {}
    static const char* static_name() { return "simd-pairwise-conversion-inline-pass"; }
    const char* pass_name() const override { return "simd-pairwise-conversion-inline-pass"; }
    bool run(Module& M) override;
private:
    bool enabled_;
};

// 6. AArch64NativeIntrinsicPass
class AArch64NativeIntrinsicPass final : public ModulePass, public PassInfoMixin<AArch64NativeIntrinsicPass> {
public:
    explicit AArch64NativeIntrinsicPass(bool force_enable = false) : force_enable_(force_enable) {}
    static const char* static_name() { return "aarch64-native-intrinsic-pass"; }
    const char* pass_name() const override { return "aarch64-native-intrinsic-pass"; }
    bool run(Module& M) override;
private:
    bool force_enable_;
};

// 7. X8664HardwareIntrinsicPass
class X8664HardwareIntrinsicPass final : public ModulePass, public PassInfoMixin<X8664HardwareIntrinsicPass> {
public:
    explicit X8664HardwareIntrinsicPass(bool force_enable = false) : force_enable_(force_enable) {}
    static const char* static_name() { return "x86-64-hardware-intrinsic-pass"; }
    const char* pass_name() const override { return "x86-64-hardware-intrinsic-pass"; }
    bool run(Module& M) override;
private:
    bool force_enable_;
};

// 8. ExclusiveMonitorLSEFastPathPass
class ExclusiveMonitorLSEFastPathPass final : public ModulePass, public PassInfoMixin<ExclusiveMonitorLSEFastPathPass> {
public:
    explicit ExclusiveMonitorLSEFastPathPass(bool enabled = true) : enabled_(enabled) {}
    static const char* static_name() { return "exclusive-monitor-lse-fastpath-pass"; }
    const char* pass_name() const override { return "exclusive-monitor-lse-fastpath-pass"; }
    bool run(Module& M) override;
private:
    bool enabled_;
};

// 9. HorizonLoweringPass
class HorizonLoweringPass final : public ModulePass, public PassInfoMixin<HorizonLoweringPass> {
public:
    static const char* static_name() { return "horizon-lowering-pass"; }
    const char* pass_name() const override { return "horizon-lowering-pass"; }
    bool run(Module& M) override;
};

// 10. HorizonTLSAndSpinWaitPass
class HorizonTLSAndSpinWaitPass final : public ModulePass, public PassInfoMixin<HorizonTLSAndSpinWaitPass> {
public:
    explicit HorizonTLSAndSpinWaitPass(bool enabled = true) : enabled_(enabled) {}
    static const char* static_name() { return "horizon-tls-spinwait-pass"; }
    const char* pass_name() const override { return "horizon-tls-spinwait-pass"; }
    bool run(Module& M) override;
private:
    bool enabled_;
};

// 11. GuestStackAndAlignmentPass
class GuestStackAndAlignmentPass final : public ModulePass, public PassInfoMixin<GuestStackAndAlignmentPass> {
public:
    explicit GuestStackAndAlignmentPass(bool enabled = true) : enabled_(enabled) {}
    static const char* static_name() { return "guest-stack-alignment-pass"; }
    const char* pass_name() const override { return "guest-stack-alignment-pass"; }
    bool run(Module& M) override;
private:
    bool enabled_;
};

// 12. AArch64IdiomDeconstructPass
class AArch64IdiomDeconstructPass final : public ModulePass, public PassInfoMixin<AArch64IdiomDeconstructPass> {
public:
    explicit AArch64IdiomDeconstructPass(bool enabled = true) : enabled_(enabled) {}
    static const char* static_name() { return "aarch64-idiom-deconstruct-pass"; }
    const char* pass_name() const override { return "aarch64-idiom-deconstruct-pass"; }
    bool run(Module& M) override;
private:
    bool enabled_;
};

// 13. JumpTableLoweringPass
class JumpTableLoweringPass final : public ModulePass, public PassInfoMixin<JumpTableLoweringPass> {
public:
    static const char* static_name() { return "jump-table-lowering-pass"; }
    const char* pass_name() const override { return "jump-table-lowering-pass"; }
    bool run(Module& M) override;
};

// 14. ShadowDispatchTablePass
class ShadowDispatchTablePass final : public ModulePass, public PassInfoMixin<ShadowDispatchTablePass> {
public:
    explicit ShadowDispatchTablePass(bool enabled = true) : enabled_(enabled) {}
    static const char* static_name() { return "shadow-dispatch-table-pass"; }
    const char* pass_name() const override { return "shadow-dispatch-table-pass"; }
    bool run(Module& M) override;
private:
    bool enabled_;
};

// 15. FCMPPredicateFusionPass
class FCMPPredicateFusionPass final : public ModulePass, public PassInfoMixin<FCMPPredicateFusionPass> {
public:
    explicit FCMPPredicateFusionPass(bool enabled = true) : enabled_(enabled) {}
    static const char* static_name() { return "fcmp-predicate-fusion-pass"; }
    const char* pass_name() const override { return "fcmp-predicate-fusion-pass"; }
    bool run(Module& M) override;
private:
    bool enabled_;
};

// 16. NZCVPredicateFusionPass
class NZCVPredicateFusionPass final : public ModulePass, public PassInfoMixin<NZCVPredicateFusionPass> {
public:
    explicit NZCVPredicateFusionPass(bool enabled = true, bool enable_interblock_dse = true)
        : enabled_(enabled), enable_interblock_dse_(enable_interblock_dse) {}
    static const char* static_name() { return "nzcv-predicate-fusion-pass"; }
    const char* pass_name() const override { return "nzcv-predicate-fusion-pass"; }
    bool run(Module& M) override;
private:
    bool enabled_;
    bool enable_interblock_dse_;
};

// 17. NZCVDeadFlagElimPass
class NZCVDeadFlagElimPass final : public ModulePass, public PassInfoMixin<NZCVDeadFlagElimPass> {
public:
    explicit NZCVDeadFlagElimPass(bool enabled = true) : enabled_(enabled) {}
    static const char* static_name() { return "nzcv-dead-flag-elim-pass"; }
    const char* pass_name() const override { return "nzcv-dead-flag-elim-pass"; }
    bool run(Module& M) override;
private:
    bool enabled_;
};

// 18. LinkRegisterElisionPass
class LinkRegisterElisionPass final : public ModulePass, public PassInfoMixin<LinkRegisterElisionPass> {
public:
    explicit LinkRegisterElisionPass(bool enabled = true) : enabled_(enabled) {}
    static const char* static_name() { return "link-register-elision-pass"; }
    const char* pass_name() const override { return "link-register-elision-pass"; }
    bool run(Module& M) override;
private:
    bool enabled_;
};

// 19. TailCallDevirtualizationPass
class TailCallDevirtualizationPass final : public ModulePass, public PassInfoMixin<TailCallDevirtualizationPass> {
public:
    explicit TailCallDevirtualizationPass(bool enabled = true) : enabled_(enabled) {}
    static const char* static_name() { return "tail-call-devirt-pass"; }
    const char* pass_name() const override { return "tail-call-devirt-pass"; }
    bool run(Module& M) override;
private:
    bool enabled_;
};

// 20. AAPCS64ABIContractPass
class AAPCS64ABIContractPass final : public ModulePass, public PassInfoMixin<AAPCS64ABIContractPass> {
public:
    explicit AAPCS64ABIContractPass(bool enabled = true, bool allow_external_scratch_elim = false)
        : enabled_(enabled), allow_external_scratch_elim_(allow_external_scratch_elim) {}
    static const char* static_name() { return "aapcs64-abi-contract-pass"; }
    const char* pass_name() const override { return "aapcs64-abi-contract-pass"; }
    bool run(Module& M) override;
private:
    bool enabled_;
    bool allow_external_scratch_elim_;
};

// 21. CpuStatePromotionPass
class CpuStatePromotionPass final : public ModulePass, public PassInfoMixin<CpuStatePromotionPass> {
public:
    explicit CpuStatePromotionPass(bool enable_entry_kill_pruning = true,
                                   bool enable_interproc_pruning = true)
        : enable_entry_kill_pruning_(enable_entry_kill_pruning),
          enable_interproc_pruning_(enable_interproc_pruning) {}
    static const char* static_name() { return "cpu-state-promotion-pass"; }
    const char* pass_name() const override { return "cpu-state-promotion-pass"; }
    bool run(Module& M) override;
private:
    bool enable_entry_kill_pruning_;
    bool enable_interproc_pruning_;
};

// 22. CrossChunkLeafInlinePass
class CrossChunkLeafInlinePass final : public ModulePass, public PassInfoMixin<CrossChunkLeafInlinePass> {
public:
    explicit CrossChunkLeafInlinePass(const std::unordered_map<uint64_t, LeafFunctionSlice>* global_leaves = nullptr,
                                      uint64_t text_lo = 0,
                                      uint64_t text_hi = 0,
                                      size_t max_leaf_words = 12,
                                      bool enabled = true)
        : global_leaves_(global_leaves),
          text_lo_(text_lo),
          text_hi_(text_hi),
          max_leaf_words_(max_leaf_words),
          enabled_(enabled) {}

    static const char* static_name() { return "cross-chunk-leaf-inline-pass"; }
    const char* pass_name() const override { return "cross-chunk-leaf-inline-pass"; }

    static std::unordered_map<uint64_t, LeafFunctionSlice> collect_nso_leaves(
        const std::vector<uint64_t>& order,
        uint64_t text_lo,
        uint64_t text_hi,
        const std::vector<uint32_t>& words,
        const std::unordered_map<uint64_t, std::string>& hle_map,
        size_t max_leaf_words = 12);

    bool run(Module& M) override;

private:
    const std::unordered_map<uint64_t, LeafFunctionSlice>* global_leaves_;
    uint64_t text_lo_;
    uint64_t text_hi_;
    size_t max_leaf_words_;
    bool enabled_;
};

// 23. AArch64StaticNCEABIPass
class AArch64StaticNCEABIPass final : public ModulePass, public PassInfoMixin<AArch64StaticNCEABIPass> {
public:
    explicit AArch64StaticNCEABIPass(bool enabled = false) : enabled_(enabled) {}
    static const char* static_name() { return "aarch64-static-nce-abi-pass"; }
    const char* pass_name() const override { return "aarch64-static-nce-abi-pass"; }
    bool run(Module& M) override;
private:
    bool enabled_;
};

struct PassPipelineConfig {
    std::vector<PatchRule> extra_patches;
    bool include_default_drs = true;
    bool force_fnprof = false;
    bool enable_fast_fp = true;
    bool enable_simd_inline = true;
    bool enable_shadow_dispatch = true;
    bool force_aarch64_intrinsics = false;
    bool force_x86_64_intrinsics = false;
    bool enable_x86_64_intrinsics = true;
    bool enable_tls_spinwait = true;
    bool enable_stack_alignment = true;
    bool enable_idiom_deconstruct = true;
    bool enable_fcmp_fusion = true;
    bool enable_nzcv_fusion = true;
    bool enable_nzcv_dead_flag_elim = true;
    bool enable_cpu_entry_kill_pruning = true;

    bool enable_rodata_constprop = true;
    bool enable_lse_fastpath = true;
    bool enable_nzcv_interblock_dse = true;
    bool enable_link_reg_elision = true;
    bool enable_tail_call_devirt = true;
    bool enable_aapcs64_abi_contract = true;
    bool allow_external_scratch_elim = false;
    bool enable_cpu_interproc_pruning = true;
    bool enable_cross_chunk_leaf_inline = true;
    bool enable_nce_abi = false;

    const uint8_t* image_data = nullptr;
    size_t image_size = 0;
    uint64_t image_base = 0;
    uint64_t rodata_lo = 0;
    uint64_t rodata_hi = 0;
    uint64_t text_lo = 0;
    uint64_t text_hi = 0;
    const std::unordered_set<uint64_t>* unrelocated_rodata_addrs = nullptr;
    const std::unordered_map<uint64_t, LeafFunctionSlice>* global_leaves = nullptr;

    void disable_high_risk_passes();
    void enable_all_passes();
    void disable_all_opt_passes();
    bool set_pass_enabled(const std::string& name, bool en);
};

// Build the full 23-pass UltraTears pipeline
void build_ultratears_pipeline(ModulePassManager& MPM, const PassPipelineConfig& cfg = {});

// Build the minimal structural lowering pipeline (Horizon system insns + JumpTables only, no optimizations)
// used when emitting compilable unoptimized reference IR (`main_XX_raw.ll`).
void build_raw_lowering_pipeline(ModulePassManager& MPM, const PassPipelineConfig& cfg = {});

// Produce a standalone, compilable unoptimized LLVM IR string from a freshly-lifted Module
// without mutating `M` prior to running the full optimization pipeline.
std::string serialize_unlowered_reference_ir(const Module& M, const PassPipelineConfig& cfg = {});

} // namespace ultrs::passes
// UltraTears LLVM IR Infrastructure (tools/lifter/llvm_ir.hpp)
//
// Thin API layer on top of LLVM 21 (`llvm::LLVMContext`, `llvm::Module`, `llvm::Function`,
// `llvm::BasicBlock`, `llvm::Instruction`, `llvm::Value`, and `llvm::IRBuilder<llvm::NoFolder>`)
// that preserves semantic AArch64-Horizon lifting metadata (`OpTag`, `guest_pc`, `insn_word`,
// `arch_reg`, `cond_code`, `cmp_kind`) while constructing real LLVM IR objects via `llvm::IRBuilder`.
#pragma once

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

namespace ultrs::llvm_ir {

// -----------------------------------------------------------------------------
// LLVM IR Types
// -----------------------------------------------------------------------------
enum class TypeKind {
    Void,
    Int1,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    Float,
    Double,
    Pointer,
    Vec16xI8,
    Vec8xI16,
    Vec4xI32,
    Vec2xI64,
    Vec4xF32,
    Vec2xF64,
};

inline const char* type_str(TypeKind t) {
    switch (t) {
        case TypeKind::Void:     return "void";
        case TypeKind::Int1:     return "i1";
        case TypeKind::Int8:     return "i8";
        case TypeKind::Int16:    return "i16";
        case TypeKind::Int32:    return "i32";
        case TypeKind::Int64:    return "i64";
        case TypeKind::Int128:   return "i128";
        case TypeKind::Float:    return "float";
        case TypeKind::Double:   return "double";
        case TypeKind::Pointer:  return "ptr";
        case TypeKind::Vec16xI8: return "<16 x i8>";
        case TypeKind::Vec8xI16: return "<8 x i16>";
        case TypeKind::Vec4xI32: return "<4 x i32>";
        case TypeKind::Vec2xI64: return "<2 x i64>";
        case TypeKind::Vec4xF32: return "<4 x float>";
        case TypeKind::Vec2xF64: return "<2 x double>";
    }
    return "i64";
}

inline llvm::Type* to_llvm_type(llvm::LLVMContext& ctx, TypeKind t) {
    switch (t) {
        case TypeKind::Void:     return llvm::Type::getVoidTy(ctx);
        case TypeKind::Int1:     return llvm::Type::getInt1Ty(ctx);
        case TypeKind::Int8:     return llvm::Type::getInt8Ty(ctx);
        case TypeKind::Int16:    return llvm::Type::getInt16Ty(ctx);
        case TypeKind::Int32:    return llvm::Type::getInt32Ty(ctx);
        case TypeKind::Int64:    return llvm::Type::getInt64Ty(ctx);
        case TypeKind::Int128:   return llvm::Type::getInt128Ty(ctx);
        case TypeKind::Float:    return llvm::Type::getFloatTy(ctx);
        case TypeKind::Double:   return llvm::Type::getDoubleTy(ctx);
        case TypeKind::Pointer:  return llvm::PointerType::getUnqual(ctx);
        case TypeKind::Vec16xI8: return llvm::FixedVectorType::get(llvm::Type::getInt8Ty(ctx), 16);
        case TypeKind::Vec8xI16: return llvm::FixedVectorType::get(llvm::Type::getInt16Ty(ctx), 8);
        case TypeKind::Vec4xI32: return llvm::FixedVectorType::get(llvm::Type::getInt32Ty(ctx), 4);
        case TypeKind::Vec2xI64: return llvm::FixedVectorType::get(llvm::Type::getInt64Ty(ctx), 2);
        case TypeKind::Vec4xF32: return llvm::FixedVectorType::get(llvm::Type::getFloatTy(ctx), 4);
        case TypeKind::Vec2xF64: return llvm::FixedVectorType::get(llvm::Type::getDoubleTy(ctx), 2);
    }
    return llvm::Type::getInt64Ty(ctx);
}

inline unsigned vec_lane_count(TypeKind vt) {
    switch (vt) {
        case TypeKind::Vec16xI8: return 16;
        case TypeKind::Vec8xI16: return 8;
        case TypeKind::Vec4xI32:
        case TypeKind::Vec4xF32: return 4;
        case TypeKind::Vec2xI64:
        case TypeKind::Vec2xF64: return 2;
        default: return 1;
    }
}

inline TypeKind int_type_for_bits(unsigned bits) {
    switch (bits) {
        case 1:   return TypeKind::Int1;
        case 8:   return TypeKind::Int8;
        case 16:  return TypeKind::Int16;
        case 32:  return TypeKind::Int32;
        case 64:  return TypeKind::Int64;
        case 128: return TypeKind::Int128;
        default:  return TypeKind::Int64;
    }
}

inline TypeKind vec_int_type(unsigned sz) {
    switch (sz) {
        case 0: return TypeKind::Vec16xI8;
        case 1: return TypeKind::Vec8xI16;
        case 2: return TypeKind::Vec4xI32;
        default: return TypeKind::Vec2xI64;
    }
}

// Format a single llvm::Value/Instruction using LLVM's native AsmWriter
inline std::string format_llvm_inst(const llvm::Value* v) {
    std::string s;
    llvm::raw_string_ostream os(s);
    v->print(os);
    os.flush();
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) {
        s.erase(s.begin());
    }
    return s;
}

inline std::string strip_ssa_pct(const std::string& name) {
    if (!name.empty() && name[0] == '%') return name.substr(1);
    return name;
}

// -----------------------------------------------------------------------------
// SSA Value Reference (Thin wrapper over llvm::Value*)
// -----------------------------------------------------------------------------
struct Value {
    TypeKind type = TypeKind::Int64;
    std::string repr; // e.g. "%t12", "42", "0", "@f_800004000"
    bool is_const = false;
    uint64_t const_u64 = 0;
    llvm::Value* llvm_val = nullptr;

    static Value reg(TypeKind t, std::string name, llvm::Value* lv = nullptr) {
        return Value{t, std::move(name), false, 0, lv};
    }
    static Value cpu_ptr(llvm::Value* lv = nullptr) {
        return Value{TypeKind::Pointer, "%c", false, 0, lv};
    }
    static Value imm(TypeKind t, uint64_t v) {
        if (t == TypeKind::Int1) {
            return Value{t, (v & 1) ? "true" : "false", true, v & 1, nullptr};
        }
        return Value{t, std::to_string(static_cast<int64_t>(v)), true, v, nullptr};
    }
    static Value imm_u64(uint64_t v) {
        return Value{TypeKind::Int64, std::to_string(v), true, v, nullptr};
    }
    static Value imm_i32(int64_t v) {
        return Value{TypeKind::Int32, std::to_string(static_cast<int32_t>(v)), true, static_cast<uint32_t>(v), nullptr};
    }
    static Value imm_i128(uint64_t lo, uint64_t hi) {
        if (hi == 0) {
            return Value{TypeKind::Int128, std::to_string(lo), true, lo, nullptr};
        }
        char buf[64];
        std::snprintf(buf, sizeof(buf), "u0x%016" PRIx64 "%016" PRIx64, hi, lo);
        return Value{TypeKind::Int128, buf, true, lo, nullptr};
    }
    static Value zero_vec(TypeKind vt) {
        return Value{vt, "zeroinitializer", true, 0, nullptr};
    }
    static Value global_sym(std::string sym) {
        if (!sym.empty() && sym[0] == '@') {
            return Value{TypeKind::Pointer, std::move(sym), true, 0, nullptr};
        }
        return Value{TypeKind::Pointer, "@" + std::move(sym), true, 0, nullptr};
    }
};

// -----------------------------------------------------------------------------
// High-Level & Target-Specific IR Opcode Tags for Pass Introspection
// -----------------------------------------------------------------------------
enum class OpTag {
    RawIR,              // Standard LLVM IR instruction
    Comment,            // IR comment (; ...)
    GuestInsnMarker,    // Metadata marker for guest PC + raw 32-bit instruction word
    CpuRegRead,         // Read of architectural register (X0..X30, SP, NZCV, V0..V31)
    CpuRegWrite,        // Write of architectural register
    FlagWrite,          // Write to NZCV flags (candidate for NZCVPredicateFusionPass & NZCVDeadFlagElimPass)
    CondBranch,         // Conditional branch using condition code / NZCV
    UncondBranch,       // Unconditional branch
    DirectCall,         // Direct guest function call (BL)
    IndirectCall,       // Indirect guest call (BLR -> rt_dispatch)
    IndirectJump,       // Indirect jump / jump-table candidate (BR)
    Return,             // RET
    HorizonSysRegRead,  // High-level MRS placeholder lowered by HorizonLoweringPass
    HorizonSysRegWrite, // High-level MSR placeholder lowered by HorizonLoweringPass
    HorizonSvc,         // High-level SVC placeholder lowered by HorizonLoweringPass
    HorizonBrk,         // High-level BRK placeholder lowered by HorizonLoweringPass
    HorizonUdf,         // High-level UDF placeholder lowered by HorizonLoweringPass
    HorizonBarrier,     // High-level DMB/DSB/ISB/CLREX placeholder lowered by HorizonLoweringPass
    HorizonDcZva,       // High-level DC ZVA placeholder lowered by HorizonLoweringPass
    GuestMemLoad,       // Guest memory load (annotated with !noalias by HorizonLoweringPass)
    GuestMemStore,      // Guest memory store (annotated with !noalias by HorizonLoweringPass)
};

struct Instruction {
    OpTag tag = OpTag::RawIR;
    std::string text;           // Rendered LLVM IR statement (generated via llvm::Instruction::print)
    std::string prefix_ir;      // Optional prefix SSA computations (e.g. packed NZCV bit math)
    std::string dst_reg;        // SSA result register name if any (e.g., "%v12")
    TypeKind dst_type = TypeKind::Void;
    llvm::Instruction* llvm_inst = nullptr; // Underlying LLVM Instruction node

    // Semantic fields for LLVM Passes to inspect & rewrite:
    uint64_t guest_pc = 0;      // Associated guest instruction PC
    uint32_t insn_word = 0;     // Original 32-bit AArch64 instruction encoding
    int arch_reg = -1;          // 0..30 = X0..X30, 31 = SP, 32 = NZCV, 64..95 = V0..V31
    int cond_code = -1;         // 0..15 AArch64 condition code (for CondBranch / CondEval)
    int cmp_kind = 0;           // 0 = none, 1 = SUBS/CMP, 2 = ADDS/CMN, 3 = ANDS/TST, 4 = FCMP_S, 5 = FCMP_D
    unsigned cmp_sf = 1;        // 1 = 64-bit, 0 = 32-bit comparison width
    int align_bytes = 1;        // Proven alignment in bytes for GuestMemLoad / GuestMemStore
    bool is_stack_access = false; // Proven SP-derived guest stack access
    uint64_t branch_target = 0; // Target guest PC for branch/call
    std::string target_label;   // Target BasicBlock label
    std::string false_label;    // Fallthrough BasicBlock label
    Value op0{};
    Value op1{};
    Value op2{};
    uint32_t sys_key = 0;       // Packed (op0,op1,crn,crm,op2) for MRS/MSR or imm16 for SVC/BRK
    bool dead = false;          // Marked dead by optimization passes
};

struct BasicBlock {
    std::string name;           // e.g. "entry", "L_8005843e0"
    uint64_t start_pc = 0;
    llvm::BasicBlock* llvm_bb = nullptr;
    std::vector<Instruction> insts;
    std::vector<std::string> preds;
    std::vector<std::string> succs;

    bool has_terminator() const {
        for (auto it = insts.rbegin(); it != insts.rend(); ++it) {
            if (it->dead || it->tag == OpTag::Comment || it->tag == OpTag::GuestInsnMarker) continue;
            if (it->tag == OpTag::UncondBranch || it->tag == OpTag::CondBranch ||
                it->tag == OpTag::Return || it->tag == OpTag::IndirectJump ||
                it->tag == OpTag::HorizonBrk || it->tag == OpTag::HorizonUdf) {
                return true;
            }
            if (it->text.rfind("ret ", 0) == 0 || it->text.rfind("br ", 0) == 0 ||
                it->text.rfind("switch ", 0) == 0 || it->text.rfind("unreachable", 0) == 0) {
                return true;
            }
            break;
        }
        return false;
    }
};

struct JumpTableInfo {
    bool is_all_labels = false;
    std::vector<uint64_t> targets;
};

struct Module;

struct Function {
    std::string name;           // e.g. "f_800004000"
    std::string linkage = "dso_local"; // e.g. "dso_local" or "internal alwaysinline"
    uint64_t entry_pc = 0;
    uint64_t end_pc = 0;
    std::string hle_symbol;     // Non-empty if overridden by --hle
    bool has_all_labels = false;
    bool aapcs64_compliant = true;
    bool has_unknown_calls = false;
    bool modifies_fpcr = false;
    bool is_inline_clone = false;
    std::set<int> summary_reads;
    std::set<int> summary_writes;
    std::set<int> dead_on_return_regs;
    std::map<uint64_t, JumpTableInfo> jump_tables;
    std::set<uint64_t> label_pcs;
    std::set<uint64_t> direct_callee_pcs;
    std::map<uint64_t, uint32_t> raw_words; // PC -> original 32-bit word for pass verification
    std::vector<std::unique_ptr<BasicBlock>> blocks;
    std::unordered_map<std::string, BasicBlock*> block_map;
    unsigned next_ssa_id = 1;

    Module* parent_mod = nullptr;
    llvm::Function* llvm_fn = nullptr;
    std::unordered_map<std::string, llvm::Value*> ssa_values;
    std::vector<std::unique_ptr<llvm::Argument>> synthetic_ssa_vals;

    BasicBlock* create_block(const std::string& label, uint64_t pc = 0);

    BasicBlock* get_block(const std::string& label) const {
        auto it = block_map.find(label);
        return it != block_map.end() ? it->second : nullptr;
    }

    std::string next_reg(const char* prefix = "t") {
        return std::string("%") + prefix + std::to_string(next_ssa_id++);
    }

    void register_ssa_value(const std::string& reg_name, llvm::Value* val) {
        if (!reg_name.empty() && val) {
            ssa_values[reg_name] = val;
        }
    }
};

struct Module {
    std::string name;
    std::string target_triple;
    std::string data_layout;
    std::unique_ptr<llvm::LLVMContext> llvm_ctx;
    std::unique_ptr<llvm::Module> llvm_mod;
    std::vector<std::unique_ptr<Function>> functions;
    std::unordered_map<std::string, Function*> fn_map;
    std::set<std::string> external_guest_fns;
    std::set<std::string> external_hle_fns;
    std::vector<std::string> global_defs;
    std::vector<std::string> extra_raw_functions; // e.g. @f_<pc>_nce Static NCE ABI entry points
    bool needs_noalias_md = true;
    bool uses_aarch64_intrinsics = false;
    bool uses_x86_pause = false;
    bool uses_x86_crc32 = false;

    Module(std::string mod_name, std::string triple = "x86_64-w64-windows-gnu")
        : name(std::move(mod_name)),
          target_triple(std::move(triple)),
          llvm_ctx(std::make_unique<llvm::LLVMContext>()),
          llvm_mod(std::make_unique<llvm::Module>(name, *llvm_ctx)) {
        if (target_triple.find("aarch64") != std::string::npos || target_triple.find("arm64") != std::string::npos) {
            data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128";
        } else if (target_triple.find("windows") != std::string::npos || target_triple.find("win32") != std::string::npos) {
            data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128";
        } else {
            data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128";
        }
        llvm_mod->setTargetTriple(llvm::Triple(target_triple));
        llvm_mod->setDataLayout(data_layout);
        init_runtime_declarations();
    }

    llvm::LLVMContext& context() const { return *llvm_ctx; }
    llvm::Module& raw_module() const { return *llvm_mod; }

    // Synchronize the underlying llvm::Module with all current pass transformations
    llvm::Module* sync_llvm_module() {
        std::string ir_str = serialize();
        llvm::SMDiagnostic diag;
        auto parsed = llvm::parseAssemblyString(ir_str, diag, *llvm_ctx);
        if (parsed) {
            llvm_mod = std::move(parsed);
        }
        return llvm_mod.get();
    }

    void init_runtime_declarations() {
        auto& ctx = *llvm_ctx;
        auto* i32_ty = llvm::Type::getInt32Ty(ctx);
        auto* i64_ty = llvm::Type::getInt64Ty(ctx);
        auto* ptr_ty = llvm::PointerType::getUnqual(ctx);
        auto* void_ty = llvm::Type::getVoidTy(ctx);

        if (!llvm_mod->getNamedGlobal("g_fnprof")) {
            auto* g = new llvm::GlobalVariable(*llvm_mod, i32_ty, false,
                                               llvm::GlobalValue::ExternalLinkage, nullptr, "g_fnprof");
            g->setDSOLocal(true);
            g->setAlignment(llvm::Align(4));
        }
        if (!llvm_mod->getNamedGlobal("g_icache_pc")) {
            auto* arr_ty = llvm::ArrayType::get(i64_ty, 65536);
            auto* g = new llvm::GlobalVariable(*llvm_mod, arr_ty, false,
                                               llvm::GlobalValue::ExternalLinkage, nullptr, "g_icache_pc");
            g->setDSOLocal(true);
            g->setAlignment(llvm::Align(8));
        }
        if (!llvm_mod->getNamedGlobal("g_icache_fn")) {
            auto* arr_ty = llvm::ArrayType::get(ptr_ty, 65536);
            auto* g = new llvm::GlobalVariable(*llvm_mod, arr_ty, false,
                                               llvm::GlobalValue::ExternalLinkage, nullptr, "g_icache_fn");
            g->setDSOLocal(true);
            g->setAlignment(llvm::Align(8));
        }

        llvm_mod->getOrInsertFunction("fnprof_hit", llvm::FunctionType::get(void_ty, {i64_ty, i64_ty, i64_ty}, false));
        llvm_mod->getOrInsertFunction("rt_dispatch", llvm::FunctionType::get(void_ty, {ptr_ty}, false));
        llvm_mod->getOrInsertFunction("rt_svc", llvm::FunctionType::get(void_ty, {ptr_ty, i32_ty, i64_ty}, false));
        llvm_mod->getOrInsertFunction("rt_brk", llvm::FunctionType::get(void_ty, {ptr_ty, i32_ty, i64_ty}, false));
        llvm_mod->getOrInsertFunction("rt_udf", llvm::FunctionType::get(void_ty, {ptr_ty, i64_ty}, false));
        llvm_mod->getOrInsertFunction("rt_ticks", llvm::FunctionType::get(i64_ty, {}, false));
        llvm_mod->getOrInsertFunction("totk_drs_enabled", llvm::FunctionType::get(i32_ty, {}, false));
        llvm_mod->getOrInsertFunction("rt_fpcr_write_ext", llvm::FunctionType::get(void_ty, {ptr_ty, i64_ty}, false));
    }

    bool is_aarch64_target() const {
        return target_triple.find("aarch64") != std::string::npos ||
               target_triple.find("arm64") != std::string::npos;
    }

    bool is_x86_64_target() const {
        return target_triple.find("x86_64") != std::string::npos ||
               target_triple.find("amd64") != std::string::npos;
    }

    Function* create_function(const std::string& fname, uint64_t entry, uint64_t end) {
        auto fn = std::make_unique<Function>();
        fn->name = fname;
        fn->entry_pc = entry;
        fn->end_pc = end;
        fn->parent_mod = this;

        auto* ptr_ty = llvm::PointerType::getUnqual(*llvm_ctx);
        auto* fn_ty = llvm::FunctionType::get(llvm::Type::getVoidTy(*llvm_ctx), {ptr_ty}, false);
        auto callee = llvm_mod->getOrInsertFunction(fname, fn_ty);
        fn->llvm_fn = llvm::dyn_cast<llvm::Function>(callee.getCallee());
        if (fn->llvm_fn) {
            fn->llvm_fn->setDSOLocal(true);
            fn->llvm_fn->addFnAttr(llvm::Attribute::NoUnwind);
            if (!fn->llvm_fn->arg_empty()) {
                llvm::Argument* arg0 = fn->llvm_fn->getArg(0);
                arg0->setName("c");
                arg0->addAttr(llvm::Attribute::NoAlias);
                arg0->addAttr(llvm::Attribute::NoUndef);
                arg0->addAttr(llvm::Attribute::NonNull);
                arg0->addAttr(llvm::Attribute::getWithAlignment(*llvm_ctx, llvm::Align(16)));
                arg0->addAttr(llvm::Attribute::getWithDereferenceableBytes(*llvm_ctx, 1024));
                fn->ssa_values["%c"] = arg0;
            }
        }

        Function* ptr = fn.get();
        fn_map[fname] = ptr;
        functions.push_back(std::move(fn));
        return ptr;
    }

    std::string serialize() const {
        std::ostringstream os;
        os << "; ModuleID = '" << name << "'\n";
        os << "source_filename = \"" << name << "\"\n";
        if (!data_layout.empty()) {
            os << "target datalayout = \"" << data_layout << "\"\n";
        }
        if (!target_triple.empty()) {
            os << "target triple = \"" << target_triple << "\"\n";
        }
        os << "\n";

        // Runtime & LLVM Intrinsic Declarations
        os << "%struct.Cpu = type { [32 x i64], i64, i64, i64, i64, i64, i64, i64, ptr, i64, i64, [176 x i8], [32 x i128] }\n\n";
        os << "@g_fnprof = external dso_local global i32, align 4\n";
        os << "@g_icache_pc = external dso_local global [65536 x i64], align 8\n";
        os << "@g_icache_fn = external dso_local global [65536 x ptr], align 8\n";
        os << "declare dso_local void @fnprof_hit(i64, i64, i64)\n";
        os << "declare dso_local void @rt_dispatch(ptr)\n";
        os << "declare dso_local void @rt_svc(ptr, i32, i64)\n";
        os << "declare dso_local void @rt_brk(ptr, i32, i64) noreturn\n";
        os << "declare dso_local void @rt_udf(ptr, i64) noreturn\n";
        os << "declare dso_local i64 @rt_ticks()\n";
        os << "declare dso_local i32 @totk_drs_enabled()\n";
        os << "declare dso_local void @rt_fpcr_write_ext(ptr, i64)\n";

        // SIMD/FP/Crypto runtime helpers (src/runtime/simd_rt.c)
        os << "declare dso_local void @rt_sl_gen(ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64)\n";
        os << "declare dso_local void @rt_sl_addlp(ptr, i32, i32, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_sl_across(ptr, i32, i32, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_sl_pmull64(ptr, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_sv_tbl(ptr, i32, i32, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_sl_ldm(ptr, i32, i32, i32, i32, i32, i64)\n";
        os << "declare dso_local void @rt_sl_stm(ptr, i32, i32, i32, i32, i32, i64)\n";
        os << "declare dso_local void @rt_sl_ldr(ptr, i32, i32, i32, i32, i64)\n";
        os << "declare dso_local void @rt_sl_lds(ptr, i32, i32, i32, i32, i64)\n";
        os << "declare dso_local void @rt_sl_sts(ptr, i32, i32, i32, i32, i64)\n";
        os << "declare dso_local void @rt_sv_aes(ptr, i32, i32, i32)\n";
        os << "declare dso_local void @rt_sv_sha1hash(ptr, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_sv_sha256hash(ptr, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_sv_sha_misc(ptr, i32, i32, i32, i32)\n";
        os << "declare dso_local i32 @rt_crc32(i32, i64, i32, i32)\n";
        os << "declare dso_local void @rt_fp_op_s(ptr, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_fp_op_d(ptr, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_fp_fma_s(ptr, i32, i32, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_fp_fma_d(ptr, i32, i32, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_fp_dp1(ptr, i32, i32, i32, i32)\n";
        os << "declare dso_local i64 @rt_fp_cmp_s(ptr, i32, i32, i32)\n";
        os << "declare dso_local i64 @rt_fp_cmp_d(ptr, i32, i32, i32)\n";
        os << "declare dso_local i64 @rt_fp_toint(ptr, i32, i32, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_fp_fromint(ptr, i32, i32, i64, i32, i32, i32)\n";
        os << "declare dso_local void @rt_fp_recpe(ptr, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_fp_recpx(ptr, i32, i32, i32)\n";
        os << "declare dso_local void @rt_vec_fop(ptr, i32, i32, i32, i32, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_vec_fpair(ptr, i32, i32, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_vec_fcmp(ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_vec_frint(ptr, i32, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_vec_fcvt_int(ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_vec_frecpe(ptr, i32, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_vec_fcvtn(ptr, i32, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_vec_fcvtl(ptr, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_sv_fred_s(ptr, i32, i32, i32)\n";
        os << "declare dso_local void @rt_scalar_fpair(ptr, i32, i32, i32, i32)\n";
        os << "declare dso_local void @rt_indexed_fp(ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32)\n";

        // Target-Independent LLVM Intrinsics
        os << "declare i32 @llvm.ctlz.i32(i32, i1 immarg) nounwind readnone speculatable willreturn\n";
        os << "declare i64 @llvm.ctlz.i64(i64, i1 immarg) nounwind readnone speculatable willreturn\n";
        os << "declare <16 x i8> @llvm.ctlz.v16i8(<16 x i8>, i1 immarg) nounwind readnone speculatable willreturn\n";
        os << "declare <8 x i16> @llvm.ctlz.v8i16(<8 x i16>, i1 immarg) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x i32> @llvm.ctlz.v4i32(<4 x i32>, i1 immarg) nounwind readnone speculatable willreturn\n";
        os << "declare <16 x i8> @llvm.ctpop.v16i8(<16 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare <8 x i8> @llvm.ctpop.v8i8(<8 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare i32 @llvm.bitreverse.i32(i32) nounwind readnone speculatable willreturn\n";
        os << "declare i64 @llvm.bitreverse.i64(i64) nounwind readnone speculatable willreturn\n";
        os << "declare <16 x i8> @llvm.bitreverse.v16i8(<16 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare i32 @llvm.bswap.i32(i32) nounwind readnone speculatable willreturn\n";
        os << "declare i64 @llvm.bswap.i64(i64) nounwind readnone speculatable willreturn\n";
        os << "declare i32 @llvm.fshr.i32(i32, i32, i32) nounwind readnone speculatable willreturn\n";
        os << "declare i64 @llvm.fshr.i64(i64, i64, i64) nounwind readnone speculatable willreturn\n";
        os << "declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) argmemonly nounwind willreturn writeonly\n";
        os << "declare float @llvm.fma.f32(float, float, float) nounwind readnone speculatable willreturn\n";
        os << "declare double @llvm.fma.f64(double, double, double) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>) nounwind readnone speculatable willreturn\n";
        os << "declare float @llvm.sqrt.f32(float) nounwind readnone speculatable willreturn\n";
        os << "declare double @llvm.sqrt.f64(double) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x float> @llvm.sqrt.v4f32(<4 x float>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x double> @llvm.sqrt.v2f64(<2 x double>) nounwind readnone speculatable willreturn\n";
        os << "declare float @llvm.fabs.f32(float) nounwind readnone speculatable willreturn\n";
        os << "declare double @llvm.fabs.f64(double) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x float> @llvm.fabs.v4f32(<4 x float>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x double> @llvm.fabs.v2f64(<2 x double>) nounwind readnone speculatable willreturn\n";
        os << "declare float @llvm.maxnum.f32(float, float) nounwind readnone speculatable willreturn\n";
        os << "declare double @llvm.maxnum.f64(double, double) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x double> @llvm.maxnum.v2f64(<2 x double>, <2 x double>) nounwind readnone speculatable willreturn\n";
        os << "declare float @llvm.minnum.f32(float, float) nounwind readnone speculatable willreturn\n";
        os << "declare double @llvm.minnum.f64(double, double) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x double> @llvm.minnum.v2f64(<2 x double>, <2 x double>) nounwind readnone speculatable willreturn\n";
        os << "declare float @llvm.rint.f32(float) nounwind readnone speculatable willreturn\n";
        os << "declare double @llvm.rint.f64(double) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x float> @llvm.rint.v4f32(<4 x float>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x double> @llvm.rint.v2f64(<2 x double>) nounwind readnone speculatable willreturn\n";
        os << "declare float @llvm.nearbyint.f32(float) nounwind readnone speculatable willreturn\n";
        os << "declare double @llvm.nearbyint.f64(double) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x float> @llvm.nearbyint.v4f32(<4 x float>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x double> @llvm.nearbyint.v2f64(<2 x double>) nounwind readnone speculatable willreturn\n";
        os << "declare float @llvm.ceil.f32(float) nounwind readnone speculatable willreturn\n";
        os << "declare double @llvm.ceil.f64(double) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x float> @llvm.ceil.v4f32(<4 x float>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x double> @llvm.ceil.v2f64(<2 x double>) nounwind readnone speculatable willreturn\n";
        os << "declare float @llvm.floor.f32(float) nounwind readnone speculatable willreturn\n";
        os << "declare double @llvm.floor.f64(double) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x float> @llvm.floor.v4f32(<4 x float>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x double> @llvm.floor.v2f64(<2 x double>) nounwind readnone speculatable willreturn\n";
        os << "declare float @llvm.trunc.f32(float) nounwind readnone speculatable willreturn\n";
        os << "declare double @llvm.trunc.f64(double) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x float> @llvm.trunc.v4f32(<4 x float>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x double> @llvm.trunc.v2f64(<2 x double>) nounwind readnone speculatable willreturn\n";
        os << "declare float @llvm.round.f32(float) nounwind readnone speculatable willreturn\n";
        os << "declare double @llvm.round.f64(double) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x float> @llvm.round.v4f32(<4 x float>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x double> @llvm.round.v2f64(<2 x double>) nounwind readnone speculatable willreturn\n";
        os << "declare i32 @llvm.fptosi.sat.i32.f32(float) nounwind readnone speculatable willreturn\n";
        os << "declare i64 @llvm.fptosi.sat.i64.f32(float) nounwind readnone speculatable willreturn\n";
        os << "declare i32 @llvm.fptosi.sat.i32.f64(double) nounwind readnone speculatable willreturn\n";
        os << "declare i64 @llvm.fptosi.sat.i64.f64(double) nounwind readnone speculatable willreturn\n";
        os << "declare i32 @llvm.fptoui.sat.i32.f32(float) nounwind readnone speculatable willreturn\n";
        os << "declare i64 @llvm.fptoui.sat.i64.f32(float) nounwind readnone speculatable willreturn\n";
        os << "declare i32 @llvm.fptoui.sat.i32.f64(double) nounwind readnone speculatable willreturn\n";
        os << "declare i64 @llvm.fptoui.sat.i64.f64(double) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x i32> @llvm.fptosi.sat.v4i32.v4f32(<4 x float>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x i64> @llvm.fptosi.sat.v2i64.v2f64(<2 x double>) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x i32> @llvm.fptoui.sat.v4i32.v4f32(<4 x float>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x i64> @llvm.fptoui.sat.v2i64.v2f64(<2 x double>) nounwind readnone speculatable willreturn\n";
        os << "declare <16 x i8> @llvm.sadd.sat.v16i8(<16 x i8>, <16 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16>, <8 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32>, <4 x i32>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x i64> @llvm.sadd.sat.v2i64(<2 x i64>, <2 x i64>) nounwind readnone speculatable willreturn\n";
        os << "declare <16 x i8> @llvm.uadd.sat.v16i8(<16 x i8>, <16 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare <8 x i16> @llvm.uadd.sat.v8i16(<8 x i16>, <8 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x i32> @llvm.uadd.sat.v4i32(<4 x i32>, <4 x i32>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x i64> @llvm.uadd.sat.v2i64(<2 x i64>, <2 x i64>) nounwind readnone speculatable willreturn\n";
        os << "declare <16 x i8> @llvm.ssub.sat.v16i8(<16 x i8>, <16 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare <8 x i16> @llvm.ssub.sat.v8i16(<8 x i16>, <8 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32>, <4 x i32>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x i64> @llvm.ssub.sat.v2i64(<2 x i64>, <2 x i64>) nounwind readnone speculatable willreturn\n";
        os << "declare <16 x i8> @llvm.usub.sat.v16i8(<16 x i8>, <16 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare <8 x i16> @llvm.usub.sat.v8i16(<8 x i16>, <8 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x i32> @llvm.usub.sat.v4i32(<4 x i32>, <4 x i32>) nounwind readnone speculatable willreturn\n";
        os << "declare <2 x i64> @llvm.usub.sat.v2i64(<2 x i64>, <2 x i64>) nounwind readnone speculatable willreturn\n";
        os << "declare <16 x i8> @llvm.smax.v16i8(<16 x i8>, <16 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare <8 x i16> @llvm.smax.v8i16(<8 x i16>, <8 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x i32> @llvm.smax.v4i32(<4 x i32>, <4 x i32>) nounwind readnone speculatable willreturn\n";
        os << "declare <16 x i8> @llvm.smin.v16i8(<16 x i8>, <16 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare <8 x i16> @llvm.smin.v8i16(<8 x i16>, <8 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x i32> @llvm.smin.v4i32(<4 x i32>, <4 x i32>) nounwind readnone speculatable willreturn\n";
        os << "declare <16 x i8> @llvm.umax.v16i8(<16 x i8>, <16 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare <8 x i16> @llvm.umax.v8i16(<8 x i16>, <8 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x i32> @llvm.umax.v4i32(<4 x i32>, <4 x i32>) nounwind readnone speculatable willreturn\n";
        os << "declare <16 x i8> @llvm.umin.v16i8(<16 x i8>, <16 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare <8 x i16> @llvm.umin.v8i16(<8 x i16>, <8 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare <4 x i32> @llvm.umin.v4i32(<4 x i32>, <4 x i32>) nounwind readnone speculatable willreturn\n";
        os << "declare i8 @llvm.vector.reduce.add.v16i8(<16 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare i8 @llvm.vector.reduce.add.v8i8(<8 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare i16 @llvm.vector.reduce.add.v16i16(<16 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare i16 @llvm.vector.reduce.add.v8i16(<8 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare i16 @llvm.vector.reduce.add.v4i16(<4 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare i32 @llvm.vector.reduce.add.v8i32(<8 x i32>) nounwind readnone speculatable willreturn\n";
        os << "declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>) nounwind readnone speculatable willreturn\n";
        os << "declare i64 @llvm.vector.reduce.add.v4i64(<4 x i64>) nounwind readnone speculatable willreturn\n";
        os << "declare i8 @llvm.vector.reduce.smax.v16i8(<16 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare i8 @llvm.vector.reduce.smax.v8i8(<8 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare i16 @llvm.vector.reduce.smax.v8i16(<8 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare i16 @llvm.vector.reduce.smax.v4i16(<4 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare i32 @llvm.vector.reduce.smax.v4i32(<4 x i32>) nounwind readnone speculatable willreturn\n";
        os << "declare i8 @llvm.vector.reduce.umax.v16i8(<16 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare i8 @llvm.vector.reduce.umax.v8i8(<8 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare i16 @llvm.vector.reduce.umax.v8i16(<8 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare i16 @llvm.vector.reduce.umax.v4i16(<4 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare i32 @llvm.vector.reduce.umax.v4i32(<4 x i32>) nounwind readnone speculatable willreturn\n";
        os << "declare i8 @llvm.vector.reduce.smin.v16i8(<16 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare i8 @llvm.vector.reduce.smin.v8i8(<8 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare i16 @llvm.vector.reduce.smin.v8i16(<8 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare i16 @llvm.vector.reduce.smin.v4i16(<4 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare i32 @llvm.vector.reduce.smin.v4i32(<4 x i32>) nounwind readnone speculatable willreturn\n";
        os << "declare i8 @llvm.vector.reduce.umin.v16i8(<16 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare i8 @llvm.vector.reduce.umin.v8i8(<8 x i8>) nounwind readnone speculatable willreturn\n";
        os << "declare i16 @llvm.vector.reduce.umin.v8i16(<8 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare i16 @llvm.vector.reduce.umin.v4i16(<4 x i16>) nounwind readnone speculatable willreturn\n";
        os << "declare i32 @llvm.vector.reduce.umin.v4i32(<4 x i32>) nounwind readnone speculatable willreturn\n";

        if (uses_x86_pause) {
            os << "declare void @llvm.x86.sse2.pause() nounwind\n";
        }
        if (uses_x86_crc32) {
            os << "declare i32 @llvm.x86.sse42.crc32.32.8(i32, i8) nounwind readnone\n";
            os << "declare i32 @llvm.x86.sse42.crc32.32.16(i32, i16) nounwind readnone\n";
            os << "declare i32 @llvm.x86.sse42.crc32.32.32(i32, i32) nounwind readnone\n";
            os << "declare i64 @llvm.x86.sse42.crc32.64.64(i64, i64) nounwind readnone\n";
        }
        if (uses_aarch64_intrinsics) {
            os << "declare void @llvm.aarch64.hint(i32 immarg) nounwind\n";
            os << "declare i32 @llvm.aarch64.crc32b(i32, i32) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.crc32h(i32, i32) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.crc32w(i32, i32) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.crc32x(i32, i64) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.crc32cb(i32, i32) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.crc32ch(i32, i32) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.crc32cw(i32, i32) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.crc32cx(i32, i64) nounwind readnone\n";
            os << "declare <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8>, <16 x i8>) nounwind readnone\n";
            os << "declare <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8>, <16 x i8>) nounwind readnone\n";
            os << "declare <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8>) nounwind readnone\n";
            os << "declare <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8>) nounwind readnone\n";
            os << "declare <16 x i8> @llvm.aarch64.neon.pmull64(i64, i64) nounwind readnone\n";
            os << "declare <16 x i8> @llvm.aarch64.neon.tbl1.v16i8(<16 x i8>, <16 x i8>) nounwind readnone\n";
            os << "declare <16 x i8> @llvm.aarch64.neon.tbx1.v16i8(<16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.neon.saddv.i32.v16i8(<16 x i8>) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.neon.saddv.i32.v8i16(<8 x i16>) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.neon.saddv.i32.v4i32(<4 x i32>) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.neon.smaxv.i32.v16i8(<16 x i8>) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.neon.smaxv.i32.v8i16(<8 x i16>) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.neon.smaxv.i32.v4i32(<4 x i32>) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.neon.umaxv.i32.v16i8(<16 x i8>) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.neon.umaxv.i32.v8i16(<8 x i16>) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.neon.umaxv.i32.v4i32(<4 x i32>) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.neon.sminv.i32.v16i8(<16 x i8>) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.neon.sminv.i32.v8i16(<8 x i16>) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.neon.sminv.i32.v4i32(<4 x i32>) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.neon.uminv.i32.v16i8(<16 x i8>) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.neon.uminv.i32.v8i16(<8 x i16>) nounwind readnone\n";
            os << "declare i32 @llvm.aarch64.neon.uminv.i32.v4i32(<4 x i32>) nounwind readnone\n";
            os << "declare <8 x i16> @llvm.aarch64.neon.sqdmulh.v8i16(<8 x i16>, <8 x i16>) nounwind readnone\n";
            os << "declare <4 x i32> @llvm.aarch64.neon.sqdmulh.v4i32(<4 x i32>, <4 x i32>) nounwind readnone\n";
            os << "declare <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16>, <8 x i16>) nounwind readnone\n";
            os << "declare <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32>, <4 x i32>) nounwind readnone\n";
        }
        os << "\n";

        std::unordered_set<std::string> defined_names;
        for (const auto& fn : functions) {
            defined_names.insert(fn->name);
        }
        for (const auto& ext : external_guest_fns) {
            if (!defined_names.count(ext)) {
                os << "declare dso_local void @" << ext << "(ptr)\n";
            }
        }
        for (const auto& hle : external_hle_fns) {
            os << "declare dso_local void @" << hle << "(ptr)\n";
        }
        for (const auto& g : global_defs) {
            os << g << "\n";
        }
        os << "\n";

        for (const auto& fn : functions) {
            os << "define " << fn->linkage << " void @" << fn->name
               << "(ptr noalias noundef nonnull align 16 dereferenceable(1024) %c)"
               << (fn->is_inline_clone ? " alwaysinline nounwind" : " nounwind") << " {\n";
            for (const auto& bb : fn->blocks) {
                os << bb->name << ":\n";
                for (const auto& inst : bb->insts) {
                    if (inst.dead) continue;
                    if (!inst.text.empty()) {
                        os << "  " << inst.text << "\n";
                    }
                }
                if (!bb->has_terminator()) {
                    os << "  ret void\n";
                }
            }
            os << "}\n\n";
        }

        for (const auto& raw_fn : extra_raw_functions) {
            os << raw_fn << "\n\n";
        }

        if (needs_noalias_md) {
            os << "!0 = !{!\"ultratears.aa.domain\"}\n";
            os << "!1 = !{!\"ultratears.aa.cpu_state\", !0}\n";
            os << "!2 = !{!\"ultratears.aa.guest_mem\", !0}\n";
            os << "!3 = !{!1}\n";
            os << "!4 = !{!2}\n";
            os << "!5 = !{!\"ultratears.aa.guest_stack\", !0}\n";
            os << "!6 = !{!5}\n";
        }
        return os.str();
    }
};

inline BasicBlock* Function::create_block(const std::string& label, uint64_t pc) {
    auto it = block_map.find(label);
    if (it != block_map.end()) return it->second;
    auto bb = std::make_unique<BasicBlock>();
    bb->name = label;
    bb->start_pc = pc;
    if (parent_mod && llvm_fn) {
        bb->llvm_bb = llvm::BasicBlock::Create(parent_mod->context(), label, llvm_fn);
    }
    BasicBlock* ptr = bb.get();
    block_map[label] = ptr;
    blocks.push_back(std::move(bb));
    return ptr;
}

// -----------------------------------------------------------------------------
// LLVM Pass Infrastructure (`PassInfoMixin` pattern)
// -----------------------------------------------------------------------------
template <typename Derived>
struct PassInfoMixin {
    static const char* name() { return Derived::static_name(); }
};

struct ModulePass {
    virtual ~ModulePass() = default;
    virtual const char* pass_name() const = 0;
    virtual bool run(Module& M) = 0;
};

class ModulePassManager {
public:
    void addPass(std::unique_ptr<ModulePass> pass) {
        passes_.push_back(std::move(pass));
    }

    bool run(Module& M) {
        bool changed = false;
        for (auto& pass : passes_) {
            changed |= pass->run(M);
        }
        return changed;
    }

private:
    std::vector<std::unique_ptr<ModulePass>> passes_;
};

// -----------------------------------------------------------------------------
// Thin IRBuilder API on top of llvm::IRBuilder<llvm::NoFolder>
// -----------------------------------------------------------------------------
class IRBuilder {
public:
    using RawBuilder = llvm::IRBuilder<llvm::NoFolder>;

    IRBuilder(Function* fn, BasicBlock* bb = nullptr)
        : fn_(fn),
          bb_(bb),
          builder_(fn->parent_mod ? fn->parent_mod->context() : get_fallback_context()) {
        if (bb_ && bb_->llvm_bb) {
            builder_.SetInsertPoint(bb_->llvm_bb);
        }
    }

    void set_insert_point(BasicBlock* bb) {
        bb_ = bb;
        if (bb_ && bb_->llvm_bb) {
            builder_.SetInsertPoint(bb_->llvm_bb);
        }
    }

    BasicBlock* insert_block() const { return bb_; }
    Function* function() const { return fn_; }
    RawBuilder& raw() { return builder_; }
    RawBuilder* operator->() { return &builder_; }
    llvm::LLVMContext& context() const { return builder_.getContext(); }

    llvm::Type* llvm_type(TypeKind t) const {
        return to_llvm_type(context(), t);
    }

    llvm::Value* resolve_llvm_value(const Value& v) {
        if (v.llvm_val) return v.llvm_val;
        auto& ctx = context();
        llvm::Type* ty = to_llvm_type(ctx, v.type);
        if (v.repr == "%c") {
            if (fn_ && fn_->llvm_fn && !fn_->llvm_fn->arg_empty()) {
                return fn_->llvm_fn->getArg(0);
            }
        }
        if (v.is_const) {
            if (v.repr == "zeroinitializer") {
                return llvm::Constant::getNullValue(ty);
            }
            if (v.type == TypeKind::Pointer && !v.repr.empty() && v.repr[0] == '@') {
                std::string gname = v.repr.substr(1);
                if (fn_ && fn_->parent_mod) {
                    if (auto* gv = fn_->parent_mod->llvm_mod->getNamedValue(gname)) {
                        return gv;
                    }
                    auto* g = new llvm::GlobalVariable(
                        *fn_->parent_mod->llvm_mod, llvm::Type::getInt8Ty(ctx), false,
                        llvm::GlobalValue::ExternalLinkage, nullptr, gname);
                    return g;
                }
            }
            if (v.type == TypeKind::Int128 && v.repr.rfind("u0x", 0) == 0) {
                llvm::APInt ap(128, v.repr.substr(3), 16);
                return llvm::ConstantInt::get(ctx, ap);
            }
            if (v.type == TypeKind::Float || v.type == TypeKind::Double) {
                return llvm::ConstantFP::get(ty, std::stod(v.repr));
            }
            return llvm::ConstantInt::get(ty, v.const_u64, false);
        }
        if (fn_) {
            auto it = fn_->ssa_values.find(v.repr);
            if (it != fn_->ssa_values.end() && it->second->getType() == ty) {
                return it->second;
            }
            auto arg = std::make_unique<llvm::Argument>(ty, strip_ssa_pct(v.repr));
            llvm::Value* ptr = arg.get();
            fn_->synthetic_ssa_vals.push_back(std::move(arg));
            fn_->ssa_values[v.repr] = ptr;
            return ptr;
        }
        return llvm::UndefValue::get(ty);
    }

    static std::string format_insts(std::initializer_list<const llvm::Value*> vals) {
        std::ostringstream os;
        bool first = true;
        for (const llvm::Value* v : vals) {
            if (!v) continue;
            if (!first) os << "\n  ";
            os << format_llvm_inst(v);
            first = false;
        }
        return os.str();
    }

    void emit_raw(std::string text, OpTag tag = OpTag::RawIR) {
        Instruction inst;
        inst.tag = tag;
        inst.text = std::move(text);
        bb_->insts.push_back(std::move(inst));
    }

    void emit_inst(Instruction inst) {
        bb_->insts.push_back(std::move(inst));
    }

    void emit_guest_marker(uint64_t pc, uint32_t word) {
        Instruction inst;
        inst.tag = OpTag::GuestInsnMarker;
        inst.guest_pc = pc;
        inst.insn_word = word;
        char buf[80];
        std::snprintf(buf, sizeof(buf), "; @pc 0x%" PRIx64 " word 0x%08x", pc, word);
        inst.text = buf;
        bb_->insts.push_back(std::move(inst));
    }

    Value binop(const char* op, TypeKind t, const Value& a, const Value& b, const char* prefix = "t") {
        std::string r = fn_->next_reg(prefix);
        llvm::Instruction::BinaryOps opc = llvm::Instruction::Add;
        if (std::strcmp(op, "add") == 0) opc = llvm::Instruction::Add;
        else if (std::strcmp(op, "sub") == 0) opc = llvm::Instruction::Sub;
        else if (std::strcmp(op, "mul") == 0) opc = llvm::Instruction::Mul;
        else if (std::strcmp(op, "udiv") == 0) opc = llvm::Instruction::UDiv;
        else if (std::strcmp(op, "sdiv") == 0) opc = llvm::Instruction::SDiv;
        else if (std::strcmp(op, "urem") == 0) opc = llvm::Instruction::URem;
        else if (std::strcmp(op, "srem") == 0) opc = llvm::Instruction::SRem;
        else if (std::strcmp(op, "shl") == 0) opc = llvm::Instruction::Shl;
        else if (std::strcmp(op, "lshr") == 0) opc = llvm::Instruction::LShr;
        else if (std::strcmp(op, "ashr") == 0) opc = llvm::Instruction::AShr;
        else if (std::strcmp(op, "and") == 0) opc = llvm::Instruction::And;
        else if (std::strcmp(op, "or") == 0) opc = llvm::Instruction::Or;
        else if (std::strcmp(op, "xor") == 0) opc = llvm::Instruction::Xor;
        else if (std::strcmp(op, "fadd") == 0) opc = llvm::Instruction::FAdd;
        else if (std::strcmp(op, "fsub") == 0) opc = llvm::Instruction::FSub;
        else if (std::strcmp(op, "fmul") == 0) opc = llvm::Instruction::FMul;
        else if (std::strcmp(op, "fdiv") == 0) opc = llvm::Instruction::FDiv;

        llvm::Value* lhs = resolve_llvm_value(a);
        llvm::Value* rhs = resolve_llvm_value(b);
        llvm::Value* res = builder_.CreateBinOp(opc, lhs, rhs, strip_ssa_pct(r));
        fn_->register_ssa_value(r, res);

        Instruction inst;
        inst.tag = OpTag::RawIR;
        inst.dst_reg = r;
        inst.dst_type = t;
        inst.op0 = a;
        inst.op1 = b;
        inst.llvm_inst = llvm::dyn_cast<llvm::Instruction>(res);
        inst.text = format_llvm_inst(res);
        bb_->insts.push_back(std::move(inst));
        return Value::reg(t, r, res);
    }

    Value CreateAdd(TypeKind t, const Value& a, const Value& b) { return binop("add", t, a, b); }
    Value CreateSub(TypeKind t, const Value& a, const Value& b) { return binop("sub", t, a, b); }
    Value CreateMul(TypeKind t, const Value& a, const Value& b) { return binop("mul", t, a, b); }
    Value CreateUDiv(TypeKind t, const Value& a, const Value& b) { return binop("udiv", t, a, b); }
    Value CreateSDiv(TypeKind t, const Value& a, const Value& b) { return binop("sdiv", t, a, b); }
    Value CreateShl(TypeKind t, const Value& a, const Value& b) { return binop("shl", t, a, b); }
    Value CreateLShr(TypeKind t, const Value& a, const Value& b) { return binop("lshr", t, a, b); }
    Value CreateAShr(TypeKind t, const Value& a, const Value& b) { return binop("ashr", t, a, b); }
    Value CreateAnd(TypeKind t, const Value& a, const Value& b) { return binop("and", t, a, b); }
    Value CreateOr(TypeKind t, const Value& a, const Value& b) { return binop("or", t, a, b); }
    Value CreateXor(TypeKind t, const Value& a, const Value& b) { return binop("xor", t, a, b); }

    static llvm::CmpInst::Predicate parse_icmp_pred(const char* pred) {
        if (std::strcmp(pred, "eq") == 0) return llvm::CmpInst::ICMP_EQ;
        if (std::strcmp(pred, "ne") == 0) return llvm::CmpInst::ICMP_NE;
        if (std::strcmp(pred, "ugt") == 0) return llvm::CmpInst::ICMP_UGT;
        if (std::strcmp(pred, "uge") == 0) return llvm::CmpInst::ICMP_UGE;
        if (std::strcmp(pred, "ult") == 0) return llvm::CmpInst::ICMP_ULT;
        if (std::strcmp(pred, "ule") == 0) return llvm::CmpInst::ICMP_ULE;
        if (std::strcmp(pred, "sgt") == 0) return llvm::CmpInst::ICMP_SGT;
        if (std::strcmp(pred, "sge") == 0) return llvm::CmpInst::ICMP_SGE;
        if (std::strcmp(pred, "slt") == 0) return llvm::CmpInst::ICMP_SLT;
        if (std::strcmp(pred, "sle") == 0) return llvm::CmpInst::ICMP_SLE;
        return llvm::CmpInst::ICMP_EQ;
    }

    Value icmp(const char* pred, TypeKind t, const Value& a, const Value& b, const char* prefix = "cmp") {
        std::string r = fn_->next_reg(prefix);
        llvm::Value* lhs = resolve_llvm_value(a);
        llvm::Value* rhs = resolve_llvm_value(b);
        llvm::Value* res = builder_.CreateICmp(parse_icmp_pred(pred), lhs, rhs, strip_ssa_pct(r));
        fn_->register_ssa_value(r, res);

        Instruction inst;
        inst.tag = OpTag::RawIR;
        inst.dst_reg = r;
        inst.dst_type = TypeKind::Int1;
        inst.op0 = a;
        inst.op1 = b;
        inst.llvm_inst = llvm::dyn_cast<llvm::Instruction>(res);
        inst.text = format_llvm_inst(res);
        bb_->insts.push_back(std::move(inst));
        return Value::reg(TypeKind::Int1, r, res);
    }

    Value CreateICmpEQ(TypeKind t, const Value& a, const Value& b, const char* prefix = "cmp") {
        return icmp("eq", t, a, b, prefix);
    }
    Value CreateICmpNE(TypeKind t, const Value& a, const Value& b, const char* prefix = "cmp") {
        return icmp("ne", t, a, b, prefix);
    }

    // Vector ICmp + SExt to all-ones/all-zeros mask vector (e.g. CMEQ, CMGT, CMGE, CMHI, CMHS, CMTST)
    Value CreateVecICmpSExt(const char* pred, TypeKind vt, const Value& a, const Value& b) {
        std::string cmp_r = fn_->next_reg("vcmp");
        std::string sx_r = fn_->next_reg("vsx");
        llvm::Value* lhs = resolve_llvm_value(a);
        llvm::Value* rhs = resolve_llvm_value(b);
        llvm::Value* cmp_v = builder_.CreateICmp(parse_icmp_pred(pred), lhs, rhs, strip_ssa_pct(cmp_r));
        llvm::Value* sx_v = builder_.CreateSExt(cmp_v, llvm_type(vt), strip_ssa_pct(sx_r));
        fn_->register_ssa_value(cmp_r, cmp_v);
        fn_->register_ssa_value(sx_r, sx_v);

        Instruction inst1;
        inst1.tag = OpTag::RawIR;
        inst1.dst_reg = cmp_r;
        inst1.llvm_inst = llvm::dyn_cast<llvm::Instruction>(cmp_v);
        inst1.text = format_llvm_inst(cmp_v);
        bb_->insts.push_back(std::move(inst1));

        Instruction inst2;
        inst2.tag = OpTag::RawIR;
        inst2.dst_reg = sx_r;
        inst2.dst_type = vt;
        inst2.llvm_inst = llvm::dyn_cast<llvm::Instruction>(sx_v);
        inst2.text = format_llvm_inst(sx_v);
        bb_->insts.push_back(std::move(inst2));
        return Value::reg(vt, sx_r, sx_v);
    }

    // Vector ABS via ICmpSLT < 0 + Select(neg, pos)
    Value CreateVecAbs(TypeKind vt, const Value& a, const Value& neg) {
        std::string cmp_r = fn_->next_reg("vcmp");
        std::string sel_r = fn_->next_reg("vsel");
        llvm::Value* av = resolve_llvm_value(a);
        llvm::Value* nv = resolve_llvm_value(neg);
        llvm::Value* zv = llvm::Constant::getNullValue(llvm_type(vt));
        llvm::Value* cmp_v = builder_.CreateICmpSLT(av, zv, strip_ssa_pct(cmp_r));
        llvm::Value* sel_v = builder_.CreateSelect(cmp_v, nv, av, strip_ssa_pct(sel_r));
        fn_->register_ssa_value(cmp_r, cmp_v);
        fn_->register_ssa_value(sel_r, sel_v);

        Instruction inst1;
        inst1.tag = OpTag::RawIR;
        inst1.dst_reg = cmp_r;
        inst1.llvm_inst = llvm::dyn_cast<llvm::Instruction>(cmp_v);
        inst1.text = format_llvm_inst(cmp_v);
        bb_->insts.push_back(std::move(inst1));

        Instruction inst2;
        inst2.tag = OpTag::RawIR;
        inst2.dst_reg = sel_r;
        inst2.dst_type = vt;
        inst2.llvm_inst = llvm::dyn_cast<llvm::Instruction>(sel_v);
        inst2.text = format_llvm_inst(sel_v);
        bb_->insts.push_back(std::move(inst2));
        return Value::reg(vt, sel_r, sel_v);
    }

    Value select(TypeKind t, const Value& cond, const Value& tv, const Value& fv, const char* prefix = "sel") {
        std::string r = fn_->next_reg(prefix);
        llvm::Value* cv = resolve_llvm_value(cond);
        llvm::Value* tv_v = resolve_llvm_value(tv);
        llvm::Value* fv_v = resolve_llvm_value(fv);
        llvm::Value* res = builder_.CreateSelect(cv, tv_v, fv_v, strip_ssa_pct(r));
        fn_->register_ssa_value(r, res);

        Instruction inst;
        inst.tag = OpTag::RawIR;
        inst.dst_reg = r;
        inst.dst_type = t;
        inst.op0 = cond;
        inst.op1 = tv;
        inst.op2 = fv;
        inst.llvm_inst = llvm::dyn_cast<llvm::Instruction>(res);
        inst.text = format_llvm_inst(res);
        bb_->insts.push_back(std::move(inst));
        return Value::reg(t, r, res);
    }

    Value cast(const char* op, const Value& v, TypeKind dst_t, const char* prefix = "c") {
        if (v.type == dst_t) return v;
        std::string r = fn_->next_reg(prefix);
        llvm::Instruction::CastOps opc = llvm::Instruction::BitCast;
        if (std::strcmp(op, "zext") == 0) opc = llvm::Instruction::ZExt;
        else if (std::strcmp(op, "sext") == 0) opc = llvm::Instruction::SExt;
        else if (std::strcmp(op, "trunc") == 0) opc = llvm::Instruction::Trunc;
        else if (std::strcmp(op, "bitcast") == 0) opc = llvm::Instruction::BitCast;
        else if (std::strcmp(op, "inttoptr") == 0) opc = llvm::Instruction::IntToPtr;
        else if (std::strcmp(op, "ptrtoint") == 0) opc = llvm::Instruction::PtrToInt;
        else if (std::strcmp(op, "fpext") == 0) opc = llvm::Instruction::FPExt;
        else if (std::strcmp(op, "fptrunc") == 0) opc = llvm::Instruction::FPTrunc;
        else if (std::strcmp(op, "sitofp") == 0) opc = llvm::Instruction::SIToFP;
        else if (std::strcmp(op, "uitofp") == 0) opc = llvm::Instruction::UIToFP;
        else if (std::strcmp(op, "fptosi") == 0) opc = llvm::Instruction::FPToSI;
        else if (std::strcmp(op, "fptoui") == 0) opc = llvm::Instruction::FPToUI;

        llvm::Value* src = resolve_llvm_value(v);
        llvm::Value* res = builder_.CreateCast(opc, src, llvm_type(dst_t), strip_ssa_pct(r));
        fn_->register_ssa_value(r, res);

        Instruction inst;
        inst.tag = OpTag::RawIR;
        inst.dst_reg = r;
        inst.dst_type = dst_t;
        inst.op0 = v;
        inst.llvm_inst = llvm::dyn_cast<llvm::Instruction>(res);
        inst.text = format_llvm_inst(res);
        bb_->insts.push_back(std::move(inst));
        return Value::reg(dst_t, r, res);
    }

    Value zext_or_trunc(const Value& v, TypeKind dst_t) {
        if (v.type == dst_t) return v;
        return cast(static_cast<int>(dst_t) > static_cast<int>(v.type) ? "zext" : "trunc", v, dst_t);
    }

    Value sext(const Value& v, TypeKind dst_t) {
        if (v.type == dst_t) return v;
        return cast("sext", v, dst_t);
    }

    Value bitcast(const Value& v, TypeKind dst_t) {
        if (v.type == dst_t) return v;
        return cast("bitcast", v, dst_t);
    }

    Value CreateIntToPtr(const Value& addr_i64, const char* prefix = "atp") {
        return cast("inttoptr", addr_i64, TypeKind::Pointer, prefix);
    }

    Value CreateExtractElement(TypeKind vt, TypeKind et, const Value& vec, unsigned idx, const char* prefix = "el") {
        (void)vt;
        std::string r = fn_->next_reg(prefix);
        llvm::Value* vec_v = resolve_llvm_value(vec);
        llvm::Value* idx_v = builder_.getInt32(idx);
        llvm::Value* res = builder_.CreateExtractElement(vec_v, idx_v, strip_ssa_pct(r));
        fn_->register_ssa_value(r, res);

        Instruction inst;
        inst.tag = OpTag::RawIR;
        inst.dst_reg = r;
        inst.dst_type = et;
        inst.op0 = vec;
        inst.llvm_inst = llvm::dyn_cast<llvm::Instruction>(res);
        inst.text = format_llvm_inst(res);
        bb_->insts.push_back(std::move(inst));
        return Value::reg(et, r, res);
    }

    Value CreateInsertElement(TypeKind vt, const Value& vec, const Value& elt, unsigned idx, const char* prefix = "ins") {
        std::string r = fn_->next_reg(prefix);
        llvm::Value* vec_v = resolve_llvm_value(vec);
        llvm::Value* elt_v = resolve_llvm_value(elt);
        llvm::Value* idx_v = builder_.getInt32(idx);
        llvm::Value* res = builder_.CreateInsertElement(vec_v, elt_v, idx_v, strip_ssa_pct(r));
        fn_->register_ssa_value(r, res);

        Instruction inst;
        inst.tag = OpTag::RawIR;
        inst.dst_reg = r;
        inst.dst_type = vt;
        inst.op0 = vec;
        inst.op1 = elt;
        inst.llvm_inst = llvm::dyn_cast<llvm::Instruction>(res);
        inst.text = format_llvm_inst(res);
        bb_->insts.push_back(std::move(inst));
        return Value::reg(vt, r, res);
    }

    Value CreateShuffleVector(TypeKind vt, const Value& a, const Value& b, const std::vector<int>& mask, const char* prefix = "shuf") {
        std::string r = fn_->next_reg(prefix);
        llvm::Value* av = resolve_llvm_value(a);
        llvm::Value* bv = resolve_llvm_value(b);
        llvm::Value* res = builder_.CreateShuffleVector(av, bv, mask, strip_ssa_pct(r));
        fn_->register_ssa_value(r, res);

        Instruction inst;
        inst.tag = OpTag::RawIR;
        inst.dst_reg = r;
        inst.dst_type = vt;
        inst.op0 = a;
        inst.op1 = b;
        inst.llvm_inst = llvm::dyn_cast<llvm::Instruction>(res);
        inst.text = format_llvm_inst(res);
        bb_->insts.push_back(std::move(inst));
        return Value::reg(vt, r, res);
    }

    Value CreateVolatileLoadGlobalI32(const std::string& global_name, unsigned align, const char* prefix = "prof") {
        std::string r = fn_->next_reg(prefix);
        llvm::Value* gvar = resolve_llvm_value(Value::global_sym(global_name));
        llvm::LoadInst* ld = builder_.CreateAlignedLoad(builder_.getInt32Ty(), gvar, llvm::Align(align), true, strip_ssa_pct(r));
        fn_->register_ssa_value(r, ld);

        Instruction inst;
        inst.tag = OpTag::RawIR;
        inst.dst_reg = r;
        inst.dst_type = TypeKind::Int32;
        inst.llvm_inst = ld;
        inst.text = format_llvm_inst(ld);
        bb_->insts.push_back(std::move(inst));
        return Value::reg(TypeKind::Int32, r, ld);
    }

    Value CreateAtomicLoad(TypeKind mt, const Value& ptr_val, unsigned align,
                           llvm::AtomicOrdering ordering, const char* prefix = "ldx",
                           bool guest_noalias_md = true) {
        std::string r = fn_->next_reg(prefix);
        llvm::Value* pv = resolve_llvm_value(ptr_val);
        llvm::LoadInst* ld = builder_.CreateAlignedLoad(llvm_type(mt), pv, llvm::Align(align), strip_ssa_pct(r));
        ld->setAtomic(ordering);
        fn_->register_ssa_value(r, ld);

        Instruction inst;
        inst.tag = OpTag::RawIR;
        inst.dst_reg = r;
        inst.dst_type = mt;
        inst.op0 = ptr_val;
        inst.llvm_inst = ld;
        inst.text = format_llvm_inst(ld) + (guest_noalias_md ? ", !alias.scope !4, !noalias !3" : "");
        bb_->insts.push_back(std::move(inst));
        return Value::reg(mt, r, ld);
    }

    void CreateAtomicStore(TypeKind mt, const Value& val, const Value& ptr_val, unsigned align,
                           llvm::AtomicOrdering ordering, bool guest_noalias_md = true) {
        (void)mt;
        llvm::Value* vv = resolve_llvm_value(val);
        llvm::Value* pv = resolve_llvm_value(ptr_val);
        llvm::StoreInst* st = builder_.CreateAlignedStore(vv, pv, llvm::Align(align));
        st->setAtomic(ordering);

        Instruction inst;
        inst.tag = OpTag::RawIR;
        inst.op0 = ptr_val;
        inst.op1 = val;
        inst.llvm_inst = st;
        inst.text = format_llvm_inst(st) + (guest_noalias_md ? ", !alias.scope !4, !noalias !3" : "");
        bb_->insts.push_back(std::move(inst));
    }

    // Emit atomic cmpxchg + extractvalue { mt, i1 } %cx, 1 and return the i1 success value
    Value CreateAtomicCmpXchgOk(TypeKind mt, const Value& ptr_val, const Value& exp_val,
                                const Value& new_val, unsigned align) {
        std::string cx_r = fn_->next_reg("cx");
        std::string ok_r = fn_->next_reg("cxok");
        llvm::Value* pv = resolve_llvm_value(ptr_val);
        llvm::Value* ev = resolve_llvm_value(exp_val);
        llvm::Value* nv = resolve_llvm_value(new_val);
        auto* cx = builder_.CreateAtomicCmpXchg(
            pv, ev, nv, llvm::MaybeAlign(align),
            llvm::AtomicOrdering::SequentiallyConsistent,
            llvm::AtomicOrdering::SequentiallyConsistent);
        cx->setName(strip_ssa_pct(cx_r));
        auto* ok = builder_.CreateExtractValue(cx, {1}, strip_ssa_pct(ok_r));
        fn_->register_ssa_value(cx_r, cx);
        fn_->register_ssa_value(ok_r, ok);

        Instruction inst1;
        inst1.tag = OpTag::RawIR;
        inst1.dst_reg = cx_r;
        inst1.llvm_inst = cx;
        inst1.text = format_llvm_inst(cx);
        bb_->insts.push_back(std::move(inst1));

        Instruction inst2;
        inst2.tag = OpTag::RawIR;
        inst2.dst_reg = ok_r;
        inst2.dst_type = TypeKind::Int1;
        inst2.llvm_inst = llvm::dyn_cast<llvm::Instruction>(ok);
        inst2.text = format_llvm_inst(ok);
        bb_->insts.push_back(std::move(inst2));
        return Value::reg(TypeKind::Int1, ok_r, ok);
    }

    Value CreateCall(TypeKind ret_t, const std::string& callee_name,
                     const std::vector<Value>& args, const char* prefix = "call",
                     OpTag tag = OpTag::RawIR) {
        std::string clean_name = (!callee_name.empty() && callee_name[0] == '@')
                               ? callee_name.substr(1)
                               : callee_name;
        std::vector<llvm::Type*> param_tys;
        std::vector<llvm::Value*> arg_vals;
        param_tys.reserve(args.size());
        arg_vals.reserve(args.size());
        for (const auto& a : args) {
            param_tys.push_back(llvm_type(a.type));
            arg_vals.push_back(resolve_llvm_value(a));
        }
        llvm::FunctionType* fty = llvm::FunctionType::get(llvm_type(ret_t), param_tys, false);
        llvm::FunctionCallee callee = fn_->parent_mod->llvm_mod->getOrInsertFunction(clean_name, fty);
        std::string r = (ret_t == TypeKind::Void) ? "" : fn_->next_reg(prefix);
        llvm::CallInst* call = builder_.CreateCall(callee, arg_vals, strip_ssa_pct(r));
        if (!r.empty()) {
            fn_->register_ssa_value(r, call);
        }

        Instruction inst;
        inst.tag = tag;
        inst.dst_reg = r;
        inst.dst_type = ret_t;
        inst.llvm_inst = call;
        inst.text = format_llvm_inst(call);
        bb_->insts.push_back(std::move(inst));
        return Value::reg(ret_t, r, call);
    }

    void CreateVoidCall(const std::string& callee_name, const std::vector<Value>& args,
                        OpTag tag = OpTag::RawIR) {
        CreateCall(TypeKind::Void, callee_name, args, "", tag);
    }

    void CreateBr(const std::string& target_label, uint64_t target_pc = 0, OpTag tag = OpTag::UncondBranch) {
        BasicBlock* target_bb = fn_->create_block(target_label, target_pc);
        llvm::BranchInst* br = builder_.CreateBr(target_bb->llvm_bb);
        Instruction inst;
        inst.tag = tag;
        inst.target_label = target_label;
        inst.branch_target = target_pc;
        inst.llvm_inst = br;
        inst.text = format_llvm_inst(br);
        bb_->insts.push_back(std::move(inst));
    }

    Value CreateCondEval(int cond_code, const std::string& out_reg, const Value& nzcv,
                         std::string rendered_ir, llvm::Instruction* llvm_i = nullptr) {
        Instruction inst;
        inst.tag = OpTag::RawIR;
        inst.cond_code = cond_code;
        inst.dst_reg = out_reg;
        inst.dst_type = TypeKind::Int1;
        inst.op0 = nzcv;
        inst.llvm_inst = llvm_i;
        inst.text = std::move(rendered_ir);
        bb_->insts.push_back(std::move(inst));
        return Value::reg(TypeKind::Int1, out_reg, llvm_i);
    }

    void CreateGuestCondBr(const Value& cond_i1, const std::string& true_label, const std::string& false_label,
                           int cond_code, uint64_t pc, uint64_t branch_target_pc) {
        BasicBlock* t_bb = fn_->create_block(true_label, branch_target_pc);
        BasicBlock* f_bb = fn_->create_block(false_label, pc + 4);
        llvm::Value* cv = resolve_llvm_value(cond_i1);
        llvm::BranchInst* br = builder_.CreateCondBr(cv, t_bb->llvm_bb, f_bb->llvm_bb);
        Instruction inst;
        inst.tag = OpTag::CondBranch;
        inst.guest_pc = pc;
        inst.cond_code = cond_code;
        inst.op0 = cond_i1;
        inst.branch_target = branch_target_pc;
        inst.target_label = true_label;
        inst.false_label = false_label;
        inst.llvm_inst = br;
        inst.text = format_llvm_inst(br);
        bb_->insts.push_back(std::move(inst));
    }

    void CreateCondBr(const Value& cond_i1, const std::string& true_label, const std::string& false_label,
                      int cond_code = -1, uint64_t pc = 0, OpTag tag = OpTag::CondBranch) {
        BasicBlock* t_bb = fn_->create_block(true_label);
        BasicBlock* f_bb = fn_->create_block(false_label);
        llvm::Value* cv = resolve_llvm_value(cond_i1);
        llvm::BranchInst* br = builder_.CreateCondBr(cv, t_bb->llvm_bb, f_bb->llvm_bb);
        Instruction inst;
        inst.tag = tag;
        inst.guest_pc = pc;
        inst.cond_code = cond_code;
        inst.op0 = cond_i1;
        inst.target_label = true_label;
        inst.false_label = false_label;
        inst.llvm_inst = br;
        inst.text = format_llvm_inst(br);
        bb_->insts.push_back(std::move(inst));
    }

    void CreateRetVoid(uint64_t pc = 0) {
        llvm::ReturnInst* ret = builder_.CreateRetVoid();
        Instruction inst;
        inst.tag = OpTag::Return;
        inst.guest_pc = pc;
        inst.llvm_inst = ret;
        inst.text = format_llvm_inst(ret);
        bb_->insts.push_back(std::move(inst));
    }

    void CreateTailCallRet(const std::string& callee_name, llvm::CallInst::TailCallKind tck,
                           OpTag tag = OpTag::Return, uint64_t target_pc = 0) {
        std::string clean_name = (!callee_name.empty() && callee_name[0] == '@')
                               ? callee_name.substr(1)
                               : callee_name;
        llvm::Type* ptr_ty = llvm_type(TypeKind::Pointer);
        llvm::FunctionType* fty = llvm::FunctionType::get(llvm_type(TypeKind::Void), {ptr_ty}, false);
        llvm::FunctionCallee callee = fn_->parent_mod->llvm_mod->getOrInsertFunction(clean_name, fty);
        llvm::CallInst* call = builder_.CreateCall(callee, {resolve_llvm_value(Value::cpu_ptr())});
        call->setTailCallKind(tck);
        llvm::ReturnInst* ret = builder_.CreateRetVoid();

        Instruction inst;
        inst.tag = tag;
        inst.branch_target = target_pc;
        inst.llvm_inst = call;
        inst.text = format_insts({call, ret});
        bb_->insts.push_back(std::move(inst));
    }

    void CreateDirectCall(const std::string& callee_name, uint64_t pc, uint64_t target_pc) {
        std::string clean_name = (!callee_name.empty() && callee_name[0] == '@')
                               ? callee_name.substr(1)
                               : callee_name;
        llvm::Type* ptr_ty = llvm_type(TypeKind::Pointer);
        llvm::FunctionType* fty = llvm::FunctionType::get(llvm_type(TypeKind::Void), {ptr_ty}, false);
        llvm::FunctionCallee callee = fn_->parent_mod->llvm_mod->getOrInsertFunction(clean_name, fty);
        llvm::CallInst* call = builder_.CreateCall(callee, {resolve_llvm_value(Value::cpu_ptr())});

        Instruction inst;
        inst.tag = OpTag::DirectCall;
        inst.guest_pc = pc;
        inst.branch_target = target_pc;
        inst.llvm_inst = call;
        inst.text = format_llvm_inst(call);
        bb_->insts.push_back(std::move(inst));
    }

    void CreateIndirectCall(const Value& target, uint64_t pc) {
        llvm::Type* ptr_ty = llvm_type(TypeKind::Pointer);
        llvm::FunctionType* fty = llvm::FunctionType::get(llvm_type(TypeKind::Void), {ptr_ty}, false);
        llvm::FunctionCallee callee = fn_->parent_mod->llvm_mod->getOrInsertFunction("rt_dispatch", fty);
        llvm::CallInst* call = builder_.CreateCall(callee, {resolve_llvm_value(Value::cpu_ptr())});

        Instruction inst;
        inst.tag = OpTag::IndirectCall;
        inst.guest_pc = pc;
        inst.op0 = target;
        inst.llvm_inst = call;
        inst.text = format_llvm_inst(call);
        bb_->insts.push_back(std::move(inst));
    }

    void CreateIndirectJump(const Value& target, uint64_t pc) {
        Instruction inst;
        inst.tag = OpTag::IndirectJump;
        inst.guest_pc = pc;
        inst.op0 = target;
        inst.text = "; indirect jump br " + target.repr;
        bb_->insts.push_back(std::move(inst));
    }

    void CreateHorizonUdf(uint64_t pc) {
        llvm::Type* ptr_ty = llvm_type(TypeKind::Pointer);
        llvm::Type* i64_ty = llvm_type(TypeKind::Int64);
        llvm::FunctionType* fty = llvm::FunctionType::get(llvm_type(TypeKind::Void), {ptr_ty, i64_ty}, false);
        llvm::FunctionCallee callee = fn_->parent_mod->llvm_mod->getOrInsertFunction("rt_udf", fty);
        llvm::CallInst* call = builder_.CreateCall(callee, {resolve_llvm_value(Value::cpu_ptr()), builder_.getInt64(pc)});
        llvm::UnreachableInst* unr = builder_.CreateUnreachable();

        Instruction inst;
        inst.tag = OpTag::HorizonUdf;
        inst.guest_pc = pc;
        inst.llvm_inst = call;
        inst.text = format_insts({call, unr});
        bb_->insts.push_back(std::move(inst));
    }

    void CreateHorizonSvc(uint32_t imm, uint64_t pc) {
        Instruction inst;
        inst.tag = OpTag::HorizonSvc;
        inst.guest_pc = pc;
        inst.sys_key = imm;
        inst.text = "; horizon.svc #" + std::to_string(imm);
        bb_->insts.push_back(std::move(inst));
    }

    void CreateHorizonBrk(uint32_t imm, uint64_t pc) {
        Instruction inst;
        inst.tag = OpTag::HorizonBrk;
        inst.guest_pc = pc;
        inst.sys_key = imm;
        inst.text = "; horizon.brk #" + std::to_string(imm);
        bb_->insts.push_back(std::move(inst));
    }

    void CreateHorizonBarrier(uint32_t op2, uint64_t pc) {
        Instruction inst;
        inst.tag = OpTag::HorizonBarrier;
        inst.guest_pc = pc;
        inst.sys_key = op2;
        inst.text = "; horizon.barrier op2=" + std::to_string(op2);
        bb_->insts.push_back(std::move(inst));
    }

    void CreateHorizonDcZva(const Value& addr, uint64_t pc) {
        Instruction inst;
        inst.tag = OpTag::HorizonDcZva;
        inst.guest_pc = pc;
        inst.op0 = addr;
        inst.text = "; horizon.dc_zva " + addr.repr;
        bb_->insts.push_back(std::move(inst));
    }

    void CreateHorizonSysRegRead(int rt, uint32_t key, uint64_t pc) {
        Instruction inst;
        inst.tag = OpTag::HorizonSysRegRead;
        inst.guest_pc = pc;
        inst.arch_reg = rt;
        inst.sys_key = key;
        inst.text = "; horizon.mrs key=" + std::to_string(key);
        bb_->insts.push_back(std::move(inst));
    }

    void CreateHorizonSysRegWrite(const Value& val, uint32_t key, uint64_t pc) {
        Instruction inst;
        inst.tag = OpTag::HorizonSysRegWrite;
        inst.guest_pc = pc;
        inst.sys_key = key;
        inst.op0 = val;
        inst.text = "; horizon.msr key=" + std::to_string(key);
        bb_->insts.push_back(std::move(inst));
    }

    // Read architectural register (0..30 = X0..X30, 31 = SP, 32 = NZCV, 64..95 = V0..V31)
    Value read_arch_reg(int reg, TypeKind t = TypeKind::Int64) {
        TypeKind storage_t = (reg >= 64) ? TypeKind::Int128 : TypeKind::Int64;
        std::string r = fn_->next_reg("r");
        std::string ptr_reg = r + ".p";
        int offset = (reg < 31) ? (reg * 8)
                   : (reg == 31) ? 256
                   : (reg == 32) ? 272
                   : (512 + (reg - 64) * 16);
        int align = (reg >= 64) ? 16 : 8;

        llvm::Value* cpu_v = resolve_llvm_value(Value::cpu_ptr());
        llvm::Value* gep = builder_.CreateInBoundsGEP(
            builder_.getInt8Ty(), cpu_v, builder_.getInt64(offset), strip_ssa_pct(ptr_reg));
        llvm::LoadInst* ld = builder_.CreateAlignedLoad(
            llvm_type(storage_t), gep, llvm::Align(align), strip_ssa_pct(r));
        fn_->register_ssa_value(ptr_reg, gep);
        fn_->register_ssa_value(r, ld);

        Instruction inst;
        inst.tag = OpTag::CpuRegRead;
        inst.arch_reg = reg;
        inst.dst_reg = r;
        inst.dst_type = storage_t;
        inst.llvm_inst = ld;
        inst.text = format_llvm_inst(gep) + "\n  " +
                    format_llvm_inst(ld) + ", !alias.scope !3, !noalias !4";
        bb_->insts.push_back(std::move(inst));

        Value val = Value::reg(storage_t, r, ld);
        if (t != storage_t) {
            if (storage_t == TypeKind::Int64 && t == TypeKind::Int32) {
                return cast("trunc", val, TypeKind::Int32);
            }
            if (storage_t == TypeKind::Int128 && (t == TypeKind::Vec16xI8 || t == TypeKind::Vec8xI16 ||
                t == TypeKind::Vec4xI32 || t == TypeKind::Vec2xI64 || t == TypeKind::Vec4xF32 || t == TypeKind::Vec2xF64)) {
                return bitcast(val, t);
            }
        }
        return val;
    }

    // Write architectural register
    void write_arch_reg(int reg, const Value& v, OpTag tag = OpTag::CpuRegWrite,
                        std::string prefix = "", int cmp_kind = 0, unsigned cmp_sf = 1,
                        Value cmp_lhs = {}, Value cmp_rhs = {}, std::string cmp_res = "") {
        TypeKind storage_t = (reg >= 64) ? TypeKind::Int128 : TypeKind::Int64;
        Value store_val = v;
        if (v.type != storage_t) {
            if (storage_t == TypeKind::Int64 && v.type == TypeKind::Int32) {
                store_val = cast("zext", v, TypeKind::Int64);
            } else if (storage_t == TypeKind::Int64 && v.type == TypeKind::Int1) {
                store_val = cast("zext", v, TypeKind::Int64);
            } else if (storage_t == TypeKind::Int128 && (v.type == TypeKind::Int64 || v.type == TypeKind::Int32 ||
                       v.type == TypeKind::Int16 || v.type == TypeKind::Int8)) {
                store_val = cast("zext", v, TypeKind::Int128);
            } else if (storage_t == TypeKind::Int128) {
                store_val = bitcast(v, TypeKind::Int128);
            }
        }
        int offset = (reg < 31) ? (reg * 8)
                   : (reg == 31) ? 256
                   : (reg == 32) ? 272
                   : (512 + (reg - 64) * 16);
        int align = (reg >= 64) ? 16 : 8;
        std::string ptr_reg = fn_->next_reg("wp");

        llvm::Value* cpu_v = resolve_llvm_value(Value::cpu_ptr());
        llvm::Value* gep = builder_.CreateInBoundsGEP(
            builder_.getInt8Ty(), cpu_v, builder_.getInt64(offset), strip_ssa_pct(ptr_reg));
        llvm::StoreInst* st = builder_.CreateAlignedStore(
            resolve_llvm_value(store_val), gep, llvm::Align(align));
        fn_->register_ssa_value(ptr_reg, gep);

        Instruction inst;
        inst.tag = tag;
        inst.arch_reg = reg;
        inst.op0 = store_val;
        inst.op1 = cmp_lhs;
        inst.op2 = cmp_rhs;
        inst.cmp_kind = cmp_kind;
        inst.cmp_sf = cmp_sf;
        inst.dst_reg = std::move(cmp_res);
        inst.prefix_ir = std::move(prefix);
        inst.llvm_inst = st;
        std::string store_ir = format_llvm_inst(gep) + "\n  " +
                               format_llvm_inst(st) + ", !alias.scope !3, !noalias !4";
        inst.text = inst.prefix_ir.empty() ? store_ir : (inst.prefix_ir + "\n  " + store_ir);
        bb_->insts.push_back(std::move(inst));
    }

    void write_cpu_field_i64(int byte_offset, const Value& v) {
        Value v64 = (v.type == TypeKind::Int64) ? v : cast("zext", v, TypeKind::Int64);
        std::string ptr_reg = fn_->next_reg("fp");
        llvm::Value* cpu_v = resolve_llvm_value(Value::cpu_ptr());
        llvm::Value* gep = builder_.CreateInBoundsGEP(
            builder_.getInt8Ty(), cpu_v, builder_.getInt64(byte_offset), strip_ssa_pct(ptr_reg));
        llvm::StoreInst* st = builder_.CreateAlignedStore(
            resolve_llvm_value(v64), gep, llvm::Align(8));
        fn_->register_ssa_value(ptr_reg, gep);

        Instruction inst;
        inst.tag = OpTag::RawIR;
        inst.llvm_inst = st;
        inst.text = format_llvm_inst(gep) + "\n  " +
                    format_llvm_inst(st) + ", !alias.scope !3, !noalias !4";
        bb_->insts.push_back(std::move(inst));
    }

    Value read_cpu_field_i64(int byte_offset) {
        std::string ptr_reg = fn_->next_reg("fp");
        std::string r = fn_->next_reg("fv");
        llvm::Value* cpu_v = resolve_llvm_value(Value::cpu_ptr());
        llvm::Value* gep = builder_.CreateInBoundsGEP(
            builder_.getInt8Ty(), cpu_v, builder_.getInt64(byte_offset), strip_ssa_pct(ptr_reg));
        llvm::LoadInst* ld = builder_.CreateAlignedLoad(
            builder_.getInt64Ty(), gep, llvm::Align(8), strip_ssa_pct(r));
        fn_->register_ssa_value(ptr_reg, gep);
        fn_->register_ssa_value(r, ld);

        Instruction inst;
        inst.tag = OpTag::RawIR;
        inst.dst_reg = r;
        inst.dst_type = TypeKind::Int64;
        inst.llvm_inst = ld;
        inst.text = format_llvm_inst(gep) + "\n  " +
                    format_llvm_inst(ld) + ", !alias.scope !3, !noalias !4";
        bb_->insts.push_back(std::move(inst));
        return Value::reg(TypeKind::Int64, r, ld);
    }

    Value guest_load(TypeKind t, const Value& addr_i64, uint64_t pc = 0) {
        std::string p = fn_->next_reg("gmp");
        std::string r = fn_->next_reg("gml");
        llvm::Value* pv = builder_.CreateIntToPtr(
            resolve_llvm_value(addr_i64), llvm_type(TypeKind::Pointer), strip_ssa_pct(p));
        llvm::LoadInst* ld = builder_.CreateAlignedLoad(
            llvm_type(t), pv, llvm::Align(1), strip_ssa_pct(r));
        fn_->register_ssa_value(p, pv);
        fn_->register_ssa_value(r, ld);

        Instruction inst;
        inst.tag = OpTag::GuestMemLoad;
        inst.guest_pc = pc;
        inst.dst_reg = r;
        inst.dst_type = t;
        inst.op0 = addr_i64;
        inst.align_bytes = 1;
        inst.llvm_inst = ld;
        inst.text = format_insts({pv, ld});
        bb_->insts.push_back(std::move(inst));
        return Value::reg(t, r, ld);
    }

    void guest_store(TypeKind t, const Value& addr_i64, const Value& val, uint64_t pc = 0) {
        (void)t;
        std::string p = fn_->next_reg("gmp");
        llvm::Value* pv = builder_.CreateIntToPtr(
            resolve_llvm_value(addr_i64), llvm_type(TypeKind::Pointer), strip_ssa_pct(p));
        llvm::StoreInst* st = builder_.CreateAlignedStore(
            resolve_llvm_value(val), pv, llvm::Align(1));
        fn_->register_ssa_value(p, pv);

        Instruction inst;
        inst.tag = OpTag::GuestMemStore;
        inst.guest_pc = pc;
        inst.op0 = addr_i64;
        inst.op1 = val;
        inst.align_bytes = 1;
        inst.llvm_inst = st;
        inst.text = format_insts({pv, st});
        bb_->insts.push_back(std::move(inst));
    }

private:
    Function* fn_;
    BasicBlock* bb_;
    RawBuilder builder_;

    static llvm::LLVMContext& get_fallback_context() {
        static llvm::LLVMContext fallback_ctx;
        return fallback_ctx;
    }
};

} // namespace ultrs::llvm_ir
// UltraTears LLVM C++ Pass Builder Utilities (tools/lifter/passes_llvm_utils.hpp)
//
// Provides direct `llvm::IRBuilder<llvm::NoFolder>`, `llvm::Instruction`, `llvm::SwitchInst`,
// `llvm::AllocaInst`, `llvm::CallInst`, `llvm::LoadInst`, `llvm::StoreInst`, `llvm::ICmpInst`,
// and `llvm::FCmpInst` helpers used across all 23 UltraTears optimization and lowering passes.
#pragma once

#include "llvm_ir.hpp"

namespace ultrs::passes {

using namespace ultrs::llvm_ir;

class PassLLVMBuilder {
public:
    explicit PassLLVMBuilder(Function& fn, BasicBlock* bb = nullptr)
        : fn_(fn),
          bb_(bb ? bb : (!fn.blocks.empty() ? fn.blocks.front().get() : nullptr)),
          builder_(bb_ && bb_->llvm_bb ? bb_->llvm_bb : fn.create_block("entry")->llvm_bb) {}

    ~PassLLVMBuilder() = default;

    llvm::LLVMContext& ctx() const {
        return fn_.parent_mod->context();
    }

    llvm::Module* mod() const {
        return &fn_.parent_mod->raw_module();
    }

    llvm::IRBuilder<llvm::NoFolder>& raw() { return builder_; }

    llvm::Type* type(TypeKind tk) const {
        return to_llvm_type(ctx(), tk);
    }

    llvm::BasicBlock* get_or_create_llvm_bb(const std::string& label) {
        auto it = fn_.block_map.find(label);
        if (it != fn_.block_map.end() && it->second->llvm_bb) {
            return it->second->llvm_bb;
        }
        return llvm::BasicBlock::Create(ctx(), label, fn_.llvm_fn);
    }

    llvm::Value* val(TypeKind tk, const std::string& repr) {
        llvm::Type* ty = type(tk);
        if (repr == "%c") {
            if (fn_.llvm_fn && !fn_.llvm_fn->arg_empty()) return fn_.llvm_fn->getArg(0);
        }
        if (repr == "zeroinitializer") {
            return llvm::Constant::getNullValue(ty);
        }
        if (repr == "poison") {
            return llvm::PoisonValue::get(ty);
        }
        if (repr == "true") return llvm::ConstantInt::getTrue(ctx());
        if (repr == "false") return llvm::ConstantInt::getFalse(ctx());
        if (!repr.empty() && repr[0] == '@') {
            std::string gname = repr.substr(1);
            if (auto* m = mod()) {
                if (auto* gv = m->getNamedValue(gname)) return gv;
                return new llvm::GlobalVariable(*m, llvm::Type::getInt8Ty(ctx()), false,
                                                llvm::GlobalValue::ExternalLinkage, nullptr, gname);
            }
        }
        if (!repr.empty() && repr[0] != '%') {
            if (tk == TypeKind::Float || tk == TypeKind::Double) {
                return llvm::ConstantFP::get(ty, std::stod(repr));
            }
            if (tk == TypeKind::Int128 && repr.rfind("u0x", 0) == 0) {
                llvm::APInt ap(128, repr.substr(3), 16);
                return llvm::ConstantInt::get(ctx(), ap);
            }
            if (repr[0] == '-') {
                int64_t sv = std::stoll(repr);
                return llvm::ConstantInt::getSigned(ty, sv);
            }
            uint64_t uv = std::stoull(repr);
            return llvm::ConstantInt::get(ty, uv, false);
        }
        auto it = fn_.ssa_values.find(repr);
        if (it != fn_.ssa_values.end() && it->second->getType() == ty) {
            return it->second;
        }
        auto arg = std::make_unique<llvm::Argument>(ty, strip_ssa_pct(repr));
        llvm::Value* ptr = arg.get();
        fn_.synthetic_ssa_vals.push_back(std::move(arg));
        fn_.ssa_values[repr] = ptr;
        return ptr;
    }

    std::string emit_alloca(TypeKind tk, unsigned align, const std::string& loc_reg) {
        auto* al = builder_.CreateAlloca(type(tk), nullptr, strip_ssa_pct(loc_reg));
        al->setAlignment(llvm::Align(align));
        fn_.register_ssa_value(loc_reg, al);
        return format_llvm_inst(al);
    }

    std::string emit_local_load(TypeKind tk, const std::string& loc_reg, unsigned align, const std::string& dst_reg) {
        auto* ld = builder_.CreateAlignedLoad(type(tk), val(TypeKind::Pointer, loc_reg),
                                              llvm::Align(align), strip_ssa_pct(dst_reg));
        fn_.register_ssa_value(dst_reg, ld);
        return format_llvm_inst(ld);
    }

    std::string emit_local_store(TypeKind tk, const std::string& val_repr, const std::string& loc_reg, unsigned align) {
        auto* st = builder_.CreateAlignedStore(val(tk, val_repr), val(TypeKind::Pointer, loc_reg),
                                               llvm::Align(align));
        return format_llvm_inst(st);
    }

    std::string emit_cpu_gep_load(int byte_offset, TypeKind tk, unsigned align,
                                  const std::string& p_reg, const std::string& v_reg) {
        auto* gep = builder_.CreateInBoundsGEP(builder_.getInt8Ty(), val(TypeKind::Pointer, "%c"),
                                               builder_.getInt64(byte_offset), strip_ssa_pct(p_reg));
        auto* ld = builder_.CreateAlignedLoad(type(tk), gep, llvm::Align(align), strip_ssa_pct(v_reg));
        fn_.register_ssa_value(p_reg, gep);
        fn_.register_ssa_value(v_reg, ld);
        return format_llvm_inst(gep) + "\n  " + format_llvm_inst(ld) + ", !alias.scope !3, !noalias !4";
    }

    std::string emit_cpu_gep_store(int byte_offset, TypeKind tk, const std::string& val_repr,
                                   unsigned align, const std::string& p_reg) {
        auto* gep = builder_.CreateInBoundsGEP(builder_.getInt8Ty(), val(TypeKind::Pointer, "%c"),
                                               builder_.getInt64(byte_offset), strip_ssa_pct(p_reg));
        auto* st = builder_.CreateAlignedStore(val(tk, val_repr), gep, llvm::Align(align));
        fn_.register_ssa_value(p_reg, gep);
        return format_llvm_inst(gep) + "\n  " + format_llvm_inst(st) + ", !alias.scope !3, !noalias !4";
    }

    std::string emit_icmp(const char* pred, TypeKind tk, const std::string& lhs_repr,
                          const std::string& rhs_repr, const std::string& dst_reg) {
        auto* cmp = builder_.CreateICmp(IRBuilder::parse_icmp_pred(pred),
                                        val(tk, lhs_repr), val(tk, rhs_repr),
                                        strip_ssa_pct(dst_reg));
        fn_.register_ssa_value(dst_reg, cmp);
        return format_llvm_inst(cmp);
    }

    static llvm::CmpInst::Predicate parse_fcmp_pred(const char* pred) {
        if (std::strcmp(pred, "oeq") == 0) return llvm::CmpInst::FCMP_OEQ;
        if (std::strcmp(pred, "ogt") == 0) return llvm::CmpInst::FCMP_OGT;
        if (std::strcmp(pred, "oge") == 0) return llvm::CmpInst::FCMP_OGE;
        if (std::strcmp(pred, "olt") == 0) return llvm::CmpInst::FCMP_OLT;
        if (std::strcmp(pred, "ole") == 0) return llvm::CmpInst::FCMP_OLE;
        if (std::strcmp(pred, "one") == 0) return llvm::CmpInst::FCMP_ONE;
        if (std::strcmp(pred, "ord") == 0) return llvm::CmpInst::FCMP_ORD;
        if (std::strcmp(pred, "uno") == 0) return llvm::CmpInst::FCMP_UNO;
        if (std::strcmp(pred, "ueq") == 0) return llvm::CmpInst::FCMP_UEQ;
        if (std::strcmp(pred, "ugt") == 0) return llvm::CmpInst::FCMP_UGT;
        if (std::strcmp(pred, "uge") == 0) return llvm::CmpInst::FCMP_UGE;
        if (std::strcmp(pred, "ult") == 0) return llvm::CmpInst::FCMP_ULT;
        if (std::strcmp(pred, "ule") == 0) return llvm::CmpInst::FCMP_ULE;
        if (std::strcmp(pred, "une") == 0) return llvm::CmpInst::FCMP_UNE;
        return llvm::CmpInst::FCMP_OEQ;
    }

    std::string emit_fcmp(const char* pred, TypeKind tk, const std::string& lhs_repr,
                          const std::string& rhs_repr, const std::string& dst_reg) {
        auto* cmp = builder_.CreateFCmp(parse_fcmp_pred(pred),
                                        val(tk, lhs_repr), val(tk, rhs_repr),
                                        strip_ssa_pct(dst_reg));
        fn_.register_ssa_value(dst_reg, cmp);
        return format_llvm_inst(cmp);
    }

    std::string emit_switch(const std::string& cond_repr, const std::string& def_label,
                            const std::vector<std::pair<uint64_t, std::string>>& cases) {
        llvm::BasicBlock* def_bb = get_or_create_llvm_bb(def_label);
        auto* sw = builder_.CreateSwitch(val(TypeKind::Int64, cond_repr), def_bb,
                                         static_cast<unsigned>(cases.size()));
        for (const auto& [case_val, target_lbl] : cases) {
            sw->addCase(builder_.getInt64(case_val), get_or_create_llvm_bb(target_lbl));
        }
        return format_llvm_inst(sw);
    }

private:
    Function& fn_;
    BasicBlock* bb_;
    llvm::IRBuilder<llvm::NoFolder> builder_;
};

} // namespace ultrs::passes

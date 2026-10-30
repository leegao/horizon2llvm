// UltraTears Direct LLVM Backend Bridge Implementation (tools/lifter/llvm_backend.cpp)
//
// Directly links (DT_NEEDED) against tools/lifter/third-party/llvm/prebuilts/x86-64/libLLVM.so
// without dlopen/dlsym, providing in-process IR verification, New Pass Manager optimization,
// bitcode/object emission, and AArch64 MCDisassembler formatting.

#include "llvm_backend.hpp"

#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "llvm-c/Analysis.h"
#include "llvm-c/BitWriter.h"
#include "llvm-c/Core.h"
#include "llvm-c/Disassembler.h"
#include "llvm-c/Error.h"
#include "llvm-c/IRReader.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include "llvm-c/Transforms/PassBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

namespace ultrs::llvm_backend {

LibLLVM& LibLLVM::instance() {
    static LibLLVM inst;
    return inst;
}

LibLLVM::LibLLVM() {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();
    LLVMInitializeX86Disassembler();

    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64TargetMC();
    LLVMInitializeAArch64AsmPrinter();
    LLVMInitializeAArch64AsmParser();
    LLVMInitializeAArch64Disassembler();
}

bool LibLLVM::process_module(llvm::Module& cpp_mod,
                             bool verify, const std::string& opt_pipeline,
                             std::string* out_optimized_ll,
                             const std::string& out_bc_path,
                             const std::string& out_obj_path,
                             const std::string& target_triple,
                             const std::string& cpu_arch) const {
    LLVMModuleRef mod = llvm::wrap(&cpp_mod);
    const std::string module_id = cpp_mod.getModuleIdentifier();

    if (verify) {
        std::string err_str;
        llvm::raw_string_ostream err_os(err_str);
        if (llvm::verifyModule(cpp_mod, &err_os)) {
            err_os.flush();
            throw std::runtime_error("llvm::verifyModule failed for " + module_id + ":\n" + err_str);
        }
    }

    LLVMTargetMachineRef tm = nullptr;
    if (!opt_pipeline.empty() || !out_obj_path.empty()) {
        LLVMTargetRef target = nullptr;
        char* targ_err = nullptr;
        if (LLVMGetTargetFromTriple(target_triple.c_str(), &target, &targ_err) != 0) {
            if (targ_err) LLVMDisposeMessage(targ_err);
        } else {
            std::string arch = cpu_arch;
            if (target_triple.find("aarch64") != std::string::npos ||
                target_triple.find("arm64") != std::string::npos) {
                if (arch == "x86-64-v3") arch = "generic";
            }
            tm = LLVMCreateTargetMachine(
                target, target_triple.c_str(), arch.c_str(), "",
                LLVMCodeGenLevelDefault, LLVMRelocPIC, LLVMCodeModelDefault);
        }
    }

    if (!opt_pipeline.empty()) {
        LLVMPassBuilderOptionsRef opts = LLVMCreatePassBuilderOptions();
        LLVMErrorRef pass_err = LLVMRunPasses(mod, opt_pipeline.c_str(), tm, opts);
        LLVMDisposePassBuilderOptions(opts);
        if (pass_err) {
            char* msg = LLVMGetErrorMessage(pass_err);
            std::string s = msg ? msg : "LLVMRunPasses error";
            if (msg) LLVMDisposeErrorMessage(msg);
            if (tm) LLVMDisposeTargetMachine(tm);
            throw std::runtime_error("LLVMRunPasses(" + opt_pipeline + ") failed: " + s);
        }
    }

    if (out_optimized_ll) {
        out_optimized_ll->clear();
        llvm::raw_string_ostream os(*out_optimized_ll);
        cpp_mod.print(os, nullptr);
        os.flush();
    }

    if (!out_bc_path.empty()) {
        if (LLVMWriteBitcodeToFile(mod, out_bc_path.c_str()) != 0) {
            if (tm) LLVMDisposeTargetMachine(tm);
            throw std::runtime_error("LLVMWriteBitcodeToFile failed: " + out_bc_path);
        }
    }

    if (!out_obj_path.empty() && tm) {
        char* emit_err = nullptr;
        if (LLVMTargetMachineEmitToFile(tm, mod, const_cast<char*>(out_obj_path.c_str()),
                                        LLVMObjectFile, &emit_err) != 0) {
            std::string err = emit_err ? emit_err : "unknown codegen error";
            if (emit_err) LLVMDisposeMessage(emit_err);
            LLVMDisposeTargetMachine(tm);
            throw std::runtime_error("LLVMTargetMachineEmitToFile failed: " + err);
        }
    }

    if (tm) LLVMDisposeTargetMachine(tm);
    return true;
}

bool LibLLVM::process_ir(const std::string& ir_text, const std::string& module_id,
                         bool verify, const std::string& opt_pipeline,
                         std::string* out_optimized_ll,
                         const std::string& out_bc_path,
                         const std::string& out_obj_path,
                         const std::string& target_triple,
                         const std::string& cpu_arch) const {
    LLVMContextRef ctx = LLVMContextCreate();
    LLVMMemoryBufferRef mem_buf = LLVMCreateMemoryBufferWithMemoryRangeCopy(
        ir_text.data(), ir_text.size(), module_id.c_str());
    LLVMModuleRef mod = nullptr;
    char* err_msg = nullptr;

    if (LLVMParseIRInContext(ctx, mem_buf, &mod, &err_msg) != 0) {
        std::string err = err_msg ? err_msg : "unknown LLVM IR parse error";
        if (err_msg) LLVMDisposeMessage(err_msg);
        LLVMContextDispose(ctx);
        throw std::runtime_error("LLVMParseIRInContext failed for " + module_id + ":\n" + err);
    }

    try {
        process_module(*llvm::unwrap(mod), verify, opt_pipeline, out_optimized_ll,
                       out_bc_path, out_obj_path, target_triple, cpu_arch);
    } catch (...) {
        LLVMDisposeModule(mod);
        LLVMContextDispose(ctx);
        throw;
    }

    LLVMDisposeModule(mod);
    LLVMContextDispose(ctx);
    return true;
}

std::string LibLLVM::disassemble_aarch64_inst(uint64_t pc, uint32_t word) const {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lk(mtx);
    static LLVMDisasmContextRef dcr = LLVMCreateDisasm("aarch64-unknown-linux-gnu", nullptr, 0, nullptr, nullptr);
    if (!dcr) {
        char fallback[32];
        std::snprintf(fallback, sizeof(fallback), ".word 0x%08x", word);
        return fallback;
    }
    char buf[160] = {0};
    uint8_t bytes[4] = {
        static_cast<uint8_t>(word & 0xFFu),
        static_cast<uint8_t>((word >> 8) & 0xFFu),
        static_cast<uint8_t>((word >> 16) & 0xFFu),
        static_cast<uint8_t>((word >> 24) & 0xFFu),
    };
    size_t sz = LLVMDisasmInstruction(dcr, bytes, 4, pc, buf, sizeof(buf));
    if (sz == 0) {
        std::snprintf(buf, sizeof(buf), "\t.word\t0x%08x", word);
    }
    std::string s = buf;
    while (!s.empty() && (s.front() == '\t' || s.front() == ' ')) {
        s.erase(s.begin());
    }
    for (char& c : s) {
        if (c == '\t') c = ' ';
    }
    return s;
}

std::string LibLLVM::disassemble_aarch64_function(const std::string& fn_name,
                                                  uint64_t entry_pc, uint64_t end_pc,
                                                  const std::vector<uint32_t>& words,
                                                  uint64_t text_base) const {
    std::ostringstream os;
    char hdr[160];
    std::snprintf(hdr, sizeof(hdr), "; Function %s [0x%09" PRIx64 "..0x%09" PRIx64 ")\n%s:\n",
                  fn_name.c_str(), entry_pc, end_pc, fn_name.c_str());
    os << hdr;
    for (uint64_t pc = entry_pc; pc < end_pc; pc += 4) {
        uint32_t w = words[(pc - text_base) / 4];
        std::string asm_str = disassemble_aarch64_inst(pc, w);
        char line[256];
        std::snprintf(line, sizeof(line), "  0x%09" PRIx64 ":  %08x    %s\n", pc, w, asm_str.c_str());
        os << line;
    }
    os << "\n";
    return os.str();
}

} // namespace ultrs::llvm_backend

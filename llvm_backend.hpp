// UltraTears Direct LLVM Backend Bridge (tools/lifter/llvm_backend.hpp)
//
// Directly links (DT_NEEDED) against tools/lifter/third-party/llvm/prebuilts/x86-64/libLLVM.so
// to invoke LLVM 21's in-process IR Verifier, New Pass Manager optimizer (`LLVMRunPasses` /
// `llvm::PassBuilder`), Bitcode Writer, native TargetMachine Object File emitter, and
// AArch64 MCDisassembler.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llvm {
class Module;
} // namespace llvm

namespace ultrs::llvm_backend {

class LibLLVM {
public:
    static LibLLVM& instance();

    bool available() const { return true; }

    // Verify and optionally optimize/compile an in-memory llvm::Module directly
    bool process_module(llvm::Module& mod,
                        bool verify, const std::string& opt_pipeline = "",
                        std::string* out_optimized_ll = nullptr,
                        const std::string& out_bc_path = "",
                        const std::string& out_obj_path = "",
                        const std::string& target_triple = "x86_64-pc-linux-gnu",
                        const std::string& cpu_arch = "x86-64-v3") const;

    // Verify and optionally optimize/compile an in-memory LLVM IR string
    bool process_ir(const std::string& ir_text, const std::string& module_id,
                    bool verify, const std::string& opt_pipeline = "",
                    std::string* out_optimized_ll = nullptr,
                    const std::string& out_bc_path = "",
                    const std::string& out_obj_path = "",
                    const std::string& target_triple = "x86_64-pc-linux-gnu",
                    const std::string& cpu_arch = "x86-64-v3") const;

    // Disassemble a single 32-bit AArch64 instruction word using LLVM's MCDisassembler
    std::string disassemble_aarch64_inst(uint64_t pc, uint32_t word) const;

    // Disassemble an AArch64 function range into human-readable assembly text
    std::string disassemble_aarch64_function(const std::string& fn_name,
                                             uint64_t entry_pc, uint64_t end_pc,
                                             const std::vector<uint32_t>& words,
                                             uint64_t text_base) const;

private:
    LibLLVM();
    ~LibLLVM() = default;
    LibLLVM(const LibLLVM&) = delete;
    LibLLVM& operator=(const LibLLVM&) = delete;
};

} // namespace ultrs::llvm_backend

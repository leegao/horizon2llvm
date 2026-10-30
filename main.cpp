// UltraTears C++ LLVM IR Static Recompiler Driver (tools/lifter/main.cpp)
//
// Drop-in replacement for tools/lifter/gen.py that lifts Nintendo Switch NSO0 ARM64 modules
// (`main`, `subsdk0`, `sdk`) directly to LLVM IR (`.ll` / `.bc`), runs the UltraTears
// LLVM IR Pass pipeline (`build_ultratears_pipeline`), optionally runs in-process `libLLVM`
// verification + New Pass Manager `mem2reg`/`instcombine`/`simplifycfg`/`dse` optimization
// and bitcode/object emission, and writes chunked modules plus `<name>_table.ll` in parallel.
//
// Usage:
//   ultratears-lifter <name> <nso> <base> <outdir> [--hle hle_symbols.txt] [--chunk N]
//                     [--chunk-bytes B] [--big-fn B] [--limit N] [--threads T]
//                     [--target-triple TRIPLE] [--verify-llvm] [--llvm-opt PIPELINE]
//                     [--emit-bc] [--emit-obj] [--patch ADDR:EXPECTED:KIND[:ARG]]
//                     [--force-fnprof] [--no-fast-fp] [--no-shadow-dispatch]
//                     [--safe-passes] [--all-passes] [--no-opt-passes]
//                     [--disable-pass NAME[,NAME...]] [--enable-pass NAME[,NAME...]]
//                     [--find-fn ADDR] [--dump-fn ADDR] [--out-prefix PREFIX]
//                     [--aarch64-nce] [--aarch64-nce-abi]
//   ultratears-lifter --compile-ir <in.ll> <out.o> [-O0|-O2] [--target-triple TRIPLE]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "aarch64_lifter.hpp"
#include "llvm_backend.hpp"
#include "llvm_ir.hpp"
#include "nso_loader.hpp"
#include "passes.hpp"

namespace fs = std::filesystem;
using namespace ultrs;

namespace {

int64_t sext26(uint32_t w) {
    uint32_t v = w & 0x03FFFFFFu;
    return (v & 0x02000000u) ? static_cast<int64_t>(v) - 0x04000000LL : static_cast<int64_t>(v);
}

uint64_t parse_guest_addr(const std::string& raw) {
    std::string s = raw;
    if (s.rfind("f_", 0) == 0 || s.rfind("F_", 0) == 0) {
        s = s.substr(2);
    }
    if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) {
        return std::stoull(s, nullptr, 16);
    }
    return std::stoull(s, nullptr, 16);
}

void apply_pass_list(passes::PassPipelineConfig& cfg, const std::string& csv, bool enable) {
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        while (!item.empty() && (item.front() == ' ' || item.front() == '\t')) item.erase(item.begin());
        while (!item.empty() && (item.back() == ' ' || item.back() == '\t')) item.pop_back();
        if (item.empty()) continue;
        if (!cfg.set_pass_enabled(item, enable)) {
            std::fprintf(stderr, "warning: unknown pass name '%s' in %s\n",
                         item.c_str(), enable ? "--enable-pass" : "--disable-pass");
        }
    }
}

std::string sanitize_hle(const std::string& sym) {
    std::string out = "hle_";
    out.reserve(4 + sym.size());
    for (char c : sym) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }
    return out;
}

bool table_base(const std::vector<uint32_t>& words, size_t i, unsigned reg, size_t window = 40) {
    std::set<unsigned> regs = {reg};
    bool added = false;
    size_t limit = std::min(i + 1 + window, words.size());
    for (size_t j = i + 1; j < limit; ++j) {
        uint32_t w = words[j];
        if ((w & 0xFC000000u) == 0x14000000u || w == 0xD65F03C0u) return false; // B / RET
        if ((w & 0xFFFFFC1Fu) == 0xD61F0000u) { // BR Xn
            return added && (regs.count((w >> 5) & 31u) > 0);
        }
        if (((w & 0x7F200000u) == 0x0B000000u || (w & 0x7F200000u) == 0x0B200000u) &&
            (regs.count((w >> 5) & 31u) || regs.count((w >> 16) & 31u))) {
            regs.insert(w & 31u);
            added = true;
        }
    }
    return false;
}

void write_if_changed(const fs::path& path, const std::string& text) {
    if (fs::exists(path)) {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            std::ostringstream ss;
            ss << in.rdbuf();
            if (ss.str() == text) return;
        }
    }
    std::ofstream out(path, std::ios::binary);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

struct ScanResult {
    std::set<uint64_t> labels;
    std::set<uint64_t> ext;
    std::map<uint64_t, llvm_ir::JumpTableInfo> tables;
};

ScanResult scan_function(uint64_t entry, uint64_t end, uint64_t lo, uint64_t hi,
                         const std::vector<uint32_t>& words, const nso::JumpTableResolver& jt_resolver,
                         std::map<std::string, uint64_t>& stats) {
    ScanResult res;
    res.labels.insert(entry);
    bool terminated = false;
    auto in_text = [&](uint64_t a) { return a >= lo && a < hi && (a % 4) == 0; };

    for (uint64_t a = entry; a < end; a += 4) {
        uint32_t w = words[(a - lo) / 4];
        uint32_t op0 = nso::bits(w, 25, 4);
        terminated = false;

        if ((op0 & 0b1110) == 0b1010) {
            if (nso::bits(w, 26, 5) == 0b00101) { // B / BL
                uint64_t t = static_cast<uint64_t>(static_cast<int64_t>(a) + sext26(w) * 4);
                bool is_bl = nso::bits(w, 31, 1) != 0;
                if (!is_bl) {
                    if (t >= entry && t < end) res.labels.insert(t);
                    else if (in_text(t)) res.ext.insert(t);
                    terminated = true;
                } else {
                    if (in_text(t)) res.ext.insert(t);
                }
            } else if (nso::bits(w, 25, 7) == 0b0101010 && !nso::bits(w, 4, 1)) { // B.cond
                uint64_t t = static_cast<uint64_t>(static_cast<int64_t>(a) + nso::sext(nso::bits(w, 5, 19), 19) * 4);
                unsigned cond = nso::bits(w, 0, 4);
                if (t >= entry && t < end) res.labels.insert(t);
                else if (in_text(t)) res.ext.insert(t);
                if (cond >= 14) terminated = true;
            } else if (nso::bits(w, 25, 6) == 0b011010) { // CBZ / CBNZ
                uint64_t t = static_cast<uint64_t>(static_cast<int64_t>(a) + nso::sext(nso::bits(w, 5, 19), 19) * 4);
                if (t >= entry && t < end) res.labels.insert(t);
                else if (in_text(t)) res.ext.insert(t);
            } else if (nso::bits(w, 25, 6) == 0b011011) { // TBZ / TBNZ
                uint64_t t = static_cast<uint64_t>(static_cast<int64_t>(a) + nso::sext(nso::bits(w, 5, 14), 14) * 4);
                if (t >= entry && t < end) res.labels.insert(t);
                else if (in_text(t)) res.ext.insert(t);
            } else if (nso::bits(w, 25, 7) == 0b1101011) { // BR / BLR / RET
                unsigned opc = nso::bits(w, 21, 4);
                if (opc == 0) { // BR
                    auto jt = jt_resolver.resolve(a, entry, end, stats);
                    if (jt.has_value() && jt->size() > 1) {
                        llvm_ir::JumpTableInfo info;
                        info.is_all_labels = false;
                        info.targets = *jt;
                        res.tables[a] = std::move(info);
                        res.labels.insert(jt->begin(), jt->end());
                        stats["jumptables"]++;
                    } else if (jt_resolver.computed_target(a, entry)) {
                        llvm_ir::JumpTableInfo info;
                        info.is_all_labels = true;
                        res.tables[a] = std::move(info);
                        stats["br_all_labels"]++;
                    } else {
                        stats["br_dispatch"]++;
                    }
                    terminated = true;
                } else if (opc == 2) { // RET
                    terminated = true;
                }
            } else if (nso::bits(w, 24, 8) == 0b11010100) {
                if (nso::bits(w, 21, 3) == 1 && nso::bits(w, 0, 5) == 0) { // BRK
                    terminated = true;
                }
            }
        }
    }
    if (!terminated && end < hi) {
        res.ext.insert(end);
    }
    return res;
}

passes::PatchRule parse_patch_arg(const std::string& spec) {
    std::vector<std::string> parts;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ':')) parts.push_back(item);
    if (parts.size() < 3) {
        throw std::runtime_error("invalid --patch spec (expected ADDR:WORD:KIND[:ARG]): " + spec);
    }
    passes::PatchRule r;
    r.guest_pc = std::stoull(parts[0], nullptr, 0);
    r.expected_word = static_cast<uint32_t>(std::stoul(parts[1], nullptr, 0));
    r.name = parts[2];
    if (parts[2] == "drs_off") {
        r.kind = passes::PatchRule::Kind::DrsOff;
        r.target_pc = parts.size() >= 4 ? std::stoull(parts[3], nullptr, 0) : 0x800584c3cULL;
    } else if (parts[2] == "nop") {
        r.kind = passes::PatchRule::Kind::Nop;
    } else if (parts[2] == "branch") {
        r.kind = passes::PatchRule::Kind::BranchTarget;
        r.target_pc = std::stoull(parts.at(3), nullptr, 0);
    } else {
        throw std::runtime_error("unknown patch kind in --patch: " + parts[2]);
    }
    return r;
}

} // namespace

int main(int argc, char** argv) {
    // Standalone IR -> Object compiler mode (used by build_game.py on Linux when clang CLI is absent)
    if (argc >= 4 && std::string(argv[1]) == "--compile-ir") {
        std::string in_path = argv[2];
        std::string out_path = argv[3];
#ifdef _WIN32
        std::string triple = "x86_64-w64-windows-gnu";
#else
        std::string triple = "x86_64-pc-linux-gnu";
#endif
        std::string opt_pipe = "default<O2>";
        for (int i = 4; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "-O0") opt_pipe = "";
            else if (a == "-O2") opt_pipe = "default<O2>";
            else if (a == "--target-triple" && i + 1 < argc) triple = argv[++i];
        }
        try {
            if (!llvm_backend::LibLLVM::instance().available()) {
                std::fprintf(stderr, "FATAL: libLLVM shared library not available for --compile-ir\n");
                return 1;
            }
            std::ifstream in(in_path, std::ios::binary);
            if (!in) {
                std::fprintf(stderr, "FATAL: cannot open %s\n", in_path.c_str());
                return 1;
            }
            std::ostringstream ss;
            ss << in.rdbuf();
            fs::create_directories(fs::path(out_path).parent_path());
            llvm_backend::LibLLVM::instance().process_ir(
                ss.str(), fs::path(in_path).stem().string(), false, opt_pipe, nullptr, "", out_path, triple);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FATAL [--compile-ir %s]: %s\n", in_path.c_str(), e.what());
            return 1;
        }
    }

    if (argc < 5) {
        std::cerr << "usage: " << argv[0]
                  << " <name> <nso> <base> <outdir> [--hle file] [--chunk N] [--chunk-bytes B]"
                     " [--big-fn B] [--limit N] [--threads T] [--target-triple TRIPLE]"
                     " [--verify-llvm] [--llvm-opt PIPELINE] [--emit-bc] [--emit-obj] [--emit-raw] [--emit-arm64-asm]"
                     " [--patch ADDR:WORD:KIND[:ARG]] [--force-fnprof] [--no-fast-fp]"
                     " [--no-shadow-dispatch] [--safe-passes] [--all-passes] [--no-opt-passes]"
                     " [--disable-pass NAME[,NAME...]] [--enable-pass NAME[,NAME...]]"
                     " [--find-fn ADDR] [--dump-fn ADDR] [--out-prefix PREFIX]"
                     " [--aarch64-nce] [--aarch64-nce-abi]\n"
                  << "       " << argv[0] << " --compile-ir <in.ll> <out.o> [-O0|-O2] [--target-triple TRIPLE]\n";
        return 2;
    }

    std::string mod_name = argv[1];
    std::string nso_path = argv[2];
    uint64_t base = std::stoull(argv[3], nullptr, 0);
    fs::path outdir = argv[4];
    std::string hle_file;
    size_t max_chunk_fns = 1500;
    size_t max_chunk_bytes = 2500000;
    size_t big_fn_bytes = 180000;
    size_t limit_fns = 0;
    unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());
#ifdef _WIN32
    std::string target_triple = "x86_64-w64-windows-gnu";
#else
    std::string target_triple = "x86_64-pc-linux-gnu";
#endif
    bool verify_llvm = false;
    bool no_write = false;
    bool emit_bc = false;
    bool emit_obj = false;
    bool emit_raw_ll = false;
    bool emit_arm64_asm = false;
    bool force_fnprof = false;
    bool enable_fast_fp = true;
    bool enable_shadow_dispatch = true;
    bool force_aarch64_nce = false;
    bool enable_nce_abi = false;
    // By default in the full NSO recompiler, disable higher-risk interprocedural/speculative passes
    bool safe_passes_mode = true;
    bool no_opt_passes_mode = false;
    std::vector<std::pair<std::string, bool>> pass_overrides;
    std::vector<uint64_t> find_fn_addrs;
    std::vector<uint64_t> dump_fn_addrs;
    std::string dump_out_prefix;
    std::string llvm_opt_pipeline;
    std::vector<passes::PatchRule> extra_patches;

    if (const char* env_all = std::getenv("ULTRATEARS_ALL_PASSES")) {
        if (std::string(env_all) == "1") safe_passes_mode = false;
    }
    if (const char* env_safe = std::getenv("ULTRATEARS_SAFE_PASSES")) {
        if (std::string(env_safe) == "1") safe_passes_mode = true;
        else if (std::string(env_safe) == "0") safe_passes_mode = false;
    }
    if (const char* env_noopt = std::getenv("ULTRATEARS_NO_OPT_PASSES")) {
        if (std::string(env_noopt) == "1") no_opt_passes_mode = true;
    }
    if (const char* env_dis = std::getenv("ULTRATEARS_DISABLE_PASSES")) {
        pass_overrides.emplace_back(env_dis, false);
    }
    if (const char* env_en = std::getenv("ULTRATEARS_ENABLE_PASSES")) {
        pass_overrides.emplace_back(env_en, true);
    }

    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--hle" && i + 1 < argc) hle_file = argv[++i];
        else if (arg == "--chunk" && i + 1 < argc) max_chunk_fns = std::stoull(argv[++i]);
        else if (arg == "--chunk-bytes" && i + 1 < argc) max_chunk_bytes = std::stoull(argv[++i]);
        else if (arg == "--big-fn" && i + 1 < argc) big_fn_bytes = std::stoull(argv[++i]);
        else if (arg == "--limit" && i + 1 < argc) limit_fns = std::stoull(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) num_threads = std::max(1u, static_cast<unsigned>(std::stoul(argv[++i])));
        else if (arg == "--target-triple" && i + 1 < argc) target_triple = argv[++i];
        else if (arg == "--verify-llvm") verify_llvm = true;
        else if (arg == "--no-write") no_write = true;
        else if (arg == "--llvm-opt" && i + 1 < argc) llvm_opt_pipeline = argv[++i];
        else if (arg == "--emit-bc") emit_bc = true;
        else if (arg == "--emit-obj") emit_obj = true;
        else if (arg == "--emit-raw" || arg == "--emit-raw-ll") { emit_raw_ll = true; emit_arm64_asm = true; }
        else if (arg == "--emit-arm64-asm" || arg == "--emit-asm") emit_arm64_asm = true;
        else if (arg == "--patch" && i + 1 < argc) extra_patches.push_back(parse_patch_arg(argv[++i]));
        else if (arg == "--force-fnprof") force_fnprof = true;
        else if (arg == "--no-fast-fp") enable_fast_fp = false;
        else if (arg == "--no-shadow-dispatch") enable_shadow_dispatch = false;
        else if (arg == "--safe-passes") { safe_passes_mode = true; no_opt_passes_mode = false; }
        else if (arg == "--all-passes") { safe_passes_mode = false; no_opt_passes_mode = false; }
        else if (arg == "--no-opt-passes" || arg == "--minimal-passes") { no_opt_passes_mode = true; }
        else if (arg == "--disable-pass" && i + 1 < argc) pass_overrides.emplace_back(argv[++i], false);
        else if (arg == "--enable-pass" && i + 1 < argc) pass_overrides.emplace_back(argv[++i], true);
        else if (arg == "--find-fn" && i + 1 < argc) find_fn_addrs.push_back(parse_guest_addr(argv[++i]));
        else if (arg == "--dump-fn" && i + 1 < argc) dump_fn_addrs.push_back(parse_guest_addr(argv[++i]));
        else if (arg == "--out-prefix" && i + 1 < argc) dump_out_prefix = argv[++i];
        else if (arg == "--aarch64-nce") force_aarch64_nce = true;
        else if (arg == "--aarch64-nce-abi") enable_nce_abi = true;
        else if (arg == "--simd" && i + 1 < argc) { ++i; /* accepted for CLI compatibility with gen.py */ }
    }

    auto t0 = std::chrono::steady_clock::now();
    try {
        nso::NsoFile nso(nso_path);
        uint64_t lo = base + nso.text.first;
        uint64_t hi = lo + nso.text.second;
        uint64_t ro_lo = base + nso.rodata.first;
        uint64_t ro_hi = ro_lo + nso.rodata.second;
        size_t nwords = nso.text.second / 4;
        std::vector<uint32_t> words(nwords);
        std::memcpy(words.data(), nso.image.data() + nso.text.first, nwords * 4);

        // Track .rodata addresses with dynamic symbol relocations (`r.sym != 0`) so rodata folding never folds them
        std::unordered_set<uint64_t> unrelocated_rodata_addrs;
        for (const auto& r : nso.relocs) {
            if (r.sym != 0) {
                uint64_t va = base + r.offset;
                if (va >= ro_lo && va < ro_hi) {
                    unrelocated_rodata_addrs.insert(va);
                }
            }
        }

        auto in_text = [&](uint64_t a) { return a >= lo && a < hi && (a % 4) == 0; };

        // 1. Discover initial boundaries
        std::set<uint64_t> starts;
        for (uint64_t a : nso::eh_frame_functions(nso)) {
            if (in_text(base + a)) starts.insert(base + a);
        }
        for (const auto& s : nso.syms) {
            if (s.shndx && (s.info & 0xF) == 2 && in_text(base + s.value)) {
                starts.insert(base + s.value);
            }
        }
        for (const auto& r : nso.relocs) {
            if (r.type == nso::R_AARCH64_RELATIVE && in_text(base + static_cast<uint64_t>(r.addend))) {
                starts.insert(base + static_cast<uint64_t>(r.addend));
            }
        }
        size_t initial_boundaries = starts.size();

        // HLE symbols
        std::unordered_map<uint64_t, std::string> hle_map;
        if (!hle_file.empty() && fs::exists(hle_file)) {
            std::unordered_set<std::string> want;
            std::ifstream hf(hle_file);
            std::string line;
            while (std::getline(hf, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) line.pop_back();
                if (!line.empty() && line[0] != '#') want.insert(line);
            }
            for (const auto& s : nso.syms) {
                if (s.shndx && want.count(s.name)) {
                    hle_map[base + s.value] = sanitize_hle(s.name);
                }
            }
        }

        std::set<uint64_t> start_set = starts;
        for (size_t i = 0; i < words.size(); ++i) {
            uint32_t w = words[i];
            if ((w & 0xFC000000u) == 0x94000000u) {
                uint64_t t = static_cast<uint64_t>(static_cast<int64_t>(lo + i * 4) + sext26(w) * 4);
                if (in_text(t)) start_set.insert(t);
            }
        }
        std::unordered_map<unsigned, std::pair<uint64_t, size_t>> adrp_regs;
        for (size_t i = 0; i < words.size(); ++i) {
            uint32_t w = words[i];
            uint64_t pc = lo + i * 4;
            if ((w & 0x9F000000u) == 0x10000000u) { // ADR
                int64_t imm = nso::sext((nso::bits(w, 5, 19) << 2) | nso::bits(w, 29, 2), 21);
                uint64_t t = static_cast<uint64_t>(static_cast<int64_t>(pc) + imm);
                if (in_text(t) && !table_base(words, i, w & 31u)) {
                    start_set.insert(t);
                }
            } else if ((w & 0x9F000000u) == 0x90000000u) { // ADRP
                int64_t imm = nso::sext((nso::bits(w, 5, 19) << 2) | nso::bits(w, 29, 2), 21);
                adrp_regs[w & 31u] = {(pc & ~0xFFFULL) + static_cast<uint64_t>(imm << 12), i};
            } else if ((w & 0xFFC00000u) == 0x91000000u) { // ADD (imm, 64-bit, unshifted)
                unsigned rn = (w >> 5) & 31u;
                auto it = adrp_regs.find(rn);
                if (it != adrp_regs.end() && i - it->second.second <= 8) {
                    uint64_t t = it->second.first + ((w >> 10) & 0xFFFu);
                    if (in_text(t)) start_set.insert(t);
                }
            }
        }

        // 2. Fixpoint CFG & Jump-Table Discovery
        nso::JumpTableResolver jt_resolver(nso, base, words);
        std::map<std::pair<uint64_t, uint64_t>, ScanResult> scanned;
        std::map<std::string, uint64_t> stats;
        unsigned rounds = 0;
        std::vector<uint64_t> order;

        while (true) {
            ++rounds;
            order.assign(start_set.begin(), start_set.end());
            std::set<uint64_t> newly_discovered;
            for (size_t i = 0; i < order.size(); ++i) {
                uint64_t a = order[i];
                uint64_t end = (i + 1 < order.size()) ? order[i + 1] : hi;
                auto key = std::make_pair(a, end);
                auto it = scanned.find(key);
                if (it == scanned.end()) {
                    it = scanned.emplace(key, scan_function(a, end, lo, hi, words, jt_resolver, stats)).first;
                }
                for (uint64_t ext_t : it->second.ext) {
                    if (!start_set.count(ext_t)) newly_discovered.insert(ext_t);
                }
            }
            if (newly_discovered.empty()) break;
            start_set.insert(newly_discovered.begin(), newly_discovered.end());
        }

        double scan_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (mod_name == "main") {
            auto word_at = [&](uint64_t va) -> uint32_t {
                if (va < lo || va >= hi) return 0;
                return words[(va - lo) / 4];
            };
            uint32_t w143 = word_at(passes::UltraTearsPatchPass::DRS_PC_143);
            uint32_t w142 = word_at(passes::UltraTearsPatchPass::DRS_PC_142);
            if (w143 == passes::UltraTearsPatchPass::DRS_WORD_143) {
                std::fprintf(stderr, "main: detected TOTK 1.4.3 (drs_off at 0x%" PRIx64 " -> 0x%" PRIx64 ")\n",
                             passes::UltraTearsPatchPass::DRS_PC_143, passes::UltraTearsPatchPass::DRS_TARGET_143);
            } else if (w142 == passes::UltraTearsPatchPass::DRS_WORD_142) {
                std::fprintf(stderr, "main: detected TOTK 1.4.2 (drs_off at 0x%" PRIx64 " -> 0x%" PRIx64 ")\n",
                             passes::UltraTearsPatchPass::DRS_PC_142, passes::UltraTearsPatchPass::DRS_TARGET_142);
            } else {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "patch drs_off: neither TOTK 1.4.3 (0x%" PRIx64 ": expected 0x%08x, found 0x%08x) "
                              "nor TOTK 1.4.2 (0x%" PRIx64 ": expected 0x%08x, found 0x%08x) matched",
                              passes::UltraTearsPatchPass::DRS_PC_143, passes::UltraTearsPatchPass::DRS_WORD_143, w143,
                              passes::UltraTearsPatchPass::DRS_PC_142, passes::UltraTearsPatchPass::DRS_WORD_142, w142);
                throw std::runtime_error(buf);
            }
        }
        std::fprintf(stderr, "%s: %zu boundaries, %zu functions, %u rounds, %.1fs\n",
                     mod_name.c_str(), initial_boundaries, order.size(), rounds, scan_sec);

        struct ChunkSpec {
            size_t chunk_idx;
            std::vector<std::pair<uint64_t, uint64_t>> fn_ranges;
        };
        std::vector<ChunkSpec> chunks;
        {
            size_t ci = 0;
            size_t chunk_idx = 0;
            size_t effective_order_size = (limit_fns > 0 && order.size() > limit_fns) ? limit_fns : order.size();
            while (ci < effective_order_size) {
                ChunkSpec cs;
                cs.chunk_idx = chunk_idx++;
                size_t est_bytes = 0;
                while (ci < effective_order_size && cs.fn_ranges.size() < max_chunk_fns && est_bytes < max_chunk_bytes) {
                    uint64_t a = order[ci];
                    uint64_t end = (ci + 1 < start_set.size())
                                 ? *std::next(start_set.find(a))
                                 : hi;
                    size_t fn_insns = (end - a) / 4;
                    size_t fn_est = fn_insns * 120;
                    if (fn_est > big_fn_bytes && !cs.fn_ranges.empty()) {
                        break;
                    }
                    cs.fn_ranges.push_back({a, end});
                    est_bytes += fn_est;
                    ++ci;
                    if (fn_est > big_fn_bytes) {
                        break;
                    }
                }
                chunks.push_back(std::move(cs));
            }
        }

        passes::PassPipelineConfig pipe_cfg;
        pipe_cfg.extra_patches = extra_patches;
        pipe_cfg.include_default_drs = true;
        pipe_cfg.force_fnprof = force_fnprof;
        pipe_cfg.enable_fast_fp = enable_fast_fp;
        pipe_cfg.enable_shadow_dispatch = enable_shadow_dispatch;
        pipe_cfg.force_aarch64_intrinsics = force_aarch64_nce;
        pipe_cfg.enable_nce_abi = enable_nce_abi;
        pipe_cfg.image_data = nso.image.data();
        pipe_cfg.image_size = nso.image.size();
        pipe_cfg.image_base = base;
        pipe_cfg.rodata_lo = ro_lo;
        pipe_cfg.rodata_hi = ro_hi;
        pipe_cfg.text_lo = lo;
        pipe_cfg.text_hi = hi;
        pipe_cfg.unrelocated_rodata_addrs = &unrelocated_rodata_addrs;
        if (no_opt_passes_mode) {
            pipe_cfg.disable_all_opt_passes();
        } else if (safe_passes_mode) {
            pipe_cfg.disable_high_risk_passes();
        }
        if (!enable_fast_fp) pipe_cfg.enable_fast_fp = false;
        if (!enable_shadow_dispatch) pipe_cfg.enable_shadow_dispatch = false;
        if (enable_nce_abi) pipe_cfg.enable_nce_abi = true;
        for (const auto& [csv, en] : pass_overrides) {
            apply_pass_list(pipe_cfg, csv, en);
        }

        std::unordered_map<uint64_t, passes::LeafFunctionSlice> global_leaves;
        if (pipe_cfg.enable_cross_chunk_leaf_inline) {
            global_leaves = passes::CrossChunkLeafInlinePass::collect_nso_leaves(
                order, lo, hi, words, hle_map, 8);
            pipe_cfg.global_leaves = &global_leaves;
        }

        auto populate_fn_ir = [&](llvm_ir::Module& M, lifter::AArch64ToLLVMLifter& arm_lifter,
                                  uint64_t a, uint64_t end) -> llvm_ir::Function* {
            const auto& sr = scanned.at({a, end});
            llvm_ir::Function* fn = M.create_function(lifter::fn_symbol(a), a, end);
            if (hle_map.count(a)) {
                fn->hle_symbol = hle_map.at(a);
            }
            fn->jump_tables = sr.tables;
            for (const auto& [br_pc, jti] : sr.tables) {
                if (jti.is_all_labels) fn->has_all_labels = true;
            }
            if (fn->has_all_labels) {
                for (uint64_t pc = a; pc < end; pc += 4) fn->label_pcs.insert(pc);
            } else {
                fn->label_pcs = sr.labels;
                if (a <= passes::UltraTearsPatchPass::DRS_TARGET_143 &&
                    passes::UltraTearsPatchPass::DRS_TARGET_143 < end) {
                    fn->label_pcs.insert(passes::UltraTearsPatchPass::DRS_TARGET_143);
                }
                if (a <= passes::UltraTearsPatchPass::DRS_TARGET_142 &&
                    passes::UltraTearsPatchPass::DRS_TARGET_142 < end) {
                    fn->label_pcs.insert(passes::UltraTearsPatchPass::DRS_TARGET_142);
                }
                for (const auto& pr : extra_patches) {
                    if (pr.target_pc >= a && pr.target_pc < end) {
                        fn->label_pcs.insert(pr.target_pc);
                    }
                }
            }
            arm_lifter.lift_function(M, fn, words, lo);
            return fn;
        };

        auto find_fn_by_addr = [&](uint64_t query_addr, uint64_t& out_entry, uint64_t& out_end, size_t& out_chunk_idx) -> bool {
            if (query_addr < lo || query_addr >= hi) return false;
            auto it = std::upper_bound(order.begin(), order.end(), query_addr);
            if (it == order.begin()) return false;
            --it;
            out_entry = *it;
            size_t idx = static_cast<size_t>(std::distance(order.begin(), it));
            out_end = (idx + 1 < order.size()) ? order[idx + 1] : hi;
            out_chunk_idx = 0;
            for (const auto& cs : chunks) {
                for (auto [ca, cend] : cs.fn_ranges) {
                    if (ca == out_entry) {
                        out_chunk_idx = cs.chunk_idx;
                        return true;
                    }
                }
            }
            return true;
        };

        // Handle `--find-fn` and `--dump-fn` debugging queries immediately without full NSO codegen
        if (!find_fn_addrs.empty() || !dump_fn_addrs.empty()) {
            for (uint64_t q : find_fn_addrs) {
                uint64_t f_entry = 0, f_end = 0;
                size_t c_idx = 0;
                if (!find_fn_by_addr(q, f_entry, f_end, c_idx)) {
                    std::printf("0x%" PRIx64 ": outside %s .text [0x%" PRIx64 "..0x%" PRIx64 ")\n",
                                q, mod_name.c_str(), lo, hi);
                } else {
                    std::printf("0x%" PRIx64 " -> %s+0x%" PRIx64 " [0x%" PRIx64 "..0x%" PRIx64 ", %zu insns] in %s_%03zu.ll (%s+0x%" PRIx64 ")\n",
                                q, lifter::fn_symbol(f_entry).c_str(), q - f_entry,
                                f_entry, f_end, static_cast<size_t>((f_end - f_entry) / 4),
                                mod_name.c_str(), c_idx, mod_name.c_str(), q - base);
                }
            }
            for (uint64_t q : dump_fn_addrs) {
                uint64_t f_entry = 0, f_end = 0;
                size_t c_idx = 0;
                if (!find_fn_by_addr(q, f_entry, f_end, c_idx)) {
                    std::fprintf(stderr, "error: 0x%" PRIx64 " is outside %s .text [0x%" PRIx64 "..0x%" PRIx64 ")\n",
                                 q, mod_name.c_str(), lo, hi);
                    return 1;
                }
                lifter::AArch64ToLLVMLifter arm_lifter(lo, hi);
                std::string fn_name = lifter::fn_symbol(f_entry);
                llvm_ir::Module M(fn_name, target_triple);
                populate_fn_ir(M, arm_lifter, f_entry, f_end);
                std::string pre_ir = M.serialize();

                llvm_ir::ModulePassManager MPM;
                passes::build_ultratears_pipeline(MPM, pipe_cfg);
                MPM.run(M);
                std::string post_ir = M.serialize();

                fs::create_directories(outdir);
                std::string prefix = !dump_out_prefix.empty()
                                   ? dump_out_prefix
                                   : (outdir / fn_name).string();
                fs::path pre_path = prefix + "_prepass.ll";
                fs::path post_path = prefix + "_postpass.ll";
                write_if_changed(pre_path, pre_ir);
                write_if_changed(post_path, post_ir);
                std::printf("dumped %s [0x%" PRIx64 "..0x%" PRIx64 ", query 0x%" PRIx64 " = +0x%" PRIx64 ", chunk %s_%03zu.ll]:\n"
                            "  pre-pass  IR: %s\n"
                            "  post-pass IR: %s\n",
                            fn_name.c_str(), f_entry, f_end, q, q - f_entry,
                            mod_name.c_str(), c_idx,
                            pre_path.string().c_str(), post_path.string().c_str());
            }
            return 0;
        }

        fs::create_directories(outdir);
        if (limit_fns > 0 && order.size() > limit_fns) {
            order.resize(limit_fns);
        }

        // Write `<mod>_chunks.map` index mapping every function `f_<entry>` to its `<mod>_<NNN>.ll` chunk file
        if (!no_write) {
            std::ostringstream map_os;
            map_os << "# UltraTears function-to-chunk map for module '" << mod_name << "'\n";
            map_os << "# Format: <chunk_file> <fn_symbol> <start_pc> <end_pc> <insn_count>\n";
            for (const auto& cs : chunks) {
                char chunk_file[64];
                std::snprintf(chunk_file, sizeof(chunk_file), "%s_%03zu.ll", mod_name.c_str(), cs.chunk_idx);
                for (auto [a, end] : cs.fn_ranges) {
                    char line_buf[160];
                    std::snprintf(line_buf, sizeof(line_buf), "%-16s %-16s 0x%09" PRIx64 " 0x%09" PRIx64 " %zu\n",
                                  chunk_file, lifter::fn_symbol(a).c_str(), a, end, static_cast<size_t>((end - a) / 4));
                    map_os << line_buf;
                }
            }
            write_if_changed(outdir / (mod_name + "_chunks.map"), map_os.str());
        }

        // 3. Multi-threaded LLVM IR Lifting + Pass Execution per Chunk
        std::atomic<size_t> next_chunk{0};
        std::atomic<size_t> completed_fns{0};
        std::mutex log_mtx;
        std::mutex llvm_mtx;

        auto worker = [&]() {
            lifter::AArch64ToLLVMLifter arm_lifter(lo, hi);
            while (true) {
                size_t idx = next_chunk.fetch_add(1);
                if (idx >= chunks.size()) break;
                const auto& cs = chunks[idx];

                char mod_id[64];
                std::snprintf(mod_id, sizeof(mod_id), "%s_%03zu", mod_name.c_str(), cs.chunk_idx);
                llvm_ir::Module M(mod_id, target_triple);

                for (auto [a, end] : cs.fn_ranges) {
                    populate_fn_ir(M, arm_lifter, a, end);
                }

                std::ostringstream hdr;
                if (!cs.fn_ranges.empty()) {
                    char hbuf[160];
                    std::snprintf(hbuf, sizeof(hbuf),
                                  "; UltraTears chunk %s: %zu functions [0x%" PRIx64 "..0x%" PRIx64 ")\n",
                                  mod_id, cs.fn_ranges.size(),
                                  cs.fn_ranges.front().first, cs.fn_ranges.back().second);
                    hdr << hbuf;
                }

                if (!no_write && emit_raw_ll) {
                    std::string raw_ir_text = hdr.str() + passes::serialize_unlowered_reference_ir(M, pipe_cfg);
                    fs::path raw_path = outdir / (std::string(mod_id) + "_raw.ll");
                    write_if_changed(raw_path, raw_ir_text);
                }

                if (!no_write && emit_arm64_asm) {
                    std::ostringstream asm_os;
                    asm_os << hdr.str() << "\n";
                    for (auto [a, end] : cs.fn_ranges) {
                        asm_os << llvm_backend::LibLLVM::instance().disassemble_aarch64_function(
                            lifter::fn_symbol(a), a, end, words, lo);
                    }
                    fs::path asm_path = outdir / (std::string(mod_id) + "_arm64.asm");
                    write_if_changed(asm_path, asm_os.str());
                }

                // Run the UltraTears LLVM IR Pass Pipeline
                llvm_ir::ModulePassManager MPM;
                passes::build_ultratears_pipeline(MPM, pipe_cfg);
                MPM.run(M);

                std::string ir_text = hdr.str() + M.serialize();
                if ((verify_llvm || !llvm_opt_pipeline.empty() || emit_bc || emit_obj) &&
                    llvm_backend::LibLLVM::instance().available()) {
                    std::string opt_ll;
                    std::string bc_path = emit_bc ? (outdir / (std::string(mod_id) + ".bc")).string() : "";
                    std::string obj_path = emit_obj ? (outdir / (std::string(mod_id) + ".o")).string() : "";
                    std::lock_guard<std::mutex> lk(llvm_mtx);
                    llvm_backend::LibLLVM::instance().process_ir(
                        ir_text, mod_id, verify_llvm, llvm_opt_pipeline,
                        !llvm_opt_pipeline.empty() ? &opt_ll : nullptr,
                        bc_path, obj_path, target_triple);
                    if (!opt_ll.empty()) ir_text = std::move(opt_ll);
                }

                fs::path out_path = outdir / (std::string(mod_id) + ".ll");
                if (!no_write) write_if_changed(out_path, ir_text);

                size_t done = completed_fns.fetch_add(cs.fn_ranges.size()) + cs.fn_ranges.size();
                {
                    std::lock_guard<std::mutex> lk(log_mtx);
                    std::fprintf(stderr, "%s: progress %zu/%zu files %zu\n",
                                 mod_name.c_str(), done, order.size(), idx + 1);
                    std::fflush(stderr);
                }
            }
        };

        unsigned actual_threads = std::min<unsigned>(num_threads, std::max<size_t>(1, chunks.size()));
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < actual_threads; ++t) {
            pool.emplace_back(worker);
        }
        for (auto& th : pool) th.join();

        std::regex chunk_re("^" + mod_name + "_(\\d+)\\.(ll|c)$");
        for (const auto& entry : fs::directory_iterator(outdir)) {
            std::string fn = entry.path().filename().string();
            std::smatch mm;
            if (std::regex_match(fn, mm, chunk_re)) {
                size_t cnum = std::stoull(mm[1].str());
                if (mm[2].str() == "c" || cnum >= chunks.size()) {
                    fs::remove(entry.path());
                }
            }
        }
        if (fs::exists(outdir / (mod_name + "_table.c"))) {
            fs::remove(outdir / (mod_name + "_table.c"));
        }

        // 4. Emit `<name>_table.ll`
        {
            std::ostringstream ts;
            ts << "; ModuleID = '" << mod_name << "_table'\n";
            ts << "target triple = \"" << target_triple << "\"\n\n";
            for (uint64_t a : order) {
                ts << "declare dso_local void @" << lifter::fn_symbol(a) << "(ptr)\n";
            }
            ts << "\n@" << mod_name << "_fn_addrs = dso_local constant [" << order.size() << " x i64] [\n";
            for (size_t i = 0; i < order.size(); ++i) {
                ts << "  i64 " << order[i] << (i + 1 < order.size() ? ",\n" : "\n");
            }
            ts << "], align 8\n\n";

            ts << "@" << mod_name << "_fn_ptrs = dso_local global [" << order.size() << " x ptr] [\n";
            for (size_t i = 0; i < order.size(); ++i) {
                ts << "  ptr @" << lifter::fn_symbol(order[i]) << (i + 1 < order.size() ? ",\n" : "\n");
            }
            ts << "], align 8\n\n";

            ts << "@" << mod_name << "_fn_count = dso_local constant i64 " << order.size() << ", align 8\n";
            std::string table_ir = ts.str();
            if ((verify_llvm || emit_bc || emit_obj) && llvm_backend::LibLLVM::instance().available()) {
                std::string bc_path = emit_bc ? (outdir / (mod_name + "_table.bc")).string() : "";
                std::string obj_path = emit_obj ? (outdir / (mod_name + "_table.o")).string() : "";
                llvm_backend::LibLLVM::instance().process_ir(
                    table_ir, mod_name + "_table", verify_llvm, "", nullptr, bc_path, obj_path, target_triple);
            }
            if (!no_write) write_if_changed(outdir / (mod_name + "_table.ll"), table_ir);
        }

        double total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "%s: wrote %zu LLVM IR chunks (%s mode), %.1fs total\n",
                     mod_name.c_str(), chunks.size(),
                     no_opt_passes_mode ? "no-opt-passes" : (safe_passes_mode ? "safe-passes" : "all-passes"),
                     total_sec);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL [%s]: %s\n", mod_name.c_str(), e.what());
        return 1;
    }
    return 0;
}
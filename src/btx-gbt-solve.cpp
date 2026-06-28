// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/license/mit/.
//
// btx-gbt-solve: external-miner solver. Takes block-header fields as CLI flags,
// runs SolveMatMul once with a configurable max_tries budget, and prints a
// single JSON object on stdout describing the result. Designed to be called
// by an external miner orchestrator (Python) that polls getblocktemplate and
// builds the submitblock hex.

#include <arith_uint256.h>
#include <chainparams.h>
#include <common/args.h>
#include <cuda/matmul_accel.h>
#include <matmul/accelerated_solver.h>
#include <matmul/matmul_pow.h>
#include <matmul/matrix.h>
#include <pow.h>
#include <primitives/block.h>
#include <uint256.h>
#include <util/chaintype.h>
#include <util/strencodings.h>
#include <util/translation.h>

#include <univalue.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

const TranslateFn G_TRANSLATION_FUN{nullptr};

// Tip-change preempt (daemon mode): the wrapper sends SIGUSR1 when a new block
// (real parent change) supersedes the in-flight slice. The handler flips this
// flag; RunOneJob's watchdog bridges it into the per-job abort_flag so the
// current SolveMatMul returns within one scan window (~tens of ms) WITHOUT
// killing the daemon — the CUDA context + high GPU clock are preserved and the
// next job (new tip) starts immediately. async-signal-safe: only an atomic store.
std::atomic<bool> g_preempt{false};

extern "C" void BtxPreemptSignalHandler(int /*signum*/)
{
    g_preempt.store(true, std::memory_order_relaxed);
}

namespace {

constexpr uint32_t MAINNET_LIVE_LIKE_EPSILON_BITS{18U};

uint64_t CounterDelta(uint64_t before, uint64_t after)
{
    return after >= before ? after - before : after;
}

void SetEnvOverride(const char* name, const char* value)
{
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, /*overwrite=*/1);
#endif
}

void ClearEnvOverride(const char* name)
{
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

struct Options {
    int32_t version{0x20000000};
    uint256 prev_hash{};
    uint256 merkle_root{};
    uint32_t time{0};
    uint32_t bits{0};
    uint256 seed_a{};
    uint256 seed_b{};
    uint16_t matmul_n{512};
    uint32_t matmul_b{16};
    uint32_t matmul_r{8};
    uint32_t epsilon_bits{MAINNET_LIVE_LIKE_EPSILON_BITS};
    int32_t block_height{-1};
    // Parent block's median-time-past. Required for V3 seed derivation at
    // height >= nMatMulParentMtpSeedHeight (130,500); ignored below activation.
    // Supplied via --parent-mtp (one-shot) or job["parent_mtp"] (daemon).
    std::optional<int64_t> parent_mtp;
    uint64_t nonce_start{0};
    uint64_t max_tries{100'000'000ULL};
    // Optional 256-bit BE share target. When set the solver returns early
    // on digests <= share_target instead of the block-target derived from
    // --bits. header.nBits still uses --bits so the matmul digest stays
    // consensus-consistent — the lever pool miners need for share-tier
    // early-exit.
    std::optional<uint256> share_target;
    double max_seconds{0.0};  // 0 = no wall-clock cap
    std::optional<std::string> backend_override;
    std::optional<std::string> solver_threads_override;
    std::optional<std::string> batch_size_override;
    std::optional<std::string> async_override;
    std::optional<std::string> pool_slots_override;
    // Daemon mode: keep the process alive between jobs, read JSON job
    // payloads from stdin, emit one JSON result line per job on stdout.
    // Eliminates CUDA-context-init + cubin-load cost per slice (~5s of
    // 25s slice = 20% duty-cycle loss on the per-subprocess model).
    bool daemon_mode{false};
    // Diagnostic: print the consensus-derived seed_a/seed_b (V2 or V3 per
    // block_height + parent_mtp) for the given header, then exit. Lets us
    // bit-exact-validate the pool/Rust seed derivation against btxd's.
    bool emit_seeds{false};
};

class ScopedEnvOverride {
public:
    ScopedEnvOverride(const char* name, const std::optional<std::string>& value)
        : m_name{name}
    {
        const char* prev = std::getenv(name);
        if (prev != nullptr) {
            m_had_previous = true;
            m_previous = prev;
        }
        if (value.has_value()) {
            SetEnvOverride(name, value->c_str());
            m_set = true;
        }
    }
    ~ScopedEnvOverride() {
        if (!m_set) return;
        if (m_had_previous) {
            SetEnvOverride(m_name, m_previous.c_str());
        } else {
            ClearEnvOverride(m_name);
        }
    }
private:
    const char* m_name;
    bool m_had_previous{false};
    bool m_set{false};
    std::string m_previous;
};

std::optional<uint64_t> ParseUintArg(std::string_view text)
{
    try {
        size_t consumed{0};
        std::string value_text{text};
        int base{10};
        if (value_text.size() > 2 && value_text[0] == '0' &&
            (value_text[1] == 'x' || value_text[1] == 'X')) {
            base = 16;
        }
        const uint64_t value = std::stoull(value_text, &consumed, base);
        if (consumed != text.size()) return std::nullopt;
        return value;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

uint256 ParseUint256Hex(std::string_view hex, const char* arg_name)
{
    const auto parsed = uint256::FromHex(hex);
    if (!parsed.has_value()) {
        throw std::runtime_error(std::string("invalid uint256 for ") + arg_name);
    }
    return *parsed;
}

void PrintUsage(std::ostream& out)
{
    out << "Usage: btx-gbt-solve [flags]\n"
        << "Required flags (from getblocktemplate):\n"
        << "  --version <int>\n"
        << "  --prev-hash <hex64>\n"
        << "  --merkle-root <hex64>\n"
        << "  --time <uint32>\n"
        << "  --bits <hex compact, e.g. 1e02a876>\n"
        << "  --seed-a <hex64>\n"
        << "  --seed-b <hex64>\n"
        << "  --block-height <int>\n"
        << "Optional:\n"
        << "  --parent-mtp <int64>    parent block median-time-past; required at height >= 130500 (V3)\n"
        << "  --matmul-n <uint16>     default 512\n"
        << "  --matmul-b <uint32>     default 16\n"
        << "  --matmul-r <uint32>     default 8\n"
        << "  --epsilon-bits <uint32> default 18\n"
        << "  --share-target <hex64>  optional looser target for pool share-tier early-exit\n"
        << "  --nonce-start <uint64>  default 1\n"
        << "  --max-tries <uint64>    default 100,000,000\n"
        << "  --max-seconds <double>  default 0 (no cap)\n"
        << "  --backend <cpu|cuda|metal|mlx>\n"
        << "  --solver-threads <N>\n"
        << "  --batch-size <N>\n"
        << "  --async <0|1>\n"
        << "  --pool-slots <N>\n"
        << "  --daemon                stay alive, read JSON jobs from stdin\n"
        << "  --emit-seeds            print consensus-derived seed_a/seed_b (V2/V3) for the header and exit\n"
        << "Outputs ONE JSON line on stdout, see source for schema.\n"
        << "In --daemon mode, ignores required CLI fields; each stdin\n"
        << "line is a job: {version,prev_hash,merkle_root,time,bits,\n"
        << "share_target?,seed_a,seed_b,block_height,nonce_start,\n"
        << "max_tries,max_seconds}.\n";
}

bool ParseArgs(int argc, char* argv[], Options& options)
{
    auto need_value = [&](int& i, const std::string& arg) -> const char* {
        if (i + 1 >= argc) {
            std::cerr << "error: " << arg << " requires a value\n";
            return nullptr;
        }
        return argv[++i];
    };

    bool got_required[7] = {false, false, false, false, false, false, false};
    auto required_idx = [&](std::string_view name) -> int {
        if (name == "--version")      return 0;
        if (name == "--prev-hash")    return 1;
        if (name == "--merkle-root")  return 2;
        if (name == "--time")         return 3;
        if (name == "--bits")         return 4;
        if (name == "--seed-a")       return 5;
        if (name == "--seed-b")       return 6;
        return -1;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg{argv[i]};
        if (arg == "-h" || arg == "--help") { PrintUsage(std::cout); return false; }

        const char* val = nullptr;
        if (arg == "--version")           { val = need_value(i, arg); if (!val) return false; options.version = static_cast<int32_t>(*ParseUintArg(val)); got_required[required_idx(arg)] = true; }
        else if (arg == "--prev-hash")    { val = need_value(i, arg); if (!val) return false; options.prev_hash = ParseUint256Hex(val, "--prev-hash"); got_required[required_idx(arg)] = true; }
        else if (arg == "--merkle-root")  { val = need_value(i, arg); if (!val) return false; options.merkle_root = ParseUint256Hex(val, "--merkle-root"); got_required[required_idx(arg)] = true; }
        else if (arg == "--time")         { val = need_value(i, arg); if (!val) return false; options.time = static_cast<uint32_t>(*ParseUintArg(val)); got_required[required_idx(arg)] = true; }
        else if (arg == "--bits")         { val = need_value(i, arg); if (!val) return false; options.bits = static_cast<uint32_t>(*ParseUintArg(val)); got_required[required_idx(arg)] = true; }
        else if (arg == "--seed-a")       { val = need_value(i, arg); if (!val) return false; options.seed_a = ParseUint256Hex(val, "--seed-a"); got_required[required_idx(arg)] = true; }
        else if (arg == "--seed-b")       { val = need_value(i, arg); if (!val) return false; options.seed_b = ParseUint256Hex(val, "--seed-b"); got_required[required_idx(arg)] = true; }
        else if (arg == "--matmul-n")     { val = need_value(i, arg); if (!val) return false; options.matmul_n = static_cast<uint16_t>(*ParseUintArg(val)); }
        else if (arg == "--matmul-b")     { val = need_value(i, arg); if (!val) return false; options.matmul_b = static_cast<uint32_t>(*ParseUintArg(val)); }
        else if (arg == "--matmul-r")     { val = need_value(i, arg); if (!val) return false; options.matmul_r = static_cast<uint32_t>(*ParseUintArg(val)); }
        else if (arg == "--epsilon-bits") { val = need_value(i, arg); if (!val) return false; options.epsilon_bits = static_cast<uint32_t>(*ParseUintArg(val)); }
        else if (arg == "--share-target") { val = need_value(i, arg); if (!val) return false; options.share_target = ParseUint256Hex(val, "--share-target"); }
        else if (arg == "--block-height") { val = need_value(i, arg); if (!val) return false; options.block_height = static_cast<int32_t>(*ParseUintArg(val)); }
        else if (arg == "--parent-mtp")   { val = need_value(i, arg); if (!val) return false; options.parent_mtp = static_cast<int64_t>(*ParseUintArg(val)); }
        else if (arg == "--nonce-start")  { val = need_value(i, arg); if (!val) return false; options.nonce_start = *ParseUintArg(val); }
        else if (arg == "--max-tries")    { val = need_value(i, arg); if (!val) return false; options.max_tries = *ParseUintArg(val); }
        else if (arg == "--max-seconds")  { val = need_value(i, arg); if (!val) return false; options.max_seconds = std::stod(val); }
        else if (arg == "--backend")            { val = need_value(i, arg); if (!val) return false; options.backend_override = val; }
        else if (arg == "--solver-threads")     { val = need_value(i, arg); if (!val) return false; options.solver_threads_override = val; }
        else if (arg == "--batch-size")         { val = need_value(i, arg); if (!val) return false; options.batch_size_override = val; }
        else if (arg == "--async")              { val = need_value(i, arg); if (!val) return false; options.async_override = val; }
        else if (arg == "--pool-slots")         { val = need_value(i, arg); if (!val) return false; options.pool_slots_override = val; }
        else if (arg == "--daemon")             { options.daemon_mode = true; }
        else if (arg == "--emit-seeds")         { options.emit_seeds = true; }
        else {
            std::cerr << "error: unknown arg: " << arg << "\n";
            return false;
        }
    }

    // In daemon mode required fields come per-job via stdin JSON, not CLI.
    if (!options.daemon_mode) {
        for (int i = 0; i < 7; ++i) {
            if (!got_required[i]) {
                std::cerr << "error: missing required arg #" << i << "\n";
                PrintUsage(std::cerr);
                return false;
            }
        }
    }
    return true;
}

std::string HexBytesLE32(const std::vector<uint32_t>& v)
{
    // Each uint32 -> 4 LE bytes -> 8 hex chars. Total 8 * N hex chars.
    std::string out;
    out.reserve(v.size() * 8);
    static const char* hex = "0123456789abcdef";
    for (uint32_t value : v) {
        for (int byte = 0; byte < 4; ++byte) {
            uint8_t b = static_cast<uint8_t>((value >> (byte * 8)) & 0xFF);
            out.push_back(hex[b >> 4]);
            out.push_back(hex[b & 0xF]);
        }
    }
    return out;
}

matmul::accelerated::BackendRuntimeStats DeltaBackendRuntimeStats(
    const matmul::accelerated::BackendRuntimeStats& before,
    const matmul::accelerated::BackendRuntimeStats& after)
{
    matmul::accelerated::BackendRuntimeStats out;
    out.digest_requests = CounterDelta(before.digest_requests, after.digest_requests);
    out.requested_cpu = CounterDelta(before.requested_cpu, after.requested_cpu);
    out.requested_metal = CounterDelta(before.requested_metal, after.requested_metal);
    out.requested_cuda = CounterDelta(before.requested_cuda, after.requested_cuda);
    out.requested_unknown = CounterDelta(before.requested_unknown, after.requested_unknown);
    out.metal_successes = CounterDelta(before.metal_successes, after.metal_successes);
    out.metal_fallbacks_to_cpu = CounterDelta(before.metal_fallbacks_to_cpu, after.metal_fallbacks_to_cpu);
    out.metal_digest_mismatches = CounterDelta(before.metal_digest_mismatches, after.metal_digest_mismatches);
    out.metal_retry_without_uploaded_base_attempts = CounterDelta(before.metal_retry_without_uploaded_base_attempts, after.metal_retry_without_uploaded_base_attempts);
    out.metal_retry_without_uploaded_base_successes = CounterDelta(before.metal_retry_without_uploaded_base_successes, after.metal_retry_without_uploaded_base_successes);
    out.cuda_successes = CounterDelta(before.cuda_successes, after.cuda_successes);
    out.cuda_fallbacks_to_cpu = CounterDelta(before.cuda_fallbacks_to_cpu, after.cuda_fallbacks_to_cpu);
    out.gpu_input_generation_attempts = CounterDelta(before.gpu_input_generation_attempts, after.gpu_input_generation_attempts);
    out.gpu_input_generation_successes = CounterDelta(before.gpu_input_generation_successes, after.gpu_input_generation_successes);
    out.gpu_input_generation_failures = CounterDelta(before.gpu_input_generation_failures, after.gpu_input_generation_failures);
    out.gpu_input_auto_disabled_skips = CounterDelta(before.gpu_input_auto_disabled_skips, after.gpu_input_auto_disabled_skips);
    out.cuda_variable_base_batches = CounterDelta(before.cuda_variable_base_batches, after.cuda_variable_base_batches);
    out.cuda_variable_base_last_device_us = after.cuda_variable_base_last_device_us;
    out.cuda_variable_base_last_host_digest_us = after.cuda_variable_base_last_host_digest_us;
    out.cuda_variable_base_total_host_digest_us = CounterDelta(before.cuda_variable_base_total_host_digest_us, after.cuda_variable_base_total_host_digest_us);
    out.gpu_input_auto_disabled = after.gpu_input_auto_disabled;
    out.last_metal_fallback_error = after.last_metal_fallback_error;
    out.last_cuda_fallback_error = after.last_cuda_fallback_error;
    out.last_gpu_input_error = after.last_gpu_input_error;
    return out;
}

UniValue BuildBackendRuntimeStatsObject(const matmul::accelerated::BackendRuntimeStats& stats)
{
    UniValue obj(UniValue::VOBJ);
    obj.pushKV("digest_requests", stats.digest_requests);
    obj.pushKV("requested_cpu", stats.requested_cpu);
    obj.pushKV("requested_metal", stats.requested_metal);
    obj.pushKV("requested_cuda", stats.requested_cuda);
    obj.pushKV("requested_unknown", stats.requested_unknown);
    obj.pushKV("metal_successes", stats.metal_successes);
    obj.pushKV("metal_fallbacks_to_cpu", stats.metal_fallbacks_to_cpu);
    obj.pushKV("metal_digest_mismatches", stats.metal_digest_mismatches);
    obj.pushKV("metal_retry_without_uploaded_base_attempts", stats.metal_retry_without_uploaded_base_attempts);
    obj.pushKV("metal_retry_without_uploaded_base_successes", stats.metal_retry_without_uploaded_base_successes);
    obj.pushKV("cuda_successes", stats.cuda_successes);
    obj.pushKV("cuda_fallbacks_to_cpu", stats.cuda_fallbacks_to_cpu);
    obj.pushKV("gpu_input_generation_attempts", stats.gpu_input_generation_attempts);
    obj.pushKV("gpu_input_generation_successes", stats.gpu_input_generation_successes);
    obj.pushKV("gpu_input_generation_failures", stats.gpu_input_generation_failures);
    obj.pushKV("gpu_input_auto_disabled_skips", stats.gpu_input_auto_disabled_skips);
    obj.pushKV("gpu_input_auto_disabled", stats.gpu_input_auto_disabled);
    obj.pushKV("cuda_variable_base_batches", stats.cuda_variable_base_batches);
    obj.pushKV("cuda_variable_base_last_device_us", stats.cuda_variable_base_last_device_us);
    obj.pushKV("cuda_variable_base_last_host_digest_us", stats.cuda_variable_base_last_host_digest_us);
    obj.pushKV("cuda_variable_base_total_host_digest_us", stats.cuda_variable_base_total_host_digest_us);
    obj.pushKV("last_metal_fallback_error", stats.last_metal_fallback_error);
    obj.pushKV("last_cuda_fallback_error", stats.last_cuda_fallback_error);
    obj.pushKV("last_gpu_input_error", stats.last_gpu_input_error);
    return obj;
}

UniValue BuildCudaProfilingStatsObject(const btx::cuda::MatMulProfilingStats& stats)
{
    UniValue obj(UniValue::VOBJ);
    obj.pushKV("available", stats.available);
    obj.pushKV("samples", stats.samples);
    obj.pushKV("last_n", stats.last_n);
    obj.pushKV("last_b", stats.last_b);
    obj.pushKV("last_r", stats.last_r);
    obj.pushKV("last_batch_size", stats.last_batch_size);
    obj.pushKV("last_host_stage_us", stats.last_host_stage_us);
    obj.pushKV("last_submit_h2d_us", stats.last_submit_h2d_us);
    obj.pushKV("last_submit_d2d_us", stats.last_submit_d2d_us);
    obj.pushKV("last_stream_wait_event_us", stats.last_stream_wait_event_us);
    obj.pushKV("last_launch_build_perturbed_us", stats.last_launch_build_perturbed_us);
    obj.pushKV("last_launch_finalize_us", stats.last_launch_finalize_us);
    obj.pushKV("last_submit_d2h_us", stats.last_submit_d2h_us);
    obj.pushKV("last_stream_sync_us", stats.last_stream_sync_us);
    obj.pushKV("last_total_wall_ms", stats.last_total_wall_ms);
    obj.pushKV("last_event_build_a_device_us", stats.last_event_build_a_device_us);
    obj.pushKV("last_event_build_b_midstate_device_us", stats.last_event_build_b_midstate_device_us);
    obj.pushKV("last_event_rhs_device_us", stats.last_event_rhs_device_us);
    obj.pushKV("last_event_words_device_us", stats.last_event_words_device_us);
    obj.pushKV("last_event_digest_device_us", stats.last_event_digest_device_us);
    obj.pushKV("last_used_low_rank_path", stats.last_used_low_rank_path);
    obj.pushKV("last_used_device_prepared_inputs", stats.last_used_device_prepared_inputs);
    obj.pushKV("last_used_pinned_host_staging", stats.last_used_pinned_host_staging);
    obj.pushKV("last_base_matrix_cache_hit", stats.last_base_matrix_cache_hit);
    obj.pushKV("last_mode", stats.last_mode);
    obj.pushKV("reason", stats.reason);
    return obj;
}

} // namespace

// Run one solve against the parameters in `options`. Writes one JSON
// result line to stdout. Returns whether a solution (share-target or block-
// target) was found. Used by both one-shot and daemon-mode paths.
bool RunOneJob(Options& options, const Consensus::Params& consensus)
{
    CBlockHeader header{};
    header.nVersion = options.version;
    header.hashPrevBlock = options.prev_hash;
    header.hashMerkleRoot = options.merkle_root;
    header.nTime = options.time;
    header.nBits = options.bits;
    header.nNonce64 = options.nonce_start;
    header.nNonce = static_cast<uint32_t>(options.nonce_start);
    header.matmul_dim = options.matmul_n;
    header.seed_a = options.seed_a;
    header.seed_b = options.seed_b;
    header.matmul_digest.SetNull();

    if (options.emit_seeds) {
        // Diagnostic: derive seeds via btxd's own consensus function (V2 or V3
        // per block_height + parent_mtp) and print them, then exit. Used to
        // bit-exact-validate the pool/Rust seed derivation against consensus.
        CBlockHeader h{header};
        const bool ok = SetDeterministicMatMulSeeds(h, consensus, options.block_height, options.parent_mtp);
        UniValue out(UniValue::VOBJ);
        out.pushKV("ok", ok);
        out.pushKV("block_height", options.block_height);
        out.pushKV("nonce64", static_cast<uint64_t>(options.nonce_start));
        out.pushKV("parent_mtp", options.parent_mtp.has_value() ? UniValue(static_cast<int64_t>(*options.parent_mtp)) : UniValue());
        out.pushKV("seed_a", h.seed_a.GetHex());
        out.pushKV("seed_b", h.seed_b.GetHex());
        std::cout << out.write() << std::endl;
        return ok;
    }

    std::atomic<bool> abort_flag{false};
    std::vector<uint32_t> matrix_c_data;

    // Clear any preempt that arrived between jobs (e.g. a SIGUSR1 racing the
    // stdin read) so it cannot abort THIS job before it starts. From here on a
    // SIGUSR1 aborts the in-flight solve.
    g_preempt.store(false, std::memory_order_relaxed);

    // Watchdog bridges (a) the max_seconds deadline and (b) a tip-change
    // preempt (SIGUSR1 -> g_preempt) into the per-job abort_flag. Run it in
    // daemon mode even without a deadline so preempt still works.
    std::thread watchdog;
    const bool run_watchdog = options.max_seconds > 0.0 || options.daemon_mode;
    if (run_watchdog) {
        watchdog = std::thread([&abort_flag, max_seconds = options.max_seconds]() {
            const bool have_deadline = max_seconds > 0.0;
            const auto deadline = std::chrono::steady_clock::now() +
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(have_deadline ? max_seconds : 0.0));
            while (!have_deadline || std::chrono::steady_clock::now() < deadline) {
                if (abort_flag.load(std::memory_order_relaxed)) return;
                if (g_preempt.load(std::memory_order_relaxed)) {  // tip-change preempt
                    abort_flag.store(true, std::memory_order_relaxed);
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            abort_flag.store(true, std::memory_order_relaxed);
        });
    }

    const auto start = std::chrono::steady_clock::now();
    const auto backend_runtime_before = matmul::accelerated::ProbeMatMulBackendRuntimeStats();
    uint64_t tries_budget = options.max_tries;
    const uint256* share_target_ptr = options.share_target ? &(*options.share_target) : nullptr;
    const auto block_target = DeriveTarget(options.bits, consensus.powLimit);

    // C4 fix: collect ALL shares in this slice instead of returning on the
    // first one. Looping SolveMatMul *inside* the daemon (no host round-trip /
    // stdin read between shares) keeps the GPU continuously fed — that is the
    // saturation fix. Bounded by the max_seconds watchdog (abort_flag),
    // max_tries, or nonce64 overflow. Per-share matrix_c is emitted only for
    // block-tier hits (share-tier submits are digest-only, per the protocol).
    UniValue shares(UniValue::VARR);
    bool have_first = false;
    uint64_t first_nonce = 0;
    std::string first_digest;
    bool first_is_block = false;
    std::string first_matrix_hex;
    uint64_t first_matrix_words = 0;
    while (!abort_flag.load(std::memory_order_relaxed) && tries_budget > 0) {
        matrix_c_data.clear();
        header.matmul_digest.SetNull();
        const bool found_one = SolveMatMul(header, consensus, tries_budget,
                                           options.block_height, &abort_flag,
                                           &matrix_c_data, share_target_ptr,
                                           options.parent_mtp);
        if (!found_one) break;
        const std::string dg = header.matmul_digest.GetHex();
        const bool is_block = block_target
            ? UintToArith256(header.matmul_digest) <= *block_target
            : false;
        std::string mhex;
        uint64_t mwords = 0;
        if (is_block) {
            mhex = HexBytesLE32(matrix_c_data);
            mwords = static_cast<uint64_t>(matrix_c_data.size());
        }
        UniValue sh(UniValue::VOBJ);
        sh.pushKV("nonce64", header.nNonce64);
        sh.pushKV("matmul_digest", dg);
        sh.pushKV("is_block", is_block);
        if (is_block) {
            sh.pushKV("matrix_c_data_hex", mhex);
            sh.pushKV("matrix_c_data_words", mwords);
        }
        shares.push_back(sh);
        if (!have_first) {
            have_first = true;
            first_nonce = header.nNonce64;
            first_digest = dg;
            first_is_block = is_block;
            first_matrix_hex = mhex;
            first_matrix_words = mwords;
        }
        if (header.nNonce64 == UINT64_MAX) break;
        header.nNonce64 += 1;
    }
    const auto stop = std::chrono::steady_clock::now();
    const auto backend_runtime_after = matmul::accelerated::ProbeMatMulBackendRuntimeStats();
    const auto backend_runtime_delta = DeltaBackendRuntimeStats(backend_runtime_before, backend_runtime_after);
    const auto cuda_profile = btx::cuda::ProbeMatMulProfilingStats();
    const double elapsed_s = std::chrono::duration<double>(stop - start).count();
    const uint64_t tries_used = options.max_tries - tries_budget;

    abort_flag.store(true, std::memory_order_relaxed);
    if (watchdog.joinable()) watchdog.join();

    const bool was_preempted = g_preempt.load(std::memory_order_relaxed);
    const size_t n_shares = shares.size();
    UniValue out(UniValue::VOBJ);
    out.pushKV("found", n_shares > 0);
    out.pushKV("share_count", static_cast<uint64_t>(n_shares));
    out.pushKV("shares", shares);
    out.pushKV("tries_used", tries_used);
    out.pushKV("elapsed_s", elapsed_s);
    out.pushKV("nonce64_end", header.nNonce64);
    out.pushKV("preempted", was_preempted);  // tip-change SIGUSR1 cut this slice short
    out.pushKV("backend_runtime", BuildBackendRuntimeStatsObject(backend_runtime_delta));
    out.pushKV("cuda_profile", BuildCudaProfilingStatsObject(cuda_profile));
    if (n_shares > 0) {
        // Backward-compat: surface the FIRST share at top level so existing
        // one-share wrappers keep working unchanged.
        out.pushKV("nonce64", first_nonce);
        out.pushKV("matmul_digest", first_digest);
        out.pushKV("is_block", first_is_block);
        if (first_is_block) {
            out.pushKV("matrix_c_data_hex", first_matrix_hex);
            out.pushKV("matrix_c_data_words", first_matrix_words);
        }
    } else {
        std::string reason{"max_tries_exhausted"};
        if (abort_flag.load() && options.max_seconds > 0.0 && elapsed_s >= options.max_seconds) {
            reason = "max_seconds_exceeded";
        }
        out.pushKV("reason", reason);
    }

    std::cout << out.write() << std::endl;
    return n_shares > 0;
}

// Parse a JSON job payload (one line, per the protocol in PrintUsage) and
// overwrite per-job fields of `options`. Daemon mode only. Throws on bad
// uint256 hex (caught at caller, emitted as error JSON). Returns false on
// malformed JSON envelope.
bool UpdateJobFromJson(const std::string& line, Options& options)
{
    UniValue job;
    if (!job.read(line) || !job.isObject()) return false;
    if (job.exists("version")) options.version = static_cast<int32_t>(job["version"].getInt<int64_t>());
    if (job.exists("prev_hash"))    options.prev_hash    = ParseUint256Hex(job["prev_hash"].get_str(), "prev_hash");
    if (job.exists("merkle_root"))  options.merkle_root  = ParseUint256Hex(job["merkle_root"].get_str(), "merkle_root");
    if (job.exists("time"))         options.time         = static_cast<uint32_t>(job["time"].getInt<int64_t>());
    if (job.exists("bits")) {
        const std::string bs = job["bits"].get_str();
        const std::string h  = (bs.rfind("0x", 0) == 0 || bs.rfind("0X", 0) == 0) ? bs.substr(2) : bs;
        options.bits = static_cast<uint32_t>(std::stoull(h, nullptr, 16));
    }
    if (job.exists("seed_a"))       options.seed_a       = ParseUint256Hex(job["seed_a"].get_str(), "seed_a");
    if (job.exists("seed_b"))       options.seed_b       = ParseUint256Hex(job["seed_b"].get_str(), "seed_b");
    if (job.exists("block_height")) options.block_height = static_cast<int32_t>(job["block_height"].getInt<int64_t>());
    // V3 parent-MTP (height >= 130500). Reset when absent so a stale value from
    // a prior job can never leak into the next one's seed derivation.
    if (job.exists("parent_mtp")) options.parent_mtp = job["parent_mtp"].getInt<int64_t>();
    else options.parent_mtp.reset();
    if (job.exists("nonce_start"))  options.nonce_start  = static_cast<uint64_t>(job["nonce_start"].getInt<int64_t>());
    if (job.exists("max_tries"))    options.max_tries    = static_cast<uint64_t>(job["max_tries"].getInt<int64_t>());
    if (job.exists("max_seconds"))  options.max_seconds  = job["max_seconds"].get_real();
    if (job.exists("share_target") && job["share_target"].isStr() && !job["share_target"].get_str().empty()) {
        options.share_target = ParseUint256Hex(job["share_target"].get_str(), "share_target");
    } else {
        options.share_target.reset();
    }
    return true;
}

int main(int argc, char* argv[])
{
    Options options;
    try {
        if (!ParseArgs(argc, argv, options)) return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    // Apply env overrides ONCE for the lifetime of this process. In daemon
    // mode these stick across all jobs so the matmul backend (CUDA in
    // particular) keeps its initialized context + loaded cubins.
    ScopedEnvOverride backend_env("BTX_MATMUL_BACKEND", options.backend_override);
    ScopedEnvOverride solver_threads_env("BTX_MATMUL_SOLVER_THREADS", options.solver_threads_override);
    ScopedEnvOverride batch_env("BTX_MATMUL_SOLVE_BATCH_SIZE", options.batch_size_override);
    ScopedEnvOverride async_env("BTX_MATMUL_PIPELINE_ASYNC", options.async_override);
    ScopedEnvOverride pool_env("BTX_MATMUL_CUDA_POOL_SLOTS", options.pool_slots_override);
    ScopedEnvOverride pool_env_metal("BTX_MATMUL_METAL_POOL_SLOTS", options.pool_slots_override);

    ArgsManager args;
    auto consensus = CreateChainParams(args, ChainType::REGTEST)->GetConsensus();
    consensus.fMatMulPOW = true;
    consensus.nMatMulDimension = options.matmul_n;
    consensus.nMatMulTranscriptBlockSize = options.matmul_b;
    consensus.nMatMulNoiseRank = options.matmul_r;
    consensus.nMatMulPreHashEpsilonBits = options.epsilon_bits;
    consensus.powLimit = uint256{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"};
    // Mainnet MatMul nonce-seed V2 activation. At height >= this, SolveMatMul
    // routes through SolveMatMulNonceSeeded which re-derives per-nonce seed_a/
    // seed_b from the (mutable) header via DeterministicMatMulSeedV2. Without
    // this, the miner would use pool-provided static seeds post-fork and the
    // pool's per-nonce digest recomputation would mismatch every share.
    consensus.nMatMulNonceSeedHeight = 125'000;
    // Mainnet MatMul V3 parent-MTP-seed activation (BTX v0.32.10, height 130,500).
    // At/above this, SolveMatMul derives seeds via DeterministicMatMulSeedV3 which
    // additionally binds the parent block's median-time-past (supplied per job via
    // --parent-mtp / job["parent_mtp"]). Below it, behavior is unchanged (V2). Keep
    // this EXACTLY at 130,500 — activating early would invalidate pre-fork blocks.
    consensus.nMatMulParentMtpSeedHeight = 130'500;

    if (!options.daemon_mode) {
        // One-shot path: run a single job from the CLI-parsed options, exit.
        const bool found = RunOneJob(options, consensus);
        return found ? 0 : 2;
    }

    // Tip-change preempt: SIGUSR1 from the wrapper aborts the in-flight slice
    // (see g_preempt). Default SIGUSR1 action terminates the process, so we must
    // install a handler where the platform provides it.
#ifdef SIGUSR1
    std::signal(SIGUSR1, BtxPreemptSignalHandler);
    const bool supports_signal_preempt{true};
#else
    const bool supports_signal_preempt{false};
#endif

    // Daemon mode: emit a ready marker on stderr (cheap handshake for the
    // wrapper), then read one JSON job per line on stdin. Each job updates
    // the per-job fields of `options` and runs RunOneJob, which emits one
    // JSON line per job on stdout. EOF on stdin terminates cleanly.
    std::cerr << "{\"event\":\"daemon_ready\",\"preempt\":"
              << (supports_signal_preempt ? "true" : "false")
              << "}" << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        try {
            if (!UpdateJobFromJson(line, options)) {
                UniValue err(UniValue::VOBJ);
                err.pushKV("error", "bad job json envelope");
                std::cout << err.write() << std::endl;
                std::cout.flush();
                continue;
            }
            RunOneJob(options, consensus);
        } catch (const std::exception& e) {
            UniValue err(UniValue::VOBJ);
            err.pushKV("error", std::string("job exception: ") + e.what());
            std::cout << err.write() << std::endl;
        }
        std::cout.flush();
    }
    return 0;
}

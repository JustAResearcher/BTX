// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/license/mit/.

#include <arith_uint256.h>
#include <chainparams.h>
#include <common/args.h>
#include <cuda/matmul_accel.h>
#include <cuda/oracle_accel.h>
#include <matmul/backend_capabilities.h>
#include <matmul/accelerated_solver.h>
#include <metal/matmul_accel.h>
#include <pow.h>
#include <primitives/block.h>
#include <uint256.h>
#include <util/chaintype.h>
#include <util/translation.h>

#include <univalue.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

const TranslateFn G_TRANSLATION_FUN{nullptr};

namespace {
constexpr uint32_t MAINNET_POST_PRODUCT_HEIGHT{61'000U};
constexpr uint32_t MAINNET_LIVE_LIKE_EPSILON_BITS{18U};
constexpr uint32_t MAINNET_LIVE_LIKE_NBITS{0x1e063c74U};

struct Options {
    uint32_t iterations{8};
    uint32_t warmup_iterations{0};
    uint64_t max_tries{2048};
    uint32_t n{512};
    uint32_t b{16};
    uint32_t r{8};
    uint32_t nbits{MAINNET_LIVE_LIKE_NBITS};
    uint32_t epsilon_bits{MAINNET_LIVE_LIKE_EPSILON_BITS};
    int32_t block_height{static_cast<int32_t>(MAINNET_POST_PRODUCT_HEIGHT)};
    std::optional<int32_t> nonce_seed_height_override;
    std::optional<int32_t> parent_mtp_seed_height_override;
    std::optional<int64_t> parent_mtp_override;
    std::optional<int32_t> product_digest_height_override;
    uint32_t parallel{1};
    std::optional<std::string> backend_override;
    std::optional<std::string> async_override;
    std::optional<std::string> gpu_inputs_override;
    std::optional<std::string> batch_size_override;
    std::optional<std::string> digest_slice_size_override;
    std::optional<std::string> prefetch_depth_override;
    std::optional<std::string> prepare_workers_override;
    std::optional<std::string> pool_slots_override;
    std::optional<std::string> solver_threads_override;
    bool per_iteration{false};
};

std::optional<uint64_t> ParseUintArg(std::string_view text)
{
    try {
        size_t consumed{0};
        std::string value_text{text};
        int base{10};
        if (value_text.size() > 2 &&
            value_text[0] == '0' &&
            (value_text[1] == 'x' || value_text[1] == 'X')) {
            base = 16;
        }
        const uint64_t value = std::stoull(value_text, &consumed, base);
        if (consumed != text.size()) {
            return std::nullopt;
        }
        return value;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

uint256 ParseUint256(std::string_view hex)
{
    const auto parsed = uint256::FromHex(hex);
    if (!parsed.has_value()) {
        throw std::runtime_error("invalid uint256 literal in matmul solve benchmark");
    }
    return *parsed;
}

void PrintUsage(std::ostream& out)
{
    out << "Usage: btx-matmul-solve-bench"
        << " [--iterations <count>] [--warmup <count>] [--tries <count>]"
        << " [--n <dim>] [--b <block>] [--r <rank>]"
        << " [--nbits <compact>] [--epsilon-bits <count>]"
        << " [--block-height <height>] [--nonce-seed-height <height>]"
        << " [--parent-mtp-seed-height <height>] [--parent-mtp <time>]"
        << " [--product-digest-height <height>]"
        << " [--parallel <count>]"
        << " [--backend <cpu|metal|cuda|mlx>]"
        << " [--async <0|1>] [--gpu-inputs <0|1>]"
        << " [--batch-size <count>] [--digest-slice-size <count>] [--prefetch-depth <count>] [--prepare-workers <count>]"
        << " [--pool-slots <count>] [--solver-threads <count>]"
        << " [--per-iteration]" << std::endl;
}

bool ParseArgs(int argc, char* argv[], Options& options)
{
    auto parse_uint32 = [&](std::string_view arg_name, std::string_view value, uint32_t& out) -> bool {
        const auto parsed = ParseUintArg(value);
        if (!parsed.has_value() || *parsed == 0 || *parsed > std::numeric_limits<uint32_t>::max()) {
            std::cerr << "error: invalid value for " << arg_name << ": " << value << std::endl;
            return false;
        }
        out = static_cast<uint32_t>(*parsed);
        return true;
    };
    auto parse_uint32_allow_zero = [&](std::string_view arg_name, std::string_view value, uint32_t& out) -> bool {
        const auto parsed = ParseUintArg(value);
        if (!parsed.has_value() || *parsed > std::numeric_limits<uint32_t>::max()) {
            std::cerr << "error: invalid value for " << arg_name << ": " << value << std::endl;
            return false;
        }
        out = static_cast<uint32_t>(*parsed);
        return true;
    };

    auto parse_uint64 = [&](std::string_view arg_name, std::string_view value, uint64_t& out) -> bool {
        const auto parsed = ParseUintArg(value);
        if (!parsed.has_value() || *parsed == 0) {
            std::cerr << "error: invalid value for " << arg_name << ": " << value << std::endl;
            return false;
        }
        out = *parsed;
        return true;
    };

    auto parse_int32 = [&](std::string_view arg_name, std::string_view value, int32_t& out) -> bool {
        try {
            size_t consumed{0};
            const long parsed = std::stol(std::string{value}, &consumed, 10);
            if (consumed != value.size() ||
                parsed < std::numeric_limits<int32_t>::min() ||
                parsed > std::numeric_limits<int32_t>::max()) {
                std::cerr << "error: invalid value for " << arg_name << ": " << value << std::endl;
                return false;
            }
            out = static_cast<int32_t>(parsed);
            return true;
        } catch (const std::exception&) {
            std::cerr << "error: invalid value for " << arg_name << ": " << value << std::endl;
            return false;
        }
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg{argv[i]};
        if (arg == "--help" || arg == "-h") {
            PrintUsage(std::cout);
            return false;
        }

        auto parse_kv = [&](std::string_view name, auto&& setter) -> bool {
            const std::string prefix = std::string{name} + "=";
            if (arg.rfind(prefix, 0) == 0) {
                return setter(std::string_view{arg}.substr(prefix.size()));
            }
            if (arg == name) {
                if (i + 1 >= argc) {
                    std::cerr << "error: " << name << " requires a value" << std::endl;
                    return false;
                }
                return setter(argv[++i]);
            }
            return true;
        };

        bool consumed = false;
        if (arg == "--iterations" || arg.rfind("--iterations=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--iterations", [&](std::string_view value) { return parse_uint32("--iterations", value, options.iterations); })) return false;
        } else if (arg == "--warmup" || arg.rfind("--warmup=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--warmup", [&](std::string_view value) { return parse_uint32_allow_zero("--warmup", value, options.warmup_iterations); })) return false;
        } else if (arg == "--tries" || arg.rfind("--tries=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--tries", [&](std::string_view value) { return parse_uint64("--tries", value, options.max_tries); })) return false;
        } else if (arg == "--n" || arg.rfind("--n=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--n", [&](std::string_view value) { return parse_uint32("--n", value, options.n); })) return false;
        } else if (arg == "--b" || arg.rfind("--b=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--b", [&](std::string_view value) { return parse_uint32("--b", value, options.b); })) return false;
        } else if (arg == "--r" || arg.rfind("--r=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--r", [&](std::string_view value) { return parse_uint32("--r", value, options.r); })) return false;
        } else if (arg == "--nbits" || arg.rfind("--nbits=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--nbits", [&](std::string_view value) { return parse_uint32("--nbits", value, options.nbits); })) return false;
        } else if (arg == "--epsilon-bits" || arg.rfind("--epsilon-bits=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--epsilon-bits", [&](std::string_view value) { return parse_uint32_allow_zero("--epsilon-bits", value, options.epsilon_bits); })) return false;
        } else if (arg == "--block-height" || arg.rfind("--block-height=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--block-height", [&](std::string_view value) { return parse_int32("--block-height", value, options.block_height); })) return false;
        } else if (arg == "--nonce-seed-height" || arg.rfind("--nonce-seed-height=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--nonce-seed-height", [&](std::string_view value) {
                    int32_t parsed{0};
                    if (!parse_int32("--nonce-seed-height", value, parsed)) return false;
                    if (parsed < 0) {
                        std::cerr << "error: invalid value for --nonce-seed-height: " << value << std::endl;
                        return false;
                    }
                    options.nonce_seed_height_override = parsed;
                    return true;
                })) return false;
        } else if (arg == "--parent-mtp-seed-height" || arg.rfind("--parent-mtp-seed-height=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--parent-mtp-seed-height", [&](std::string_view value) {
                    int32_t parsed{0};
                    if (!parse_int32("--parent-mtp-seed-height", value, parsed)) return false;
                    if (parsed < 0) {
                        std::cerr << "error: invalid value for --parent-mtp-seed-height: " << value << std::endl;
                        return false;
                    }
                    options.parent_mtp_seed_height_override = parsed;
                    return true;
                })) return false;
        } else if (arg == "--parent-mtp" || arg.rfind("--parent-mtp=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--parent-mtp", [&](std::string_view value) {
                    try {
                        size_t consumed_chars{0};
                        const long long parsed = std::stoll(std::string{value}, &consumed_chars, 10);
                        if (consumed_chars != value.size()) {
                            std::cerr << "error: invalid value for --parent-mtp: " << value << std::endl;
                            return false;
                        }
                        options.parent_mtp_override = static_cast<int64_t>(parsed);
                        return true;
                    } catch (const std::exception&) {
                        std::cerr << "error: invalid value for --parent-mtp: " << value << std::endl;
                        return false;
                    }
                })) return false;
        } else if (arg == "--product-digest-height" || arg.rfind("--product-digest-height=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--product-digest-height", [&](std::string_view value) {
                    int32_t parsed{0};
                    if (!parse_int32("--product-digest-height", value, parsed)) return false;
                    if (parsed < 0) {
                        std::cerr << "error: invalid value for --product-digest-height: " << value << std::endl;
                        return false;
                    }
                    options.product_digest_height_override = parsed;
                    return true;
                })) return false;
        } else if (arg == "--parallel" || arg.rfind("--parallel=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--parallel", [&](std::string_view value) { return parse_uint32("--parallel", value, options.parallel); })) return false;
        } else if (arg == "--backend" || arg.rfind("--backend=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--backend", [&](std::string_view value) {
                    options.backend_override = std::string{value};
                    return true;
                })) return false;
        } else if (arg == "--async" || arg.rfind("--async=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--async", [&](std::string_view value) {
                    options.async_override = std::string{value};
                    return true;
                })) return false;
        } else if (arg == "--gpu-inputs" || arg.rfind("--gpu-inputs=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--gpu-inputs", [&](std::string_view value) {
                    options.gpu_inputs_override = std::string{value};
                    return true;
                })) return false;
        } else if (arg == "--batch-size" || arg.rfind("--batch-size=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--batch-size", [&](std::string_view value) {
                    options.batch_size_override = std::string{value};
                    return true;
                })) return false;
        } else if (arg == "--digest-slice-size" || arg.rfind("--digest-slice-size=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--digest-slice-size", [&](std::string_view value) {
                    options.digest_slice_size_override = std::string{value};
                    return true;
                })) return false;
        } else if (arg == "--prefetch-depth" || arg.rfind("--prefetch-depth=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--prefetch-depth", [&](std::string_view value) {
                    options.prefetch_depth_override = std::string{value};
                    return true;
                })) return false;
        } else if (arg == "--prepare-workers" || arg.rfind("--prepare-workers=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--prepare-workers", [&](std::string_view value) {
                    options.prepare_workers_override = std::string{value};
                    return true;
                })) return false;
        } else if (arg == "--pool-slots" || arg.rfind("--pool-slots=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--pool-slots", [&](std::string_view value) {
                    options.pool_slots_override = std::string{value};
                    return true;
                })) return false;
        } else if (arg == "--solver-threads" || arg.rfind("--solver-threads=", 0) == 0) {
            consumed = true;
            if (!parse_kv("--solver-threads", [&](std::string_view value) {
                    options.solver_threads_override = std::string{value};
                    return true;
                })) return false;
        } else if (arg == "--per-iteration") {
            consumed = true;
            options.per_iteration = true;
        }

        if (!consumed) {
            std::cerr << "error: unknown argument: " << arg << std::endl;
            PrintUsage(std::cerr);
            return false;
        }
    }

    return true;
}

class ScopedEnvOverride
{
public:
    ScopedEnvOverride(const char* name, const std::optional<std::string>& value) : m_name(name)
    {
        const char* current = std::getenv(name);
        if (current != nullptr) {
            m_had_original = true;
            m_original = current;
        }
        if (!value.has_value()) {
            return;
        }
        m_applied = true;
#if defined(WIN32)
        _putenv_s(name, value->c_str());
#else
        setenv(name, value->c_str(), 1);
#endif
    }

    ~ScopedEnvOverride()
    {
        if (!m_applied) {
            return;
        }
#if defined(WIN32)
        _putenv_s(m_name, m_had_original ? m_original.c_str() : "");
#else
        if (m_had_original) {
            setenv(m_name, m_original.c_str(), 1);
        } else {
            unsetenv(m_name);
        }
#endif
    }

private:
    const char* m_name;
    bool m_applied{false};
    bool m_had_original{false};
    std::string m_original;
};

double Mean(const std::vector<double>& values)
{
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double Median(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if ((values.size() & 1U) == 0U) {
        return (values[mid - 1] + values[mid]) / 2.0;
    }
    return values[mid];
}

UniValue SummarizeSeries(const std::vector<double>& values)
{
    UniValue out(UniValue::VOBJ);
    out.pushKV("count", static_cast<uint64_t>(values.size()));
    if (values.empty()) {
        out.pushKV("mean", 0.0);
        out.pushKV("median", 0.0);
        out.pushKV("min", 0.0);
        out.pushKV("max", 0.0);
        return out;
    }

    const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    out.pushKV("mean", Mean(values));
    out.pushKV("median", Median(values));
    out.pushKV("min", *min_it);
    out.pushKV("max", *max_it);
    return out;
}

double PerSecond(uint64_t count, double elapsed_s)
{
    return elapsed_s > 0.0 ? static_cast<double>(count) / elapsed_s : 0.0;
}

uint64_t CounterDelta(uint64_t before, uint64_t after)
{
    return after >= before ? after - before : after;
}

MatMulGpuPreHashScanStats DeltaGpuPreHashScanStats(const MatMulGpuPreHashScanStats& before,
                                                   const MatMulGpuPreHashScanStats& after)
{
    MatMulGpuPreHashScanStats out;
    out.attempts = CounterDelta(before.attempts, after.attempts);
    out.successes = CounterDelta(before.successes, after.successes);
    out.failures = CounterDelta(before.failures, after.failures);
    out.metal_fallbacks_to_cpu = CounterDelta(before.metal_fallbacks_to_cpu, after.metal_fallbacks_to_cpu);
    out.cuda_fallbacks_to_cpu = CounterDelta(before.cuda_fallbacks_to_cpu, after.cuda_fallbacks_to_cpu);
    out.total_elapsed_us = CounterDelta(before.total_elapsed_us, after.total_elapsed_us);
    out.last_elapsed_us = after.last_elapsed_us;
    out.max_elapsed_us = after.max_elapsed_us;
    out.total_scanned_count = CounterDelta(before.total_scanned_count, after.total_scanned_count);
    out.last_scanned_count = after.last_scanned_count;
    out.total_pass_flags = CounterDelta(before.total_pass_flags, after.total_pass_flags);
    out.last_pass_flags = after.last_pass_flags;
    out.total_selected_headers = CounterDelta(before.total_selected_headers, after.total_selected_headers);
    out.last_selected_headers = after.last_selected_headers;
    out.last_scan_limit = after.last_scan_limit;
    out.last_backend = after.last_backend;
    out.last_error = after.last_error;
    return out;
}

MatMulDigestCompareStats DeltaDigestCompareStats(const MatMulDigestCompareStats& before,
                                                 const MatMulDigestCompareStats& after)
{
    MatMulDigestCompareStats out;
    out.enabled = after.enabled;
    out.compared_attempts = CounterDelta(before.compared_attempts, after.compared_attempts);
    out.first_divergence_captured = after.first_divergence_captured && !before.first_divergence_captured;
    if (out.first_divergence_captured) {
        out.first_divergence_nonce64 = after.first_divergence_nonce64;
        out.first_divergence_nonce32 = after.first_divergence_nonce32;
        out.first_divergence_header_hash = after.first_divergence_header_hash;
        out.first_divergence_backend_digest = after.first_divergence_backend_digest;
        out.first_divergence_cpu_digest = after.first_divergence_cpu_digest;
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

UniValue BuildPipelineStatsObject(const MatMulSolvePipelineStats& stats)
{
    UniValue obj(UniValue::VOBJ);
    obj.pushKV("parallel_solver_enabled", stats.parallel_solver_enabled);
    obj.pushKV("parallel_solver_threads", stats.parallel_solver_threads);
    obj.pushKV("async_prepare_enabled", stats.async_prepare_enabled);
    obj.pushKV("cpu_confirm_candidates", stats.cpu_confirm_candidates);
    obj.pushKV("prepared_inputs", stats.prepared_inputs);
    obj.pushKV("overlapped_prepares", stats.overlapped_prepares);
    obj.pushKV("prefetched_batches", stats.prefetched_batches);
    obj.pushKV("prefetched_inputs", stats.prefetched_inputs);
    obj.pushKV("prefetch_depth", stats.prefetch_depth);
    obj.pushKV("async_prepare_submissions", stats.async_prepare_submissions);
    obj.pushKV("async_prepare_completions", stats.async_prepare_completions);
    obj.pushKV("async_prepare_worker_threads", stats.async_prepare_worker_threads);
    obj.pushKV("batch_size", stats.batch_size);
    obj.pushKV("batched_digest_requests", stats.batched_digest_requests);
    obj.pushKV("batched_nonce_attempts", stats.batched_nonce_attempts);
    return obj;
}

UniValue BuildRuntimeStatsObject(const MatMulSolveRuntimeStats& stats)
{
    UniValue obj(UniValue::VOBJ);
    obj.pushKV("attempts", stats.attempts);
    obj.pushKV("solved_attempts", stats.solved_attempts);
    obj.pushKV("failed_attempts", stats.failed_attempts);
    obj.pushKV("total_elapsed_us", stats.total_elapsed_us);
    obj.pushKV("last_elapsed_us", stats.last_elapsed_us);
    obj.pushKV("max_elapsed_us", stats.max_elapsed_us);
    return obj;
}

UniValue BuildDigestCompareStatsObject(const MatMulDigestCompareStats& stats)
{
    UniValue obj(UniValue::VOBJ);
    obj.pushKV("enabled", stats.enabled);
    obj.pushKV("compared_attempts", stats.compared_attempts);
    obj.pushKV("first_divergence_captured", stats.first_divergence_captured);
    obj.pushKV("first_divergence_nonce64", stats.first_divergence_nonce64);
    obj.pushKV("first_divergence_nonce32", stats.first_divergence_nonce32);
    obj.pushKV("first_divergence_header_hash", stats.first_divergence_header_hash);
    obj.pushKV("first_divergence_backend_digest", stats.first_divergence_backend_digest);
    obj.pushKV("first_divergence_cpu_digest", stats.first_divergence_cpu_digest);
    return obj;
}

UniValue BuildGpuPreHashScanStatsObject(const MatMulGpuPreHashScanStats& stats)
{
    UniValue obj(UniValue::VOBJ);
    obj.pushKV("attempts", stats.attempts);
    obj.pushKV("successes", stats.successes);
    obj.pushKV("failures", stats.failures);
    obj.pushKV("metal_fallbacks_to_cpu", stats.metal_fallbacks_to_cpu);
    obj.pushKV("cuda_fallbacks_to_cpu", stats.cuda_fallbacks_to_cpu);
    obj.pushKV("total_elapsed_us", stats.total_elapsed_us);
    obj.pushKV("last_elapsed_us", stats.last_elapsed_us);
    obj.pushKV("max_elapsed_us", stats.max_elapsed_us);
    obj.pushKV("total_scanned_count", stats.total_scanned_count);
    obj.pushKV("last_scanned_count", stats.last_scanned_count);
    obj.pushKV("total_pass_flags", stats.total_pass_flags);
    obj.pushKV("last_pass_flags", stats.last_pass_flags);
    obj.pushKV("total_selected_headers", stats.total_selected_headers);
    obj.pushKV("last_selected_headers", stats.last_selected_headers);
    obj.pushKV("last_scan_limit", stats.last_scan_limit);
    obj.pushKV("last_backend", stats.last_backend);
    obj.pushKV("last_error", stats.last_error);
    return obj;
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

UniValue BuildCudaOracleProfileObject(const btx::cuda::MatMulInputGenerationProfile& stats)
{
    UniValue obj(UniValue::VOBJ);
    obj.pushKV("available", stats.available);
    obj.pushKV("pool_initialized", stats.pool_initialized);
    obj.pushKV("samples", stats.samples);
    obj.pushKV("allocation_events", stats.allocation_events);
    obj.pushKV("reuse_events", stats.reuse_events);
    obj.pushKV("last_encode_noise_us", stats.last_encode_noise_us);
    obj.pushKV("last_encode_compress_us", stats.last_encode_compress_us);
    obj.pushKV("last_submit_wait_us", stats.last_submit_wait_us);
    obj.pushKV("last_gpu_generation_ms", stats.last_gpu_generation_ms);
    obj.pushKV("library_source", stats.library_source);
    obj.pushKV("reason", stats.reason);
    return obj;
}

void AddPipelineStats(MatMulSolvePipelineStats& total, const MatMulSolvePipelineStats& stats)
{
    total.parallel_solver_enabled = total.parallel_solver_enabled || stats.parallel_solver_enabled;
    total.parallel_solver_threads = stats.parallel_solver_threads;
    total.async_prepare_enabled = total.async_prepare_enabled || stats.async_prepare_enabled;
    total.cpu_confirm_candidates = total.cpu_confirm_candidates || stats.cpu_confirm_candidates;
    total.prepared_inputs += stats.prepared_inputs;
    total.overlapped_prepares += stats.overlapped_prepares;
    total.prefetched_batches += stats.prefetched_batches;
    total.prefetched_inputs += stats.prefetched_inputs;
    total.async_prepare_submissions += stats.async_prepare_submissions;
    total.async_prepare_completions += stats.async_prepare_completions;
    total.async_prepare_worker_threads = stats.async_prepare_worker_threads;
    total.prefetch_depth = stats.prefetch_depth;
    total.batch_size = stats.batch_size;
    total.batched_digest_requests += stats.batched_digest_requests;
    total.batched_nonce_attempts += stats.batched_nonce_attempts;
}

void AddRuntimeStats(MatMulSolveRuntimeStats& total, const MatMulSolveRuntimeStats& stats)
{
    total.attempts += stats.attempts;
    total.solved_attempts += stats.solved_attempts;
    total.failed_attempts += stats.failed_attempts;
    total.total_elapsed_us += stats.total_elapsed_us;
    total.last_elapsed_us = stats.last_elapsed_us;
    total.max_elapsed_us = std::max(total.max_elapsed_us, stats.max_elapsed_us);
}

void AddGpuPreHashScanStats(MatMulGpuPreHashScanStats& total, const MatMulGpuPreHashScanStats& stats)
{
    total.attempts += stats.attempts;
    total.successes += stats.successes;
    total.failures += stats.failures;
    total.metal_fallbacks_to_cpu += stats.metal_fallbacks_to_cpu;
    total.cuda_fallbacks_to_cpu += stats.cuda_fallbacks_to_cpu;
    total.total_elapsed_us += stats.total_elapsed_us;
    total.last_elapsed_us = stats.last_elapsed_us;
    total.max_elapsed_us = std::max(total.max_elapsed_us, stats.max_elapsed_us);
    total.total_scanned_count += stats.total_scanned_count;
    total.last_scanned_count = stats.last_scanned_count;
    total.total_pass_flags += stats.total_pass_flags;
    total.last_pass_flags = stats.last_pass_flags;
    total.total_selected_headers += stats.total_selected_headers;
    total.last_selected_headers = stats.last_selected_headers;
    total.last_scan_limit = stats.last_scan_limit;
    total.last_backend = stats.last_backend;
    total.last_error = stats.last_error;
}

void AddDigestCompareStats(MatMulDigestCompareStats& total, const MatMulDigestCompareStats& stats)
{
    total.enabled = total.enabled || stats.enabled;
    total.compared_attempts += stats.compared_attempts;
    if (!total.first_divergence_captured && stats.first_divergence_captured) {
        total.first_divergence_captured = true;
        total.first_divergence_nonce64 = stats.first_divergence_nonce64;
        total.first_divergence_nonce32 = stats.first_divergence_nonce32;
        total.first_divergence_header_hash = stats.first_divergence_header_hash;
        total.first_divergence_backend_digest = stats.first_divergence_backend_digest;
        total.first_divergence_cpu_digest = stats.first_divergence_cpu_digest;
    }
}

void AddBackendRuntimeStats(matmul::accelerated::BackendRuntimeStats& total,
                            const matmul::accelerated::BackendRuntimeStats& stats)
{
    total.digest_requests += stats.digest_requests;
    total.requested_cpu += stats.requested_cpu;
    total.requested_metal += stats.requested_metal;
    total.requested_cuda += stats.requested_cuda;
    total.requested_unknown += stats.requested_unknown;
    total.metal_successes += stats.metal_successes;
    total.metal_fallbacks_to_cpu += stats.metal_fallbacks_to_cpu;
    total.metal_digest_mismatches += stats.metal_digest_mismatches;
    total.metal_retry_without_uploaded_base_attempts += stats.metal_retry_without_uploaded_base_attempts;
    total.metal_retry_without_uploaded_base_successes += stats.metal_retry_without_uploaded_base_successes;
    total.cuda_successes += stats.cuda_successes;
    total.cuda_fallbacks_to_cpu += stats.cuda_fallbacks_to_cpu;
    total.gpu_input_generation_attempts += stats.gpu_input_generation_attempts;
    total.gpu_input_generation_successes += stats.gpu_input_generation_successes;
    total.gpu_input_generation_failures += stats.gpu_input_generation_failures;
    total.gpu_input_auto_disabled_skips += stats.gpu_input_auto_disabled_skips;
    total.cuda_variable_base_batches += stats.cuda_variable_base_batches;
    total.cuda_variable_base_last_device_us = stats.cuda_variable_base_last_device_us;
    total.cuda_variable_base_last_host_digest_us = stats.cuda_variable_base_last_host_digest_us;
    total.cuda_variable_base_total_host_digest_us += stats.cuda_variable_base_total_host_digest_us;
    total.gpu_input_auto_disabled = total.gpu_input_auto_disabled || stats.gpu_input_auto_disabled;
    total.last_metal_fallback_error = stats.last_metal_fallback_error;
    total.last_cuda_fallback_error = stats.last_cuda_fallback_error;
    total.last_gpu_input_error = stats.last_gpu_input_error;
}

CBlockHeader BuildCandidateHeader(uint32_t n, uint32_t nbits, uint64_t nonce64)
{
    CBlockHeader candidate{};
    candidate.nVersion = 1;
    candidate.hashPrevBlock = ParseUint256("0000000000000000000000000000000000000000000000000000000000000011");
    candidate.hashMerkleRoot = ParseUint256("0000000000000000000000000000000000000000000000000000000000000022");
    candidate.nTime = 1'773'277'390U;
    candidate.nBits = nbits;
    candidate.nNonce64 = nonce64;
    candidate.nNonce = static_cast<uint32_t>(nonce64);
    candidate.matmul_dim = static_cast<uint16_t>(n);
    candidate.seed_a = ParseUint256("6410ee507c58dca3d22f950385d38fdd5fba9dd2e424b2657a2410e92d23dc63");
    candidate.seed_b = ParseUint256("7f165f0361461f69e2442a31fec8c26d2d95928cae37cb1673cd14fbba25f03c");
    candidate.matmul_digest.SetNull();
    return candidate;
}

struct IterationResult {
    double elapsed_s{0.0};
    double nonces_per_sec{0.0};
    uint64_t attempts{0};
    uint64_t solved_count{0};
    MatMulSolvePipelineStats pipeline{};
    MatMulSolveRuntimeStats runtime{};
    MatMulGpuPreHashScanStats gpu_prehash_scan{};
    MatMulDigestCompareStats digest_compare{};
    matmul::accelerated::BackendRuntimeStats backend_runtime{};
    btx::cuda::MatMulProfilingStats cuda_profiling{};
    uint64_t cuda_profiling_sample_delta{0};
    btx::cuda::MatMulInputGenerationProfile cuda_oracle_profile{};
    uint64_t cuda_oracle_sample_delta{0};
    uint64_t cuda_oracle_allocation_delta{0};
    uint64_t cuda_oracle_reuse_delta{0};
};

IterationResult RunSolveIteration(const Options& options, const Consensus::Params& consensus, uint32_t iteration_index)
{
    const auto prehash_before = ProbeMatMulGpuPreHashScanStats();
    const auto digest_compare_before = ProbeMatMulDigestCompareStats();
    const auto backend_runtime_before = matmul::accelerated::ProbeMatMulBackendRuntimeStats();
    const auto cuda_profiling_before = btx::cuda::ProbeMatMulProfilingStats();
    const auto cuda_oracle_before = btx::cuda::ProbeMatMulInputGenerationProfile();

    ResetMatMulSolvePipelineStats();
    ResetMatMulSolveRuntimeStats();

    IterationResult result;
    const auto start = std::chrono::steady_clock::now();
    if (options.parallel == 1) {
        CBlockHeader candidate = BuildCandidateHeader(
            options.n,
            options.nbits,
            static_cast<uint64_t>(iteration_index) * options.max_tries + 1U);
        uint64_t tries = options.max_tries;
        if (SolveMatMul(
                candidate,
                consensus,
                tries,
                options.block_height,
                nullptr,
                nullptr,
                nullptr,
                options.parent_mtp_override)) {
            ++result.solved_count;
        }
        result.attempts = options.max_tries - tries;
        result.nonces_per_sec = static_cast<double>(result.attempts);
    } else {
        std::vector<uint64_t> attempts_used(options.parallel, 0);
        std::vector<uint64_t> solved_counts(options.parallel, 0);
        std::vector<std::thread> workers;
        workers.reserve(options.parallel);

        for (uint32_t worker_index = 0; worker_index < options.parallel; ++worker_index) {
            workers.emplace_back([&, worker_index] {
                const uint64_t nonce64 =
                    ((static_cast<uint64_t>(iteration_index) * options.parallel + worker_index) * options.max_tries) + 1U;
                CBlockHeader candidate = BuildCandidateHeader(options.n, options.nbits, nonce64);
                uint64_t tries = options.max_tries;
                if (SolveMatMul(
                        candidate,
                        consensus,
                        tries,
                        options.block_height,
                        nullptr,
                        nullptr,
                        nullptr,
                        options.parent_mtp_override)) {
                    solved_counts[worker_index] = 1;
                }
                attempts_used[worker_index] = options.max_tries - tries;
            });
        }

        for (auto& worker : workers) {
            worker.join();
        }

        const uint64_t total_attempts = std::accumulate(attempts_used.begin(), attempts_used.end(), uint64_t{0});
        result.solved_count = std::accumulate(solved_counts.begin(), solved_counts.end(), uint64_t{0});
        result.attempts = total_attempts;
        result.nonces_per_sec = static_cast<double>(total_attempts);
    }
    const auto stop = std::chrono::steady_clock::now();
    result.elapsed_s = std::chrono::duration<double>(stop - start).count();
    if (result.elapsed_s > 0.0) {
        result.nonces_per_sec /= result.elapsed_s;
    } else {
        result.nonces_per_sec = 0.0;
    }
    result.pipeline = ProbeMatMulSolvePipelineStats();
    result.runtime = ProbeMatMulSolveRuntimeStats();
    result.gpu_prehash_scan = DeltaGpuPreHashScanStats(prehash_before, ProbeMatMulGpuPreHashScanStats());
    result.digest_compare = DeltaDigestCompareStats(digest_compare_before, ProbeMatMulDigestCompareStats());
    result.backend_runtime = DeltaBackendRuntimeStats(
        backend_runtime_before,
        matmul::accelerated::ProbeMatMulBackendRuntimeStats());
    result.cuda_profiling = btx::cuda::ProbeMatMulProfilingStats();
    result.cuda_profiling_sample_delta = CounterDelta(cuda_profiling_before.samples, result.cuda_profiling.samples);
    result.cuda_oracle_profile = btx::cuda::ProbeMatMulInputGenerationProfile();
    result.cuda_oracle_sample_delta = CounterDelta(cuda_oracle_before.samples, result.cuda_oracle_profile.samples);
    result.cuda_oracle_allocation_delta = CounterDelta(cuda_oracle_before.allocation_events, result.cuda_oracle_profile.allocation_events);
    result.cuda_oracle_reuse_delta = CounterDelta(cuda_oracle_before.reuse_events, result.cuda_oracle_profile.reuse_events);
    return result;
}

} // namespace

int main(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg{argv[i]};
        if (arg == "--help" || arg == "-h") {
            PrintUsage(std::cout);
            return 0;
        }
    }

    Options options;
    if (!ParseArgs(argc, argv, options)) {
        return argc > 1 ? 1 : 0;
    }

    ScopedEnvOverride backend_env("BTX_MATMUL_BACKEND", options.backend_override);
    ScopedEnvOverride async_env("BTX_MATMUL_PIPELINE_ASYNC", options.async_override);
    ScopedEnvOverride gpu_inputs_env("BTX_MATMUL_GPU_INPUTS", options.gpu_inputs_override);
    ScopedEnvOverride batch_size_env("BTX_MATMUL_SOLVE_BATCH_SIZE", options.batch_size_override);
    ScopedEnvOverride digest_slice_size_env("BTX_MATMUL_DIGEST_SLICE_SIZE", options.digest_slice_size_override);
    ScopedEnvOverride prefetch_depth_env("BTX_MATMUL_PREPARE_PREFETCH_DEPTH", options.prefetch_depth_override);
    ScopedEnvOverride prepare_workers_env("BTX_MATMUL_PREPARE_WORKERS", options.prepare_workers_override);
    ScopedEnvOverride pool_slots_env("BTX_MATMUL_METAL_POOL_SLOTS", options.pool_slots_override);
    ScopedEnvOverride cuda_pool_slots_env("BTX_MATMUL_CUDA_POOL_SLOTS", options.pool_slots_override);
    ScopedEnvOverride solver_threads_env("BTX_MATMUL_SOLVER_THREADS", options.solver_threads_override);

    ArgsManager args;
    auto consensus = CreateChainParams(args, ChainType::REGTEST)->GetConsensus();
    consensus.fMatMulPOW = true;
    consensus.fSkipMatMulValidation = false;
    consensus.nMatMulDimension = options.n;
    consensus.nMatMulTranscriptBlockSize = options.b;
    consensus.nMatMulNoiseRank = options.r;
    consensus.nMatMulPreHashEpsilonBits = options.epsilon_bits;
    if (options.nonce_seed_height_override.has_value()) {
        consensus.nMatMulNonceSeedHeight = *options.nonce_seed_height_override;
    }
    if (options.parent_mtp_seed_height_override.has_value()) {
        consensus.nMatMulParentMtpSeedHeight = *options.parent_mtp_seed_height_override;
    }
    if (options.product_digest_height_override.has_value()) {
        consensus.nMatMulProductDigestHeight = *options.product_digest_height_override;
    }
    consensus.powLimit = uint256{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"};

    std::vector<double> elapsed_s_values;
    std::vector<double> nonces_per_sec_values;
    std::vector<double> attempts_values;
    std::vector<IterationResult> iteration_results;
    elapsed_s_values.reserve(options.iterations);
    nonces_per_sec_values.reserve(options.iterations);
    attempts_values.reserve(options.iterations);
    iteration_results.reserve(options.iterations);

    uint64_t warmup_attempts{0};
    double warmup_elapsed_s{0.0};
    for (uint32_t i = 0; i < options.warmup_iterations; ++i) {
        const IterationResult warmup = RunSolveIteration(options, consensus, i);
        warmup_attempts += warmup.attempts;
        warmup_elapsed_s += warmup.elapsed_s;
    }

    uint64_t solved_count{0};
    uint64_t total_attempts{0};
    MatMulSolvePipelineStats last_pipeline{};
    MatMulSolveRuntimeStats last_runtime{};
    MatMulGpuPreHashScanStats last_gpu_prehash_scan{};
    MatMulDigestCompareStats last_digest_compare{};
    matmul::accelerated::BackendRuntimeStats last_backend_runtime{};
    btx::cuda::MatMulProfilingStats last_cuda_profiling{};
    btx::cuda::MatMulInputGenerationProfile last_cuda_oracle_profile{};
    uint64_t total_cuda_profiling_sample_delta{0};
    uint64_t total_cuda_oracle_sample_delta{0};
    uint64_t total_cuda_oracle_allocation_delta{0};
    uint64_t total_cuda_oracle_reuse_delta{0};

    MatMulSolvePipelineStats total_pipeline{};
    MatMulSolveRuntimeStats total_runtime{};
    MatMulGpuPreHashScanStats total_gpu_prehash_scan{};
    MatMulDigestCompareStats total_digest_compare{};
    matmul::accelerated::BackendRuntimeStats total_backend_runtime{};

    const auto measured_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < options.iterations; ++i) {
        const IterationResult iteration = RunSolveIteration(options, consensus, options.warmup_iterations + i);
        iteration_results.push_back(iteration);
        elapsed_s_values.push_back(iteration.elapsed_s);
        nonces_per_sec_values.push_back(iteration.nonces_per_sec);
        attempts_values.push_back(static_cast<double>(iteration.attempts));
        total_attempts += iteration.attempts;
        solved_count += iteration.solved_count;
        last_pipeline = iteration.pipeline;
        last_runtime = iteration.runtime;
        last_gpu_prehash_scan = iteration.gpu_prehash_scan;
        last_digest_compare = iteration.digest_compare;
        last_backend_runtime = iteration.backend_runtime;
        last_cuda_profiling = iteration.cuda_profiling;
        last_cuda_oracle_profile = iteration.cuda_oracle_profile;
        total_cuda_profiling_sample_delta += iteration.cuda_profiling_sample_delta;
        total_cuda_oracle_sample_delta += iteration.cuda_oracle_sample_delta;
        total_cuda_oracle_allocation_delta += iteration.cuda_oracle_allocation_delta;
        total_cuda_oracle_reuse_delta += iteration.cuda_oracle_reuse_delta;
        AddPipelineStats(total_pipeline, iteration.pipeline);
        AddRuntimeStats(total_runtime, iteration.runtime);
        AddGpuPreHashScanStats(total_gpu_prehash_scan, iteration.gpu_prehash_scan);
        AddDigestCompareStats(total_digest_compare, iteration.digest_compare);
        AddBackendRuntimeStats(total_backend_runtime, iteration.backend_runtime);
    }
    const auto measured_stop = std::chrono::steady_clock::now();
    const double measured_elapsed_s = std::chrono::duration<double>(measured_stop - measured_start).count();

    UniValue output(UniValue::VOBJ);
    UniValue options_obj(UniValue::VOBJ);
    options_obj.pushKV("iterations", options.iterations);
    options_obj.pushKV("warmup_iterations", options.warmup_iterations);
    options_obj.pushKV("max_tries", options.max_tries);
    options_obj.pushKV("n", options.n);
    options_obj.pushKV("b", options.b);
    options_obj.pushKV("r", options.r);
    options_obj.pushKV("nbits", options.nbits);
    options_obj.pushKV("epsilon_bits", options.epsilon_bits);
    options_obj.pushKV("block_height", options.block_height);
    options_obj.pushKV("nonce_seed_height", options.nonce_seed_height_override.has_value() ? UniValue(*options.nonce_seed_height_override) : UniValue());
    options_obj.pushKV("nonce_seed_active", consensus.IsMatMulNonceSeedActive(options.block_height));
    options_obj.pushKV("parent_mtp_seed_height", options.parent_mtp_seed_height_override.has_value() ? UniValue(*options.parent_mtp_seed_height_override) : UniValue());
    options_obj.pushKV("parent_mtp_seed_active", consensus.IsMatMulParentMtpSeedActive(options.block_height));
    options_obj.pushKV("parent_mtp", options.parent_mtp_override.has_value() ? UniValue(*options.parent_mtp_override) : UniValue());
    options_obj.pushKV("product_digest_height", options.product_digest_height_override.has_value() ? UniValue(*options.product_digest_height_override) : UniValue());
    options_obj.pushKV("product_digest_active", consensus.IsMatMulProductDigestActive(options.block_height));
    options_obj.pushKV("parallel", options.parallel);
    options_obj.pushKV("backend_override", options.backend_override.has_value() ? UniValue(*options.backend_override) : UniValue());
    options_obj.pushKV("async_override", options.async_override.has_value() ? UniValue(*options.async_override) : UniValue());
    options_obj.pushKV("gpu_inputs_override", options.gpu_inputs_override.has_value() ? UniValue(*options.gpu_inputs_override) : UniValue());
    options_obj.pushKV("batch_size_override", options.batch_size_override.has_value() ? UniValue(*options.batch_size_override) : UniValue());
    options_obj.pushKV("digest_slice_size_override", options.digest_slice_size_override.has_value() ? UniValue(*options.digest_slice_size_override) : UniValue());
    options_obj.pushKV("prefetch_depth_override", options.prefetch_depth_override.has_value() ? UniValue(*options.prefetch_depth_override) : UniValue());
    options_obj.pushKV("prepare_workers_override", options.prepare_workers_override.has_value() ? UniValue(*options.prepare_workers_override) : UniValue());
    options_obj.pushKV("pool_slots_override", options.pool_slots_override.has_value() ? UniValue(*options.pool_slots_override) : UniValue());
    options_obj.pushKV("solver_threads_override", options.solver_threads_override.has_value() ? UniValue(*options.solver_threads_override) : UniValue());
    options_obj.pushKV("per_iteration", options.per_iteration);
    output.pushKV("options", std::move(options_obj));

    output.pushKV("solved_count", solved_count);
    output.pushKV("total_attempts", total_attempts);
    output.pushKV("elapsed_s", SummarizeSeries(elapsed_s_values));
    output.pushKV("nonces_per_sec", SummarizeSeries(nonces_per_sec_values));
    output.pushKV("attempts", SummarizeSeries(attempts_values));

    UniValue measured_obj(UniValue::VOBJ);
    measured_obj.pushKV("iterations", options.iterations);
    measured_obj.pushKV("total_attempts", total_attempts);
    measured_obj.pushKV("total_elapsed_s", measured_elapsed_s);
    measured_obj.pushKV("aggregate_nonces_per_sec", PerSecond(total_attempts, measured_elapsed_s));
    measured_obj.pushKV("solved_count", solved_count);
    measured_obj.pushKV("warmup_iterations", options.warmup_iterations);
    measured_obj.pushKV("warmup_attempts", warmup_attempts);
    measured_obj.pushKV("warmup_elapsed_s", warmup_elapsed_s);
    measured_obj.pushKV("pipeline", BuildPipelineStatsObject(total_pipeline));
    measured_obj.pushKV("runtime", BuildRuntimeStatsObject(total_runtime));
    measured_obj.pushKV("gpu_prehash_scan", BuildGpuPreHashScanStatsObject(total_gpu_prehash_scan));
    measured_obj.pushKV("digest_compare", BuildDigestCompareStatsObject(total_digest_compare));
    measured_obj.pushKV("backend_runtime", BuildBackendRuntimeStatsObject(total_backend_runtime));
    measured_obj.pushKV("cuda_profiling_sample_delta", total_cuda_profiling_sample_delta);
    measured_obj.pushKV("cuda_oracle_sample_delta", total_cuda_oracle_sample_delta);
    measured_obj.pushKV("cuda_oracle_allocation_delta", total_cuda_oracle_allocation_delta);
    measured_obj.pushKV("cuda_oracle_reuse_delta", total_cuda_oracle_reuse_delta);
    output.pushKV("measured_totals", std::move(measured_obj));

    if (options.per_iteration) {
        UniValue iterations_arr(UniValue::VARR);
        for (uint32_t i = 0; i < iteration_results.size(); ++i) {
            const auto& iteration = iteration_results[i];
            UniValue iteration_obj(UniValue::VOBJ);
            iteration_obj.pushKV("index", i);
            iteration_obj.pushKV("nonce_range_index", options.warmup_iterations + i);
            iteration_obj.pushKV("elapsed_s", iteration.elapsed_s);
            iteration_obj.pushKV("attempts", iteration.attempts);
            iteration_obj.pushKV("nonces_per_sec", iteration.nonces_per_sec);
            iteration_obj.pushKV("solved_count", iteration.solved_count);
            iteration_obj.pushKV("pipeline", BuildPipelineStatsObject(iteration.pipeline));
            iteration_obj.pushKV("runtime", BuildRuntimeStatsObject(iteration.runtime));
            iteration_obj.pushKV("gpu_prehash_scan", BuildGpuPreHashScanStatsObject(iteration.gpu_prehash_scan));
            iteration_obj.pushKV("digest_compare", BuildDigestCompareStatsObject(iteration.digest_compare));
            iteration_obj.pushKV("backend_runtime", BuildBackendRuntimeStatsObject(iteration.backend_runtime));
            iteration_obj.pushKV("cuda_profiling_sample_delta", iteration.cuda_profiling_sample_delta);
            iteration_obj.pushKV("cuda_profiling", BuildCudaProfilingStatsObject(iteration.cuda_profiling));
            iteration_obj.pushKV("cuda_oracle_sample_delta", iteration.cuda_oracle_sample_delta);
            iteration_obj.pushKV("cuda_oracle_allocation_delta", iteration.cuda_oracle_allocation_delta);
            iteration_obj.pushKV("cuda_oracle_reuse_delta", iteration.cuda_oracle_reuse_delta);
            iteration_obj.pushKV("cuda_oracle_profile", BuildCudaOracleProfileObject(iteration.cuda_oracle_profile));
            iterations_arr.push_back(std::move(iteration_obj));
        }
        output.pushKV("iterations", std::move(iterations_arr));
    }

    const char* backend_env_value = std::getenv("BTX_MATMUL_BACKEND");
    const std::string requested_backend = backend_env_value != nullptr ? backend_env_value :
#if defined(__APPLE__)
        "metal";
#else
        "cpu";
#endif
    const auto backend_selection = matmul::backend::ResolveRequestedBackend(requested_backend);
    output.pushKV("requested_backend", matmul::backend::ToString(backend_selection.requested));
    output.pushKV("active_backend", matmul::backend::ToString(backend_selection.active));
    output.pushKV("backend_selection_reason", backend_selection.reason);

    if (backend_selection.requested == matmul::backend::Kind::METAL ||
        backend_selection.active == matmul::backend::Kind::METAL) {
        const auto device_info = btx::metal::ProbeMatMulDeviceInfo();
        UniValue metal_device_obj(UniValue::VOBJ);
        metal_device_obj.pushKV("available", device_info.available);
        metal_device_obj.pushKV("device_name", device_info.device_name);
        metal_device_obj.pushKV("gpu_core_count", device_info.gpu_core_count);
        metal_device_obj.pushKV("gpu_core_count_source", device_info.gpu_core_count_source);
        metal_device_obj.pushKV("reason", device_info.reason);
        output.pushKV("metal_device", std::move(metal_device_obj));
    }

    output.pushKV("last_pipeline_stats", BuildPipelineStatsObject(last_pipeline));
    output.pushKV("last_runtime_stats", BuildRuntimeStatsObject(last_runtime));
    output.pushKV("last_digest_compare_stats", BuildDigestCompareStatsObject(last_digest_compare));
    output.pushKV("last_gpu_prehash_scan_stats", BuildGpuPreHashScanStatsObject(last_gpu_prehash_scan));
    output.pushKV("last_backend_runtime_stats", BuildBackendRuntimeStatsObject(last_backend_runtime));

    if (backend_selection.requested == matmul::backend::Kind::CUDA ||
        backend_selection.active == matmul::backend::Kind::CUDA) {
        output.pushKV("cuda_profiling_stats", BuildCudaProfilingStatsObject(last_cuda_profiling));
        output.pushKV("cuda_oracle_profile", BuildCudaOracleProfileObject(last_cuda_oracle_profile));
    }

    UniValue pool_obj(UniValue::VOBJ);

    auto push_pool_stats = [&](const auto& pool_stats, const char* backend_name) {
        output.pushKV("buffer_pool_backend", backend_name);
        pool_obj.pushKV("available", pool_stats.available);
        pool_obj.pushKV("initialized", pool_stats.initialized);
        pool_obj.pushKV("allocation_events", pool_stats.allocation_events);
        pool_obj.pushKV("reuse_events", pool_stats.reuse_events);
        pool_obj.pushKV("wait_events", pool_stats.wait_events);
        pool_obj.pushKV("slot_count", pool_stats.slot_count);
        pool_obj.pushKV("active_slots", pool_stats.active_slots);
        pool_obj.pushKV("high_water_slots", pool_stats.high_water_slots);
        pool_obj.pushKV("inflight_submissions", pool_stats.inflight_submissions);
        pool_obj.pushKV("peak_inflight_submissions", pool_stats.peak_inflight_submissions);
        pool_obj.pushKV("completed_submissions", pool_stats.completed_submissions);
        pool_obj.pushKV("n", pool_stats.n);
        pool_obj.pushKV("b", pool_stats.b);
        pool_obj.pushKV("r", pool_stats.r);
        pool_obj.pushKV("reason", pool_stats.reason);
    };

    const auto buffer_pool_backend =
        backend_selection.requested == matmul::backend::Kind::METAL || backend_selection.requested == matmul::backend::Kind::CUDA
        ? backend_selection.requested
        : backend_selection.active;
    switch (buffer_pool_backend) {
    case matmul::backend::Kind::CUDA:
        push_pool_stats(btx::cuda::ProbeMatMulBufferPool(), "cuda");
        break;
    case matmul::backend::Kind::METAL:
        push_pool_stats(btx::metal::ProbeMatMulBufferPool(), "metal");
        break;
    case matmul::backend::Kind::CPU:
        pool_obj.pushKV("available", false);
        pool_obj.pushKV("initialized", false);
        pool_obj.pushKV("allocation_events", 0);
        pool_obj.pushKV("reuse_events", 0);
        pool_obj.pushKV("wait_events", 0);
        pool_obj.pushKV("slot_count", 0);
        pool_obj.pushKV("active_slots", 0);
        pool_obj.pushKV("high_water_slots", 0);
        pool_obj.pushKV("inflight_submissions", 0);
        pool_obj.pushKV("peak_inflight_submissions", 0);
        pool_obj.pushKV("completed_submissions", 0);
        pool_obj.pushKV("n", 0);
        pool_obj.pushKV("b", 0);
        pool_obj.pushKV("r", 0);
        pool_obj.pushKV("reason", "no_gpu_buffer_pool_for_backend");
        output.pushKV("buffer_pool_backend", "cpu");
        break;
    }
    output.pushKV("buffer_pool_stats", std::move(pool_obj));

    std::cout << output.write(2) << std::endl;
    return 0;
}

// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/license/mit/.

#include <cuda/oracle_accel.h>

#include <crypto/sha256.h>
#include <cuda/cuda_context.h>
#include <cuda/cuda_scheduler.h>
#include <cuda_runtime.h>
#include <matmul/noise.h>
#include <matmul/transcript.h>
#include <span.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <new>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace btx::cuda {
namespace {

using Element = matmul::field::Element;
constexpr uint32_t MODULUS = matmul::field::MODULUS;
constexpr uint32_t ORACLE_THREADS = 256;
constexpr uint32_t ORACLE_INPUT_SEED_KINDS = 5;
constexpr uint32_t ORACLE_SEED_MIDSTATE_WORDS = 23;

struct OracleSeedBytes {
    uint8_t data[32];
};

struct OracleProfileState {
    std::atomic<bool> pool_initialized{false};
    std::atomic<uint64_t> samples{0};
    std::atomic<uint64_t> allocation_events{0};
    std::atomic<uint64_t> reuse_events{0};
    std::mutex mutex;
    double last_encode_noise_us{0.0};
    double last_encode_compress_us{0.0};
    double last_submit_wait_us{0.0};
    double last_gpu_generation_ms{0.0};
    std::string reason{"cuda_oracle_ready"};
};

struct OracleWorkspace {
    struct HostStageBuffer {
        Element* pinned{nullptr};
        size_t capacity{0};
        bool pinned_disabled{false};
        std::vector<Element> fallback;

        ~HostStageBuffer() { cudaFreeHost(pinned); }

        bool Ensure(size_t required, std::string& error)
        {
            if (required == 0) {
                fallback.clear();
                return true;
            }

            if (!pinned_disabled && pinned != nullptr && capacity >= required) {
                return true;
            }

            if (!pinned_disabled) {
                cudaFreeHost(pinned);
                pinned = nullptr;
                capacity = 0;

                Element* candidate{nullptr};
                const cudaError_t alloc_error = cudaMallocHost(&candidate, required * sizeof(Element));
                if (alloc_error == cudaSuccess) {
                    pinned = candidate;
                    capacity = required;
                    fallback.clear();
                    return true;
                }

                pinned_disabled = true;
                error = "cudaMallocHost failed:" + std::string(cudaGetErrorString(alloc_error)) +
                    "; falling back to pageable host memory";
            }

            try {
                fallback.resize(required);
            } catch (const std::bad_alloc&) {
                error = "host staging allocation failed";
                return false;
            }
            return true;
        }

        Element* data()
        {
            return pinned != nullptr ? pinned : fallback.data();
        }
    };

    int device_index{-1};
    cudaStream_t stream{nullptr};
    Element* out_e_l{nullptr};
    Element* out_e_r{nullptr};
    Element* out_f_l{nullptr};
    Element* out_f_r{nullptr};
    Element* out_cv{nullptr};
    Element* out_scan_flags{nullptr};
    Element* out_scan_selected_offsets{nullptr};
    Element* out_scan_selected_count{nullptr};
    OracleSeedBytes* batch_seed_el{nullptr};
    OracleSeedBytes* batch_seed_er{nullptr};
    OracleSeedBytes* batch_seed_fl{nullptr};
    OracleSeedBytes* batch_seed_fr{nullptr};
    OracleSeedBytes* batch_seed_cv{nullptr};
    Element* batch_seed_midstates{nullptr};
    size_t noise_capacity{0};
    size_t compress_capacity{0};
    size_t scan_flags_capacity{0};
    size_t scan_selected_offsets_capacity{0};
    size_t scan_selected_count_capacity{0};
    size_t batch_seed_capacity{0};
    size_t batch_seed_midstates_capacity{0};
    HostStageBuffer host_e_l;
    HostStageBuffer host_e_r;
    HostStageBuffer host_f_l;
    HostStageBuffer host_f_r;
    HostStageBuffer host_cv;
    HostStageBuffer host_scan_flags;
    HostStageBuffer host_scan_selected_offsets;
    HostStageBuffer host_scan_selected_count;

    void ReleaseOutputs()
    {
        cudaFree(out_scan_selected_count);
        cudaFree(out_scan_selected_offsets);
        cudaFree(out_scan_flags);
        cudaFree(batch_seed_cv);
        cudaFree(batch_seed_fr);
        cudaFree(batch_seed_fl);
        cudaFree(batch_seed_er);
        cudaFree(batch_seed_el);
        cudaFree(batch_seed_midstates);
        cudaFree(out_cv);
        cudaFree(out_f_r);
        cudaFree(out_f_l);
        cudaFree(out_e_r);
        cudaFree(out_e_l);

        out_scan_flags = nullptr;
        out_scan_selected_offsets = nullptr;
        out_scan_selected_count = nullptr;
        batch_seed_el = nullptr;
        batch_seed_er = nullptr;
        batch_seed_fl = nullptr;
        batch_seed_fr = nullptr;
        batch_seed_cv = nullptr;
        batch_seed_midstates = nullptr;
        out_cv = nullptr;
        out_f_r = nullptr;
        out_f_l = nullptr;
        out_e_r = nullptr;
        out_e_l = nullptr;
        noise_capacity = 0;
        compress_capacity = 0;
        scan_flags_capacity = 0;
        scan_selected_offsets_capacity = 0;
        scan_selected_count_capacity = 0;
        batch_seed_capacity = 0;
        batch_seed_midstates_capacity = 0;
    }

    void ReleaseStream()
    {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }

    ~OracleWorkspace()
    {
        if (device_index >= 0) {
            cudaSetDevice(device_index);
        }
        ReleaseStream();
        ReleaseOutputs();
    }
};

thread_local OracleWorkspace g_workspace;
OracleProfileState g_profile;
std::atomic<uint64_t> g_next_device_generation{0};

struct DeviceInputPoolSlot {
    MatMulGeneratedInputsDevice inputs;
    size_t storage_capacity_words{0};
    bool in_use{false};
};

struct DeviceInputPoolContext {
    std::mutex mutex;
    std::vector<std::unique_ptr<DeviceInputPoolSlot>> slots;
    uint32_t next_slot{0};
};

struct DeviceInputBatchPoolSlot {
    int device_index{-1};
    Element* storage{nullptr};
    size_t storage_capacity_words{0};
    cudaEvent_t ready_event{nullptr};
    std::unique_ptr<MatMulGeneratedInputsDevice[]> views;
    uint32_t view_capacity{0};
    uint32_t count{0};
    bool in_use{false};

    ~DeviceInputBatchPoolSlot()
    {
        if (device_index >= 0) {
            cudaSetDevice(device_index);
        }
        if (ready_event != nullptr) {
            cudaEventDestroy(ready_event);
        }
        cudaFree(storage);
    }
};

struct DeviceInputBatchPoolContext {
    std::mutex mutex;
    std::vector<std::unique_ptr<DeviceInputBatchPoolSlot>> slots;
    uint32_t next_slot{0};
};

DeviceInputPoolContext& GetDeviceInputPoolContext()
{
    static DeviceInputPoolContext context;
    return context;
}

DeviceInputBatchPoolContext& GetDeviceInputBatchPoolContext()
{
    static DeviceInputBatchPoolContext context;
    return context;
}

std::array<uint8_t, 32> ToCanonicalBytes(const uint256& value)
{
    std::array<uint8_t, 32> out;
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = value.data()[out.size() - 1 - i];
    }
    return out;
}

uint256 CanonicalBytesToUint256(const uint8_t* bytes)
{
    std::array<unsigned char, 32> internal;
    for (size_t i = 0; i < internal.size(); ++i) {
        internal[i] = bytes[internal.size() - 1 - i];
    }
    return uint256{Span<const unsigned char>{internal.data(), internal.size()}};
}

uint256 DeriveCompressionSeed(const uint256& sigma)
{
    const auto sigma_bytes = ToCanonicalBytes(sigma);
    CSHA256 hasher;
    hasher.Write(reinterpret_cast<const uint8_t*>(matmul::transcript::COMPRESS_TAG.data()),
                 matmul::transcript::COMPRESS_TAG.size());
    hasher.Write(sigma_bytes.data(), sigma_bytes.size());

    uint8_t digest[CSHA256::OUTPUT_SIZE];
    hasher.Finalize(digest);
    return CanonicalBytesToUint256(digest);
}

OracleSeedBytes ToInternalSeedBytes(const uint256& seed)
{
    OracleSeedBytes out{};
    std::memcpy(out.data, seed.data(), sizeof(out.data));
    return out;
}

CudaRuntimeProbe RuntimeProbeFromDeviceInfo(const CudaTopologyProbe& topology, const CudaDeviceInfo& device)
{
    CudaRuntimeProbe probe;
    probe.compiled = topology.compiled;
    probe.available = topology.available && device.supported;
    probe.reason = probe.available ? "ready" : device.reason;
    probe.device_index = device.device_index;
    probe.device_name = device.device_name;
    probe.compute_capability_major = device.compute_capability_major;
    probe.compute_capability_minor = device.compute_capability_minor;
    probe.global_memory_bytes = device.global_memory_bytes;
    probe.multiprocessor_count = device.multiprocessor_count;
    probe.driver_api_version = topology.driver_api_version;
    probe.runtime_version = topology.runtime_version;
    return probe;
}

std::optional<CudaRuntimeProbe> ResolveCudaRuntimeForSelectedDevice(int device_index, std::string& error)
{
    const auto topology = ProbeCudaTopology();
    if (!topology.available) {
        error = topology.reason;
        return std::nullopt;
    }

    for (const auto& device : topology.selected_devices) {
        if (device.device_index == device_index) {
            return RuntimeProbeFromDeviceInfo(topology, device);
        }
    }

    error = "selected_cuda_device_not_enabled:" + std::to_string(device_index);
    return std::nullopt;
}

std::optional<CudaRuntimeProbe> ResolveCudaRuntimeForNextSelectedDevice(std::string& error)
{
    const auto topology = ProbeCudaTopology();
    if (!topology.available) {
        error = topology.reason;
        return std::nullopt;
    }

    std::vector<const CudaDeviceInfo*> supported_devices;
    supported_devices.reserve(topology.selected_devices.size());
    for (const auto& device : topology.selected_devices) {
        if (device.supported) {
            supported_devices.push_back(&device);
        }
    }
    if (supported_devices.empty()) {
        error = "no_supported_cuda_devices_selected";
        return std::nullopt;
    }

    const uint64_t ticket = g_next_device_generation.fetch_add(1, std::memory_order_relaxed);
    const auto* device = supported_devices[static_cast<size_t>(ticket % supported_devices.size())];
    return RuntimeProbeFromDeviceInfo(topology, *device);
}

void ResetWorkspaceForDevice(OracleWorkspace& workspace, int device_index)
{
    if (workspace.device_index == device_index) {
        return;
    }

    if (workspace.device_index >= 0) {
        cudaSetDevice(workspace.device_index);
    }
    workspace.ReleaseStream();
    workspace.ReleaseOutputs();
    workspace.device_index = device_index;
}

bool EnsureWorkspaceStream(OracleWorkspace& workspace, std::string& error)
{
    if (workspace.stream != nullptr) {
        return true;
    }

    const cudaError_t stream_error = cudaStreamCreateWithFlags(&workspace.stream, cudaStreamNonBlocking);
    if (stream_error != cudaSuccess) {
        error = "cudaStreamCreateWithFlags failed:" + std::string(cudaGetErrorString(stream_error));
        workspace.stream = nullptr;
        return false;
    }
    return true;
}

bool EnsureDeviceBuffer(Element*& buffer, size_t& capacity, size_t required, std::string& error)
{
    if (capacity >= required && buffer != nullptr) {
        return true;
    }

    cudaFree(buffer);
    buffer = nullptr;
    capacity = 0;

    if (required == 0) {
        return true;
    }

    const cudaError_t alloc_error = cudaMalloc(&buffer, required * sizeof(Element));
    if (alloc_error != cudaSuccess) {
        error = "cudaMalloc failed:" + std::string(cudaGetErrorString(alloc_error));
        return false;
    }

    capacity = required;
    return true;
}

bool EnsureSeedBuffers(OracleWorkspace& workspace, size_t required, std::string& error)
{
    if (workspace.batch_seed_capacity >= required &&
        workspace.batch_seed_el != nullptr &&
        workspace.batch_seed_er != nullptr &&
        workspace.batch_seed_fl != nullptr &&
        workspace.batch_seed_fr != nullptr &&
        workspace.batch_seed_cv != nullptr) {
        return true;
    }

    cudaFree(workspace.batch_seed_el);
    cudaFree(workspace.batch_seed_er);
    cudaFree(workspace.batch_seed_fl);
    cudaFree(workspace.batch_seed_fr);
    cudaFree(workspace.batch_seed_cv);
    workspace.batch_seed_el = nullptr;
    workspace.batch_seed_er = nullptr;
    workspace.batch_seed_fl = nullptr;
    workspace.batch_seed_fr = nullptr;
    workspace.batch_seed_cv = nullptr;
    workspace.batch_seed_capacity = 0;

    if (required == 0) {
        return true;
    }

    cudaError_t alloc_error = cudaMalloc(&workspace.batch_seed_el, required * sizeof(OracleSeedBytes));
    if (alloc_error == cudaSuccess) alloc_error = cudaMalloc(&workspace.batch_seed_er, required * sizeof(OracleSeedBytes));
    if (alloc_error == cudaSuccess) alloc_error = cudaMalloc(&workspace.batch_seed_fl, required * sizeof(OracleSeedBytes));
    if (alloc_error == cudaSuccess) alloc_error = cudaMalloc(&workspace.batch_seed_fr, required * sizeof(OracleSeedBytes));
    if (alloc_error == cudaSuccess) alloc_error = cudaMalloc(&workspace.batch_seed_cv, required * sizeof(OracleSeedBytes));
    if (alloc_error != cudaSuccess) {
        cudaFree(workspace.batch_seed_el);
        cudaFree(workspace.batch_seed_er);
        cudaFree(workspace.batch_seed_fl);
        cudaFree(workspace.batch_seed_fr);
        cudaFree(workspace.batch_seed_cv);
        workspace.batch_seed_el = nullptr;
        workspace.batch_seed_er = nullptr;
        workspace.batch_seed_fl = nullptr;
        workspace.batch_seed_fr = nullptr;
        workspace.batch_seed_cv = nullptr;
        error = "cudaMalloc batch seed buffers failed:" + std::string(cudaGetErrorString(alloc_error));
        return false;
    }

    workspace.batch_seed_capacity = required;
    return true;
}

bool EnsureOutputBuffers(OracleWorkspace& workspace,
                         size_t noise_words,
                         size_t compress_words,
                         std::string& error)
{
    const bool reused = workspace.out_e_l != nullptr &&
        workspace.out_e_r != nullptr &&
        workspace.out_f_l != nullptr &&
        workspace.out_f_r != nullptr &&
        workspace.out_cv != nullptr &&
        workspace.noise_capacity >= noise_words &&
        workspace.compress_capacity >= compress_words;

    if (reused) {
        g_profile.pool_initialized.store(true, std::memory_order_relaxed);
        g_profile.reuse_events.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    cudaFree(workspace.out_e_l);
    cudaFree(workspace.out_e_r);
    cudaFree(workspace.out_f_l);
    cudaFree(workspace.out_f_r);
    workspace.out_e_l = nullptr;
    workspace.out_e_r = nullptr;
    workspace.out_f_l = nullptr;
    workspace.out_f_r = nullptr;
    workspace.noise_capacity = 0;

    if (noise_words != 0) {
        cudaError_t alloc_error = cudaMalloc(&workspace.out_e_l, noise_words * sizeof(Element));
        if (alloc_error == cudaSuccess) {
            alloc_error = cudaMalloc(&workspace.out_e_r, noise_words * sizeof(Element));
        }
        if (alloc_error == cudaSuccess) {
            alloc_error = cudaMalloc(&workspace.out_f_l, noise_words * sizeof(Element));
        }
        if (alloc_error == cudaSuccess) {
            alloc_error = cudaMalloc(&workspace.out_f_r, noise_words * sizeof(Element));
        }
        if (alloc_error != cudaSuccess) {
            cudaFree(workspace.out_e_l);
            cudaFree(workspace.out_e_r);
            cudaFree(workspace.out_f_l);
            cudaFree(workspace.out_f_r);
            workspace.out_e_l = nullptr;
            workspace.out_e_r = nullptr;
            workspace.out_f_l = nullptr;
            workspace.out_f_r = nullptr;
            error = "cudaMalloc failed:" + std::string(cudaGetErrorString(alloc_error));
            return false;
        }
        workspace.noise_capacity = noise_words;
    }

    if (!EnsureDeviceBuffer(workspace.out_cv, workspace.compress_capacity, compress_words, error)) {
        cudaFree(workspace.out_e_l);
        cudaFree(workspace.out_e_r);
        cudaFree(workspace.out_f_l);
        cudaFree(workspace.out_f_r);
        workspace.out_e_l = nullptr;
        workspace.out_e_r = nullptr;
        workspace.out_f_l = nullptr;
        workspace.out_f_r = nullptr;
        workspace.noise_capacity = 0;
        return false;
    }

    g_profile.pool_initialized.store(true, std::memory_order_relaxed);
    g_profile.allocation_events.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool ValidateInputGenerationRequest(const MatMulInputGenerationRequest& request,
                                    std::string& error,
                                    uint32_t& noise_words,
                                    uint32_t& compress_words)
{
    if (request.n == 0 || request.b == 0 || request.r == 0) {
        error = "invalid dimensions for GPU input generation";
        return false;
    }
    if (request.r > request.n) {
        error = "noise rank exceeds matrix dimension";
        return false;
    }
    if ((request.n % request.b) != 0) {
        error = "matrix dimension must be divisible by transcript block size";
        return false;
    }

    const uint64_t noise_words64 = static_cast<uint64_t>(request.n) * request.r;
    const uint64_t compress_words64 = static_cast<uint64_t>(request.b) * request.b;
    if (noise_words64 > std::numeric_limits<uint32_t>::max() ||
        compress_words64 > std::numeric_limits<uint32_t>::max()) {
        error = "input generation dimensions exceed supported bounds";
        return false;
    }

    noise_words = static_cast<uint32_t>(noise_words64);
    compress_words = static_cast<uint32_t>(compress_words64);
    return true;
}

bool ValidateInputGenerationDeviceBatchRequest(const MatMulInputGenerationDeviceBatchRequest& request,
                                               std::string& error,
                                               uint32_t& noise_words,
                                               uint32_t& compress_words)
{
    if (request.batch_size == 0 || request.sigmas == nullptr) {
        error = "invalid CUDA input generation batch request";
        return false;
    }
    return ValidateInputGenerationRequest(
        {
            .n = request.n,
            .b = request.b,
            .r = request.r,
            .sigma = request.sigmas[0],
        },
        error,
        noise_words,
        compress_words);
}

void UpdateProfile(double encode_noise_us,
                   double encode_compress_us,
                   double submit_wait_us,
                   const char* reason)
{
    {
        std::lock_guard<std::mutex> lock(g_profile.mutex);
        g_profile.last_encode_noise_us = encode_noise_us;
        g_profile.last_encode_compress_us = encode_compress_us;
        g_profile.last_submit_wait_us = submit_wait_us;
        g_profile.last_gpu_generation_ms = submit_wait_us / 1000.0;
        g_profile.reason = reason;
    }
    g_profile.samples.fetch_add(1, std::memory_order_relaxed);
}

bool EnsureGeneratedInputsDeviceBuffers(DeviceInputPoolSlot& slot,
                                        int device_index,
                                        uint32_t n,
                                        uint32_t b,
                                        uint32_t r,
                                        uint32_t noise_words,
                                        uint32_t compress_words,
                                        std::string& error,
                                        bool& allocated)
{
    auto& inputs = slot.inputs;
    if (inputs.device_index != device_index && inputs.device_index >= 0) {
        cudaSetDevice(inputs.device_index);
        if (inputs.ready_event != nullptr) {
            cudaEventDestroy(reinterpret_cast<cudaEvent_t>(inputs.ready_event));
            inputs.ready_event = nullptr;
        }
        cudaFree(inputs.storage);
        inputs.storage = nullptr;
        inputs.noise_e_l = nullptr;
        inputs.noise_e_r = nullptr;
        inputs.noise_f_l = nullptr;
        inputs.noise_f_r = nullptr;
        inputs.compress_vec = nullptr;
        slot.storage_capacity_words = 0;
    }

    if (inputs.device_index != device_index) {
        cudaSetDevice(device_index);
    }

    inputs.device_index = device_index;
    inputs.n = n;
    inputs.b = b;
    inputs.r = r;
    inputs.noise_words = noise_words;
    inputs.compress_words = compress_words;

    const size_t total_words =
        static_cast<size_t>(noise_words) * 4U + compress_words;
    if (total_words == 0) {
        return true;
    }

    if (inputs.storage == nullptr || slot.storage_capacity_words < total_words) {
        cudaFree(inputs.storage);
        inputs.storage = nullptr;
        inputs.noise_e_l = nullptr;
        inputs.noise_e_r = nullptr;
        inputs.noise_f_l = nullptr;
        inputs.noise_f_r = nullptr;
        inputs.compress_vec = nullptr;

        const cudaError_t alloc_error = cudaMalloc(&inputs.storage, total_words * sizeof(Element));
        if (alloc_error != cudaSuccess) {
            error = "cudaMalloc failed:" + std::string(cudaGetErrorString(alloc_error));
            return false;
        }
        allocated = true;
        slot.storage_capacity_words = total_words;
    }

    inputs.noise_e_l = inputs.storage;
    inputs.noise_e_r = inputs.noise_e_l + noise_words;
    inputs.noise_f_l = inputs.noise_e_r + noise_words;
    inputs.noise_f_r = inputs.noise_f_l + noise_words;
    inputs.compress_vec = inputs.noise_f_r + noise_words;
    return true;
}

bool EnsureGeneratedInputsReadyEvent(MatMulGeneratedInputsDevice& inputs, std::string& error)
{
    if (inputs.ready_event != nullptr) {
        return true;
    }

    cudaEvent_t event_handle{nullptr};
    const cudaError_t event_error = cudaEventCreateWithFlags(&event_handle, cudaEventDisableTiming);
    if (event_error != cudaSuccess) {
        error = "cudaEventCreateWithFlags failed:" + std::string(cudaGetErrorString(event_error));
        return false;
    }

    inputs.ready_event = reinterpret_cast<void*>(event_handle);
    return true;
}

std::shared_ptr<const MatMulGeneratedInputsDevice> AcquireGeneratedInputsDevice(int device_index,
                                                                                uint32_t n,
                                                                                uint32_t b,
                                                                                uint32_t r,
                                                                                uint32_t noise_words,
                                                                                uint32_t compress_words,
                                                                                std::string& error)
{
    auto& context = GetDeviceInputPoolContext();
    std::unique_lock<std::mutex> lock(context.mutex);

    DeviceInputPoolSlot* slot_ptr{nullptr};
    bool reused_existing_slot{false};
    for (size_t offset = 0; offset < context.slots.size(); ++offset) {
        const size_t slot_index = (context.next_slot + offset) % context.slots.size();
        auto& slot = context.slots[slot_index];
        if (slot->in_use) {
            continue;
        }
        slot->in_use = true;
        context.next_slot = static_cast<uint32_t>((slot_index + 1) % std::max<size_t>(context.slots.size(), 1));
        slot_ptr = slot.get();
        reused_existing_slot = true;
        break;
    }

    if (slot_ptr == nullptr) {
        auto slot = std::make_unique<DeviceInputPoolSlot>();
        slot->in_use = true;
        slot_ptr = slot.get();
        context.slots.push_back(std::move(slot));
        context.next_slot = static_cast<uint32_t>(context.slots.size() % std::max<size_t>(context.slots.size(), 1));
    }

    lock.unlock();

    bool allocated_buffers{false};
    if (!EnsureGeneratedInputsDeviceBuffers(
            *slot_ptr,
            device_index,
            n,
            b,
            r,
            noise_words,
            compress_words,
            error,
            allocated_buffers)) {
        std::lock_guard<std::mutex> relock(context.mutex);
        slot_ptr->in_use = false;
        return {};
    }

    g_profile.pool_initialized.store(true, std::memory_order_relaxed);
    if (allocated_buffers || !reused_existing_slot) {
        g_profile.allocation_events.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_profile.reuse_events.fetch_add(1, std::memory_order_relaxed);
    }

    auto holder = std::shared_ptr<DeviceInputPoolSlot>(
        slot_ptr,
        [&context](DeviceInputPoolSlot* slot) {
            std::lock_guard<std::mutex> lock(context.mutex);
            slot->in_use = false;
        });
    return std::shared_ptr<const MatMulGeneratedInputsDevice>(holder, &slot_ptr->inputs);
}

std::shared_ptr<DeviceInputBatchPoolSlot> AcquireGeneratedInputsDeviceBatchSlot(int device_index,
                                                                               uint32_t count,
                                                                               size_t required_storage_words,
                                                                               std::string& error)
{
    auto& context = GetDeviceInputBatchPoolContext();
    std::unique_lock<std::mutex> lock(context.mutex);

    DeviceInputBatchPoolSlot* slot_ptr{nullptr};
    bool reused_existing_slot{false};
    for (size_t offset = 0; offset < context.slots.size(); ++offset) {
        const size_t slot_index = (context.next_slot + offset) % context.slots.size();
        auto& slot = context.slots[slot_index];
        if (slot->in_use) {
            continue;
        }
        slot->in_use = true;
        context.next_slot = static_cast<uint32_t>((slot_index + 1) % std::max<size_t>(context.slots.size(), 1));
        slot_ptr = slot.get();
        reused_existing_slot = true;
        break;
    }

    if (slot_ptr == nullptr) {
        auto slot = std::make_unique<DeviceInputBatchPoolSlot>();
        slot->in_use = true;
        slot_ptr = slot.get();
        context.slots.push_back(std::move(slot));
        context.next_slot = static_cast<uint32_t>(context.slots.size() % std::max<size_t>(context.slots.size(), 1));
    }

    lock.unlock();

    auto release_slot = [&context, slot_ptr]() {
        std::lock_guard<std::mutex> relock(context.mutex);
        slot_ptr->in_use = false;
    };

    bool allocated_buffers = !reused_existing_slot;
    if (slot_ptr->device_index != device_index && slot_ptr->device_index >= 0) {
        cudaSetDevice(slot_ptr->device_index);
        if (slot_ptr->ready_event != nullptr) {
            cudaEventDestroy(slot_ptr->ready_event);
            slot_ptr->ready_event = nullptr;
        }
        cudaFree(slot_ptr->storage);
        slot_ptr->storage = nullptr;
        slot_ptr->storage_capacity_words = 0;
        slot_ptr->views.reset();
        slot_ptr->view_capacity = 0;
        allocated_buffers = true;
    }

    cudaError_t cuda_error = cudaSetDevice(device_index);
    if (cuda_error != cudaSuccess) {
        error = "cudaSetDevice failed:" + std::string(cudaGetErrorString(cuda_error));
        release_slot();
        return {};
    }
    slot_ptr->device_index = device_index;

    if (slot_ptr->ready_event == nullptr) {
        cuda_error = cudaEventCreateWithFlags(&slot_ptr->ready_event, cudaEventDisableTiming);
        if (cuda_error != cudaSuccess) {
            error = "cudaEventCreateWithFlags batch generated inputs failed:" +
                std::string(cudaGetErrorString(cuda_error));
            release_slot();
            return {};
        }
        allocated_buffers = true;
    }

    if (required_storage_words != 0 &&
        (slot_ptr->storage == nullptr || slot_ptr->storage_capacity_words < required_storage_words)) {
        cudaFree(slot_ptr->storage);
        slot_ptr->storage = nullptr;
        slot_ptr->storage_capacity_words = 0;

        cuda_error = cudaMalloc(&slot_ptr->storage, required_storage_words * sizeof(Element));
        if (cuda_error != cudaSuccess) {
            error = "cudaMalloc batch generated inputs failed:" + std::string(cudaGetErrorString(cuda_error));
            release_slot();
            return {};
        }
        slot_ptr->storage_capacity_words = required_storage_words;
        allocated_buffers = true;
    }

    if (slot_ptr->view_capacity < count) {
        slot_ptr->views = std::make_unique<MatMulGeneratedInputsDevice[]>(count);
        slot_ptr->view_capacity = count;
        allocated_buffers = true;
    }
    slot_ptr->count = count;

    g_profile.pool_initialized.store(true, std::memory_order_relaxed);
    if (allocated_buffers) {
        g_profile.allocation_events.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_profile.reuse_events.fetch_add(1, std::memory_order_relaxed);
    }

    return std::shared_ptr<DeviceInputBatchPoolSlot>(
        slot_ptr,
        [&context](DeviceInputBatchPoolSlot* slot) {
            std::lock_guard<std::mutex> relock(context.mutex);
            slot->in_use = false;
        });
}

__device__ inline uint32_t RotR(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32U - n));
}

__device__ inline uint32_t ShaCh(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ ((~x) & z);
}

__device__ inline uint32_t ShaMaj(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ inline uint32_t ShaBSig0(uint32_t x)
{
    return RotR(x, 2U) ^ RotR(x, 13U) ^ RotR(x, 22U);
}

__device__ inline uint32_t ShaBSig1(uint32_t x)
{
    return RotR(x, 6U) ^ RotR(x, 11U) ^ RotR(x, 25U);
}

__device__ inline uint32_t ShaSSig0(uint32_t x)
{
    return RotR(x, 7U) ^ RotR(x, 18U) ^ (x >> 3U);
}

__device__ inline uint32_t ShaSSig1(uint32_t x)
{
    return RotR(x, 17U) ^ RotR(x, 19U) ^ (x >> 10U);
}

__device__ __constant__ uint32_t SHA256_K[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

__device__ inline void Sha256Init(uint32_t state[8])
{
    state[0] = 0x6a09e667U;
    state[1] = 0xbb67ae85U;
    state[2] = 0x3c6ef372U;
    state[3] = 0xa54ff53aU;
    state[4] = 0x510e527fU;
    state[5] = 0x9b05688cU;
    state[6] = 0x1f83d9abU;
    state[7] = 0x5be0cd19U;
}

__device__ inline void SetByte(uint32_t w[64], uint32_t offset, uint32_t byte)
{
    const uint32_t word_index = offset >> 2U;
    const uint32_t shift = (3U - (offset & 3U)) * 8U;
    w[word_index] |= (byte & 0xffU) << shift;
}

__device__ inline uint32_t Bswap32(uint32_t x)
{
    return ((x & 0x000000ffU) << 24U) |
        ((x & 0x0000ff00U) << 8U) |
        ((x & 0x00ff0000U) >> 8U) |
        ((x & 0xff000000U) >> 24U);
}

__device__ __forceinline__ uint32_t LoadBE32(const uint8_t* bytes)
{
    return (static_cast<uint32_t>(bytes[0]) << 24U) |
        (static_cast<uint32_t>(bytes[1]) << 16U) |
        (static_cast<uint32_t>(bytes[2]) << 8U) |
        static_cast<uint32_t>(bytes[3]);
}

__device__ __forceinline__ uint8_t DigestWordByte(const uint32_t digest_words[8], uint32_t byte_index)
{
    const uint32_t word = digest_words[byte_index >> 2U];
    const uint32_t shift = (3U - (byte_index & 3U)) * 8U;
    return static_cast<uint8_t>((word >> shift) & 0xffU);
}

__device__ __forceinline__ void SetLE16BlockByte(uint32_t w[16], uint32_t offset, uint16_t value)
{
    SetByte(w, offset, value & 0xffU);
    SetByte(w, offset + 1U, (value >> 8U) & 0xffU);
}

__device__ __forceinline__ void SetLE32BlockByte(uint32_t w[16], uint32_t offset, uint32_t value)
{
    SetByte(w, offset, value & 0xffU);
    SetByte(w, offset + 1U, (value >> 8U) & 0xffU);
    SetByte(w, offset + 2U, (value >> 16U) & 0xffU);
    SetByte(w, offset + 3U, (value >> 24U) & 0xffU);
}

__device__ __forceinline__ void SetLE64BlockByte(uint32_t w[16], uint32_t offset, uint64_t value)
{
    SetLE32BlockByte(w, offset, static_cast<uint32_t>(value));
    SetLE32BlockByte(w, offset + 4U, static_cast<uint32_t>(value >> 32U));
}

// WINDOWED SHA-256 compression: 16-word sliding schedule instead of a 64-word
// array. Byte-identical output (validated 200k nonces, 0 mismatches) but halves
// the per-thread local-memory stack frame (448->224 B), ~2x faster scanner.
__device__ inline void Sha256Compress(uint32_t state[8], uint32_t w[16])
{
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    #pragma unroll
    for (uint32_t t = 0; t < 64; ++t) {
        uint32_t wt;
        if (t < 16) {
            wt = w[t];
        } else {
            wt = ShaSSig1(w[(t - 2) & 15U]) + w[(t - 7) & 15U] + ShaSSig0(w[(t - 15) & 15U]) + w[(t - 16) & 15U];
            w[t & 15U] = wt;
        }
        const uint32_t t1 = h + ShaBSig1(e) + ShaCh(e, f, g) + SHA256_K[t] + wt;
        const uint32_t t2 = ShaBSig0(a) + ShaMaj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__ inline uint32_t CandidateFromSeedAndIndex(const OracleSeedBytes& seed,
                                                     uint32_t index,
                                                     bool with_retry,
                                                     uint32_t retry)
{
    uint32_t w[16] = {};
    for (uint32_t i = 0; i < 32; ++i) {
        SetByte(w, i, seed.data[31U - i]);
    }

    SetByte(w, 32U, index & 0xffU);
    SetByte(w, 33U, (index >> 8U) & 0xffU);
    SetByte(w, 34U, (index >> 16U) & 0xffU);
    SetByte(w, 35U, (index >> 24U) & 0xffU);

    uint32_t message_len = 36U;
    if (with_retry) {
        SetByte(w, 36U, retry & 0xffU);
        SetByte(w, 37U, (retry >> 8U) & 0xffU);
        SetByte(w, 38U, (retry >> 16U) & 0xffU);
        SetByte(w, 39U, (retry >> 24U) & 0xffU);
        message_len = 40U;
    }

    SetByte(w, message_len, 0x80U);
    w[15] = message_len * 8U;

    uint32_t state[8];
    Sha256Init(state);
    Sha256Compress(state, w);
    return Bswap32(state[0]) & MODULUS;
}

__device__ inline uint32_t FallbackCandidate(const OracleSeedBytes& seed, uint32_t index)
{
    uint32_t w[16] = {};
    for (uint32_t i = 0; i < 32; ++i) {
        SetByte(w, i, seed.data[31U - i]);
    }

    SetByte(w, 32U, index & 0xffU);
    SetByte(w, 33U, (index >> 8U) & 0xffU);
    SetByte(w, 34U, (index >> 16U) & 0xffU);
    SetByte(w, 35U, (index >> 24U) & 0xffU);

    constexpr uint8_t fallback_tag[15] = {
        'o', 'r', 'a', 'c', 'l', 'e', '-', 'f', 'a', 'l', 'l', 'b', 'a', 'c', 'k'
    };
    for (uint32_t i = 0; i < 15; ++i) {
        SetByte(w, 36U + i, fallback_tag[i]);
    }

    SetByte(w, 51U, 0x80U);
    w[15] = 51U * 8U;

    uint32_t state[8];
    Sha256Init(state);
    Sha256Compress(state, w);
    return Bswap32(state[0]) % MODULUS;
}

__device__ inline uint32_t FromOracle(const OracleSeedBytes& seed, uint32_t index)
{
    for (uint32_t retry = 0; retry < 256; ++retry) {
        const uint32_t candidate = retry == 0
            ? CandidateFromSeedAndIndex(seed, index, false, 0U)
            : CandidateFromSeedAndIndex(seed, index, true, retry);
        if (candidate < MODULUS) {
            return candidate;
        }
    }
    return FallbackCandidate(seed, index);
}

__device__ __forceinline__ void PackOracleSeedWords(const OracleSeedBytes& seed, uint32_t schedule[ORACLE_SEED_MIDSTATE_WORDS])
{
    #pragma unroll
    for (uint32_t i = 0; i < 8U; ++i) {
        schedule[i] = 0U;
    }
    #pragma unroll
    for (uint32_t i = 0; i < 32U; ++i) {
        SetByte(schedule, i, seed.data[31U - i]);
    }
}

__device__ __forceinline__ void PrecomputeOracleSeedMidstate(const OracleSeedBytes& seed, uint32_t* out)
{
    uint32_t schedule[ORACLE_SEED_MIDSTATE_WORDS];
    PackOracleSeedWords(seed, schedule);

    uint32_t a = 0x6a09e667U;
    uint32_t b = 0xbb67ae85U;
    uint32_t c = 0x3c6ef372U;
    uint32_t d = 0xa54ff53aU;
    uint32_t e = 0x510e527fU;
    uint32_t f = 0x9b05688cU;
    uint32_t g = 0x1f83d9abU;
    uint32_t h = 0x5be0cd19U;

    #pragma unroll
    for (uint32_t t = 0; t < 8U; ++t) {
        const uint32_t t1 = h + ShaBSig1(e) + ShaCh(e, f, g) + SHA256_K[t] + schedule[t];
        const uint32_t t2 = ShaBSig0(a) + ShaMaj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    #pragma unroll
    for (uint32_t t = 8U; t < 16U; ++t) {
        schedule[t] = 0U;
    }
    schedule[9] = 0x80000000U;
    schedule[15] = 36U * 8U;
    #pragma unroll
    for (uint32_t t = 16U; t < ORACLE_SEED_MIDSTATE_WORDS; ++t) {
        schedule[t] = ShaSSig1(schedule[t - 2U]) + schedule[t - 7U] +
            ShaSSig0(schedule[t - 15U]) + schedule[t - 16U];
    }

    out[0] = schedule[0];
    out[1] = schedule[1];
    out[2] = schedule[2];
    out[3] = schedule[3];
    out[4] = schedule[4];
    out[5] = schedule[5];
    out[6] = schedule[6];
    out[7] = schedule[7];
    out[8] = a;
    out[9] = b;
    out[10] = c;
    out[11] = d;
    out[12] = e;
    out[13] = f;
    out[14] = g;
    out[15] = h;
    out[16] = schedule[16];
    out[17] = schedule[17];
    out[18] = schedule[18];
    out[19] = schedule[19];
    out[20] = schedule[20];
    out[21] = schedule[21];
    out[22] = schedule[22];
}

__device__ __forceinline__ uint32_t CandidateFromOracleMidstate(const uint32_t* midstate, uint32_t index)
{
    uint32_t w[16];
    w[0] = midstate[0];
    w[1] = midstate[1];
    w[2] = midstate[2];
    w[3] = midstate[3];
    w[4] = midstate[4];
    w[5] = midstate[5];
    w[6] = midstate[6];
    w[7] = midstate[7];
    #pragma unroll
    for (uint32_t i = 8U; i < 16U; ++i) {
        w[i] = 0U;
    }
    w[8] = Bswap32(index);
    w[9] = 0x80000000U;
    w[15] = 36U * 8U;

    uint32_t a = midstate[8];
    uint32_t b = midstate[9];
    uint32_t c = midstate[10];
    uint32_t d = midstate[11];
    uint32_t e = midstate[12];
    uint32_t f = midstate[13];
    uint32_t g = midstate[14];
    uint32_t h = midstate[15];

    #pragma unroll
    for (uint32_t t = 8U; t < 64U; ++t) {
        uint32_t wt;
        if (t < 16U) {
            wt = w[t];
        } else if (t < ORACLE_SEED_MIDSTATE_WORDS) {
            wt = midstate[t];
            w[t & 15U] = wt;
        } else {
            wt = ShaSSig1(w[(t - 2U) & 15U]) + w[(t - 7U) & 15U] +
                ShaSSig0(w[(t - 15U) & 15U]) + w[(t - 16U) & 15U];
            w[t & 15U] = wt;
        }
        const uint32_t t1 = h + ShaBSig1(e) + ShaCh(e, f, g) + SHA256_K[t] + wt;
        const uint32_t t2 = ShaBSig0(a) + ShaMaj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    return Bswap32(0x6a09e667U + a) & MODULUS;
}

__device__ __forceinline__ Element FromOracleMidstateOrFallback(const OracleSeedBytes* seeds,
                                                                uint32_t batch,
                                                                const uint32_t* midstate,
                                                                uint32_t index)
{
    const uint32_t candidate = CandidateFromOracleMidstate(midstate, index);
    return candidate < MODULUS ? candidate : FromOracle(seeds[batch], index);
}

__device__ __forceinline__ const uint32_t* OracleInputSeedMidstate(const uint32_t* midstates,
                                                                   uint32_t batch_size,
                                                                   uint32_t kind,
                                                                   uint32_t batch)
{
    return midstates +
        (static_cast<size_t>(kind) * batch_size + batch) * ORACLE_SEED_MIDSTATE_WORDS;
}

__device__ inline void Sha256Bytes(const uint8_t* message, uint32_t message_len, uint8_t out[32])
{
    uint32_t state[8];
    Sha256Init(state);

    const uint32_t total_blocks = (message_len + 9U + 63U) / 64U;
    const uint64_t bit_len = static_cast<uint64_t>(message_len) * 8U;
    for (uint32_t block = 0; block < total_blocks; ++block) {
        uint32_t w[16] = {};
        for (uint32_t word = 0; word < 16; ++word) {
            uint32_t packed{0};
            for (uint32_t byte = 0; byte < 4; ++byte) {
                const uint32_t message_index = block * 64U + word * 4U + byte;
                uint8_t value{0};
                if (message_index < message_len) {
                    value = message[message_index];
                } else if (message_index == message_len) {
                    value = 0x80U;
                } else {
                    const uint32_t length_start = total_blocks * 64U - 8U;
                    if (message_index >= length_start) {
                        const uint32_t shift = (7U - (message_index - length_start)) * 8U;
                        value = static_cast<uint8_t>((bit_len >> shift) & 0xffU);
                    }
                }
                packed = (packed << 8U) | value;
            }
            w[word] = packed;
        }
        Sha256Compress(state, w);
    }

    for (uint32_t i = 0; i < 8; ++i) {
        out[i * 4U] = static_cast<uint8_t>((state[i] >> 24U) & 0xffU);
        out[i * 4U + 1U] = static_cast<uint8_t>((state[i] >> 16U) & 0xffU);
        out[i * 4U + 2U] = static_cast<uint8_t>((state[i] >> 8U) & 0xffU);
        out[i * 4U + 3U] = static_cast<uint8_t>(state[i] & 0xffU);
    }
}

// Midstate after the FIRST 64-byte block of a >64-byte message. Both nonce-seed
// scan messages (seed_v2: 110 B, header-hash: 150 B) keep every nonce-dependent
// byte past offset 63 (nonce at 99 / 76, `which` at 109), so this midstate is
// constant across a whole scan window and is computed once per CUDA block.
__device__ inline void Sha256Block0Midstate(const uint8_t* message, uint32_t state[8])
{
    Sha256Init(state);
    uint32_t w[16];
    for (uint32_t word = 0; word < 16; ++word) {
        uint32_t packed{0};
        for (uint32_t byte = 0; byte < 4; ++byte) {
            packed = (packed << 8U) | message[word * 4U + byte];
        }
        w[word] = packed;
    }
    Sha256Compress(state, w);
}

// Sha256Bytes, resumed after block 0 from a precomputed midstate. Byte-identical
// to Sha256Bytes for any message longer than 64 bytes whose first block matches
// the midstate's input.
__device__ inline void Sha256BytesFromMidstate(const uint32_t midstate[8],
                                               const uint8_t* message,
                                               uint32_t message_len,
                                               uint8_t out[32])
{
    uint32_t state[8];
    for (uint32_t i = 0; i < 8; ++i) {
        state[i] = midstate[i];
    }

    const uint32_t total_blocks = (message_len + 9U + 63U) / 64U;
    const uint64_t bit_len = static_cast<uint64_t>(message_len) * 8U;
    for (uint32_t block = 1; block < total_blocks; ++block) {
        uint32_t w[16] = {};
        for (uint32_t word = 0; word < 16; ++word) {
            uint32_t packed{0};
            for (uint32_t byte = 0; byte < 4; ++byte) {
                const uint32_t message_index = block * 64U + word * 4U + byte;
                uint8_t value{0};
                if (message_index < message_len) {
                    value = message[message_index];
                } else if (message_index == message_len) {
                    value = 0x80U;
                } else {
                    const uint32_t length_start = total_blocks * 64U - 8U;
                    if (message_index >= length_start) {
                        const uint32_t shift = (7U - (message_index - length_start)) * 8U;
                        value = static_cast<uint8_t>((bit_len >> shift) & 0xffU);
                    }
                }
                packed = (packed << 8U) | value;
            }
            w[word] = packed;
        }
        Sha256Compress(state, w);
    }

    for (uint32_t i = 0; i < 8; ++i) {
        out[i * 4U] = static_cast<uint8_t>((state[i] >> 24U) & 0xffU);
        out[i * 4U + 1U] = static_cast<uint8_t>((state[i] >> 16U) & 0xffU);
        out[i * 4U + 2U] = static_cast<uint8_t>((state[i] >> 8U) & 0xffU);
        out[i * 4U + 3U] = static_cast<uint8_t>(state[i] & 0xffU);
    }
}

__device__ inline void AppendByte(uint8_t* message, uint32_t& offset, uint8_t value)
{
    message[offset++] = value;
}

__device__ inline void AppendBytes(uint8_t* message, uint32_t& offset, const uint8_t* data, uint32_t size)
{
    for (uint32_t i = 0; i < size; ++i) {
        message[offset++] = data[i];
    }
}

__device__ inline void AppendLE16(uint8_t* message, uint32_t& offset, uint16_t value)
{
    AppendByte(message, offset, static_cast<uint8_t>(value & 0xffU));
    AppendByte(message, offset, static_cast<uint8_t>((value >> 8U) & 0xffU));
}

__device__ inline void AppendLE32(uint8_t* message, uint32_t& offset, uint32_t value)
{
    AppendByte(message, offset, static_cast<uint8_t>(value & 0xffU));
    AppendByte(message, offset, static_cast<uint8_t>((value >> 8U) & 0xffU));
    AppendByte(message, offset, static_cast<uint8_t>((value >> 16U) & 0xffU));
    AppendByte(message, offset, static_cast<uint8_t>((value >> 24U) & 0xffU));
}

__device__ inline void AppendLE64(uint8_t* message, uint32_t& offset, uint64_t value)
{
    for (uint32_t i = 0; i < 8; ++i) {
        AppendByte(message, offset, static_cast<uint8_t>((value >> (i * 8U)) & 0xffU));
    }
}

// Builds the seed_v2 message; the nonce lands at offset 99 and `which` at 109,
// both inside the SECOND SHA-256 block. Returns the message length (110).
__device__ inline uint32_t BuildMatMulSeedV2Message(const OracleSeedBytes& previous_block_hash,
                                                    const OracleSeedBytes& merkle_root,
                                                    uint32_t height,
                                                    uint32_t version,
                                                    uint32_t time,
                                                    uint32_t bits,
                                                    uint64_t nonce,
                                                    uint16_t matmul_dim,
                                                    uint8_t which,
                                                    uint8_t message[110])
{
    uint32_t offset{0};
    constexpr char TAG[] = "BTX_MATMUL_SEED_V2";
    AppendByte(message, offset, 18U);
    for (uint32_t i = 0; i < 18U; ++i) {
        AppendByte(message, offset, static_cast<uint8_t>(TAG[i]));
    }
    AppendBytes(message, offset, previous_block_hash.data, 32U);
    AppendLE32(message, offset, height);
    AppendLE32(message, offset, version);
    AppendBytes(message, offset, merkle_root.data, 32U);
    AppendLE32(message, offset, time);
    AppendLE32(message, offset, bits);
    AppendLE64(message, offset, nonce);
    AppendLE16(message, offset, matmul_dim);
    AppendByte(message, offset, which);
    return offset;
}

// Builds the seed_v3 message; the nonce lands at offset 107 and `which` at 117,
// both inside the SECOND SHA-256 block. Returns the message length (118).
__device__ inline uint32_t BuildMatMulSeedV3Message(const OracleSeedBytes& previous_block_hash,
                                                    const OracleSeedBytes& merkle_root,
                                                    uint64_t parent_median_time_past,
                                                    uint32_t height,
                                                    uint32_t version,
                                                    uint32_t time,
                                                    uint32_t bits,
                                                    uint64_t nonce,
                                                    uint16_t matmul_dim,
                                                    uint8_t which,
                                                    uint8_t message[118])
{
    uint32_t offset{0};
    constexpr char TAG[] = "BTX_MATMUL_SEED_V3";
    AppendByte(message, offset, 18U);
    for (uint32_t i = 0; i < 18U; ++i) {
        AppendByte(message, offset, static_cast<uint8_t>(TAG[i]));
    }
    AppendBytes(message, offset, previous_block_hash.data, 32U);
    AppendLE64(message, offset, parent_median_time_past);
    AppendLE32(message, offset, height);
    AppendLE32(message, offset, version);
    AppendBytes(message, offset, merkle_root.data, 32U);
    AppendLE32(message, offset, time);
    AppendLE32(message, offset, bits);
    AppendLE64(message, offset, nonce);
    AppendLE16(message, offset, matmul_dim);
    AppendByte(message, offset, which);
    return offset;
}

// Builds the header-hash message; the nonce lands at offset 76 and the seeds at
// 86/118, all inside the second and third SHA-256 blocks. Returns length (150).
__device__ inline uint32_t BuildMatMulHeaderHashMessage(uint32_t version,
                                                        const OracleSeedBytes& previous_block_hash,
                                                        const OracleSeedBytes& merkle_root,
                                                        uint32_t time,
                                                        uint32_t bits,
                                                        uint64_t nonce,
                                                        uint16_t matmul_dim,
                                                        const uint8_t seed_a[32],
                                                        const uint8_t seed_b[32],
                                                        uint8_t message[150])
{
    uint32_t offset{0};
    AppendLE32(message, offset, version);
    AppendBytes(message, offset, previous_block_hash.data, 32U);
    AppendBytes(message, offset, merkle_root.data, 32U);
    AppendLE32(message, offset, time);
    AppendLE32(message, offset, bits);
    AppendLE64(message, offset, nonce);
    AppendLE16(message, offset, matmul_dim);
    AppendBytes(message, offset, seed_a, 32U);
    AppendBytes(message, offset, seed_b, 32U);
    return offset;
}

__device__ __forceinline__ void Sha256SeedV3FromMidstate(const uint32_t seed_midstate[8],
                                                         const OracleSeedBytes& merkle_root,
                                                         uint32_t version,
                                                         uint32_t time,
                                                         uint32_t bits,
                                                         uint64_t nonce,
                                                         uint16_t matmul_dim,
                                                         uint8_t which,
                                                         uint32_t out_words[8])
{
    uint32_t w[16] = {};
    SetByte(w, 0U, (version >> 8U) & 0xffU);
    SetByte(w, 1U, (version >> 16U) & 0xffU);
    SetByte(w, 2U, (version >> 24U) & 0xffU);
    #pragma unroll
    for (uint32_t i = 0; i < 32U; ++i) {
        SetByte(w, 3U + i, merkle_root.data[i]);
    }
    SetLE32BlockByte(w, 35U, time);
    SetLE32BlockByte(w, 39U, bits);
    SetLE64BlockByte(w, 43U, nonce);
    SetLE16BlockByte(w, 51U, matmul_dim);
    SetByte(w, 53U, which);
    SetByte(w, 54U, 0x80U);
    w[15] = 118U * 8U;

    uint32_t state[8];
    #pragma unroll
    for (uint32_t i = 0; i < 8U; ++i) {
        state[i] = seed_midstate[i];
    }
    Sha256Compress(state, w);
    #pragma unroll
    for (uint32_t i = 0; i < 8U; ++i) {
        out_words[i] = state[i];
    }
}

__device__ __forceinline__ void Sha256HeaderHashFromSeedWords(const uint32_t header_midstate[8],
                                                              const OracleSeedBytes& merkle_root,
                                                              uint32_t time,
                                                              uint32_t bits,
                                                              uint64_t nonce,
                                                              uint16_t matmul_dim,
                                                              const uint32_t seed_a_words[8],
                                                              const uint32_t seed_b_words[8],
                                                              uint32_t out_words[8])
{
    uint32_t state[8];
    #pragma unroll
    for (uint32_t i = 0; i < 8U; ++i) {
        state[i] = header_midstate[i];
    }

    uint32_t w[16] = {};
    w[0] = LoadBE32(merkle_root.data + 28U);
    w[1] = Bswap32(time);
    w[2] = Bswap32(bits);
    w[3] = Bswap32(static_cast<uint32_t>(nonce));
    w[4] = Bswap32(static_cast<uint32_t>(nonce >> 32U));
    SetLE16BlockByte(w, 20U, matmul_dim);
    #pragma unroll
    for (uint32_t i = 0; i < 32U; ++i) {
        SetByte(w, 22U + i, DigestWordByte(seed_a_words, i));
    }
    #pragma unroll
    for (uint32_t i = 0; i < 10U; ++i) {
        SetByte(w, 54U + i, DigestWordByte(seed_b_words, i));
    }
    Sha256Compress(state, w);

    #pragma unroll
    for (uint32_t i = 0; i < 16U; ++i) {
        w[i] = 0U;
    }
    #pragma unroll
    for (uint32_t i = 10U; i < 32U; ++i) {
        SetByte(w, i - 10U, DigestWordByte(seed_b_words, i));
    }
    SetByte(w, 22U, 0x80U);
    w[15] = 150U * 8U;
    Sha256Compress(state, w);

    #pragma unroll
    for (uint32_t i = 0; i < 8U; ++i) {
        out_words[i] = state[i];
    }
}

__device__ __forceinline__ void Sha256Digest32FromWords(const uint32_t input_words[8],
                                                        uint32_t out_words[8])
{
    uint32_t state[8];
    Sha256Init(state);
    uint32_t w[16] = {};
    #pragma unroll
    for (uint32_t i = 0; i < 8U; ++i) {
        w[i] = input_words[i];
    }
    w[8] = 0x80000000U;
    w[15] = 32U * 8U;
    Sha256Compress(state, w);
    #pragma unroll
    for (uint32_t i = 0; i < 8U; ++i) {
        out_words[i] = state[i];
    }
}

__device__ inline bool Uint256InternalWordsLessOrEqual(const uint32_t lhs_words[8],
                                                       const OracleSeedBytes& rhs)
{
    for (int i = 31; i >= 0; --i) {
        const uint8_t lhs = DigestWordByte(lhs_words, static_cast<uint32_t>(i));
        if (lhs < rhs.data[i]) return true;
        if (lhs > rhs.data[i]) return false;
    }
    return true;
}

__device__ inline bool Uint256InternalBytesLessOrEqual(const uint8_t lhs[32], const OracleSeedBytes& rhs)
{
    for (int i = 31; i >= 0; --i) {
        if (lhs[i] < rhs.data[i]) return true;
        if (lhs[i] > rhs.data[i]) return false;
    }
    return true;
}

__device__ __forceinline__ void StorePreHashScanResult(bool pass,
                                                       uint32_t gid,
                                                       Element* out_flags,
                                                       Element* out_selected_offsets,
                                                       Element* out_selected_count,
                                                       uint32_t max_selected_offsets)
{
    if (out_flags != nullptr) {
        out_flags[gid] = pass ? 1U : 0U;
    }
    if (pass && out_selected_count != nullptr) {
        const Element slot = atomicAdd(out_selected_count, 1U);
        if (slot < max_selected_offsets && out_selected_offsets != nullptr) {
            out_selected_offsets[slot] = gid;
        }
    }
}

__global__ void ScanNonceSeedPreHashKernel(OracleSeedBytes previous_block_hash,
                                           OracleSeedBytes merkle_root,
                                           OracleSeedBytes pre_hash_target,
                                           uint32_t version,
                                           uint32_t height,
                                           uint32_t time,
                                           uint32_t bits,
                                           uint64_t start_nonce,
                                           uint16_t matmul_dim,
                                           uint32_t seed_version,
                                           uint64_t parent_median_time_past,
                                           Element* out_flags,
                                           Element* out_selected_offsets,
                                           Element* out_selected_count,
                                           uint32_t max_selected_offsets,
                                           uint32_t scan_count)
{
    // Block 0 of the seed and header messages is nonce-independent,
    // so its compression is done ONCE per CUDA block and shared: 8 -> 5 SHA-256
    // compressions per nonce. The midstates must be computed before the bounds
    // guard so every thread reaches the barrier.
    __shared__ uint32_t seed_midstate[8];
    __shared__ uint32_t header_midstate[8];
    if (threadIdx.x == 0) {
        uint8_t prefix[150];
        if (seed_version == 3U) {
            BuildMatMulSeedV3Message(
                previous_block_hash, merkle_root, parent_median_time_past,
                height, version, time, bits,
                /*nonce=*/0U, matmul_dim, /*which=*/0U, prefix);
        } else {
            BuildMatMulSeedV2Message(
                previous_block_hash, merkle_root, height, version, time, bits,
                /*nonce=*/0U, matmul_dim, /*which=*/0U, prefix);
        }
        Sha256Block0Midstate(prefix, seed_midstate);
        BuildMatMulHeaderHashMessage(
            version, previous_block_hash, merkle_root, time, bits,
            /*nonce=*/0U, matmul_dim,
            /*seed_a=*/previous_block_hash.data, /*seed_b=*/previous_block_hash.data,
            prefix);
        Sha256Block0Midstate(prefix, header_midstate);
    }
    __syncthreads();

    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= scan_count) {
        return;
    }

    const uint64_t nonce = start_nonce + static_cast<uint64_t>(gid);
    uint8_t message[150];
    uint8_t seed_a[32];
    uint8_t seed_b[32];
    uint8_t header_hash[32];
    uint8_t sigma[32];
    const uint32_t seed_len = seed_version == 3U
        ? BuildMatMulSeedV3Message(
            previous_block_hash, merkle_root, parent_median_time_past,
            height, version, time, bits,
            nonce, matmul_dim, 0U, message)
        : BuildMatMulSeedV2Message(
            previous_block_hash, merkle_root, height, version, time, bits,
            nonce, matmul_dim, 0U, message);
    Sha256BytesFromMidstate(seed_midstate, message, seed_len, seed_a);
    message[seed_len - 1U] = 1U; // `which` is the final byte; the rest is identical
    Sha256BytesFromMidstate(seed_midstate, message, seed_len, seed_b);
    const uint32_t header_len = BuildMatMulHeaderHashMessage(
        version, previous_block_hash, merkle_root, time, bits,
        nonce, matmul_dim, seed_a, seed_b, message);
    Sha256BytesFromMidstate(header_midstate, message, header_len, header_hash);
    Sha256Bytes(header_hash, 32U, sigma);
    StorePreHashScanResult(
        Uint256InternalBytesLessOrEqual(sigma, pre_hash_target),
        gid,
        out_flags,
        out_selected_offsets,
        out_selected_count,
        max_selected_offsets);
}

__global__ void ScanNonceSeedPreHashV3Kernel(OracleSeedBytes previous_block_hash,
                                             OracleSeedBytes merkle_root,
                                             OracleSeedBytes pre_hash_target,
                                             uint32_t version,
                                             uint32_t height,
                                             uint32_t time,
                                             uint32_t bits,
                                             uint64_t start_nonce,
                                             uint16_t matmul_dim,
                                             uint64_t parent_median_time_past,
                                             Element* out_flags,
                                             Element* out_selected_offsets,
                                             Element* out_selected_count,
                                             uint32_t max_selected_offsets,
                                             uint32_t scan_count)
{
    __shared__ uint32_t seed_midstate[8];
    __shared__ uint32_t header_midstate[8];
    if (threadIdx.x == 0) {
        uint8_t prefix[150];
        BuildMatMulSeedV3Message(
            previous_block_hash, merkle_root, parent_median_time_past,
            height, version, time, bits,
            /*nonce=*/0U, matmul_dim, /*which=*/0U, prefix);
        Sha256Block0Midstate(prefix, seed_midstate);
        BuildMatMulHeaderHashMessage(
            version, previous_block_hash, merkle_root, time, bits,
            /*nonce=*/0U, matmul_dim,
            /*seed_a=*/previous_block_hash.data, /*seed_b=*/previous_block_hash.data,
            prefix);
        Sha256Block0Midstate(prefix, header_midstate);
    }
    __syncthreads();

    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= scan_count) {
        return;
    }

    const uint64_t nonce = start_nonce + static_cast<uint64_t>(gid);
    uint32_t seed_a_words[8];
    uint32_t seed_b_words[8];
    uint32_t header_hash_words[8];
    uint32_t sigma_words[8];
    Sha256SeedV3FromMidstate(
        seed_midstate,
        merkle_root,
        version,
        time,
        bits,
        nonce,
        matmul_dim,
        0U,
        seed_a_words);
    Sha256SeedV3FromMidstate(
        seed_midstate,
        merkle_root,
        version,
        time,
        bits,
        nonce,
        matmul_dim,
        1U,
        seed_b_words);
    Sha256HeaderHashFromSeedWords(
        header_midstate,
        merkle_root,
        time,
        bits,
        nonce,
        matmul_dim,
        seed_a_words,
        seed_b_words,
        header_hash_words);
    Sha256Digest32FromWords(header_hash_words, sigma_words);
    StorePreHashScanResult(
        Uint256InternalWordsLessOrEqual(sigma_words, pre_hash_target),
        gid,
        out_flags,
        out_selected_offsets,
        out_selected_count,
        max_selected_offsets);
}

__global__ void GenerateOracleNoiseKernel(OracleSeedBytes seed_el,
                                          OracleSeedBytes seed_er,
                                          OracleSeedBytes seed_fl,
                                          OracleSeedBytes seed_fr,
                                          Element* out_e_l,
                                          Element* out_e_r,
                                          Element* out_f_l,
                                          Element* out_f_r,
                                          uint32_t count)
{
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) {
        return;
    }

    out_e_l[gid] = FromOracle(seed_el, gid);
    out_e_r[gid] = FromOracle(seed_er, gid);
    out_f_l[gid] = FromOracle(seed_fl, gid);
    out_f_r[gid] = FromOracle(seed_fr, gid);
}

__global__ void GenerateOracleVectorKernel(OracleSeedBytes seed_cv,
                                           Element* out,
                                           uint32_t count)
{
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) {
        return;
    }

    out[gid] = FromOracle(seed_cv, gid);
}

__global__ void GenerateOracleNoiseBatchKernel(const OracleSeedBytes* seed_el,
                                               const OracleSeedBytes* seed_er,
                                               const OracleSeedBytes* seed_fl,
                                               const OracleSeedBytes* seed_fr,
                                               Element* storage,
                                               uint32_t stride_words,
                                               uint32_t noise_words,
                                               size_t total_count)
{
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_count) {
        return;
    }
    const uint32_t batch = static_cast<uint32_t>(gid / noise_words);
    const uint32_t local = static_cast<uint32_t>(gid % noise_words);
    Element* base = storage + static_cast<size_t>(batch) * stride_words;
    base[local] = FromOracle(seed_el[batch], local);
    base[noise_words + local] = FromOracle(seed_er[batch], local);
    base[2U * noise_words + local] = FromOracle(seed_fl[batch], local);
    base[3U * noise_words + local] = FromOracle(seed_fr[batch], local);
}

__global__ void GenerateOracleVectorBatchKernel(const OracleSeedBytes* seed_cv,
                                                Element* storage,
                                                uint32_t stride_words,
                                                uint32_t compress_offset_words,
                                                uint32_t compress_words,
                                                size_t total_count)
{
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_count) {
        return;
    }
    const uint32_t batch = static_cast<uint32_t>(gid / compress_words);
    const uint32_t local = static_cast<uint32_t>(gid % compress_words);
    Element* base = storage + static_cast<size_t>(batch) * stride_words + compress_offset_words;
    base[local] = FromOracle(seed_cv[batch], local);
}

__global__ void PrecomputeOracleSeedMidstatesBatchKernel(const OracleSeedBytes* seed_el,
                                                         const OracleSeedBytes* seed_er,
                                                         const OracleSeedBytes* seed_fl,
                                                         const OracleSeedBytes* seed_fr,
                                                         const OracleSeedBytes* seed_cv,
                                                         uint32_t batch_size,
                                                         uint32_t* seed_midstates)
{
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t seed_count = batch_size * ORACLE_INPUT_SEED_KINDS;
    if (gid >= seed_count) {
        return;
    }

    const uint32_t kind = gid / batch_size;
    const uint32_t batch = gid - kind * batch_size;
    const OracleSeedBytes* seeds = seed_el;
    if (kind == 1U) {
        seeds = seed_er;
    } else if (kind == 2U) {
        seeds = seed_fl;
    } else if (kind == 3U) {
        seeds = seed_fr;
    } else if (kind == 4U) {
        seeds = seed_cv;
    }

    PrecomputeOracleSeedMidstate(
        seeds[batch],
        seed_midstates + static_cast<size_t>(gid) * ORACLE_SEED_MIDSTATE_WORDS);
}

__global__ void GenerateOracleNoiseBatchMidstateKernel(const OracleSeedBytes* seed_el,
                                                       const OracleSeedBytes* seed_er,
                                                       const OracleSeedBytes* seed_fl,
                                                       const OracleSeedBytes* seed_fr,
                                                       const uint32_t* seed_midstates,
                                                       Element* storage,
                                                       uint32_t batch_size,
                                                       uint32_t stride_words,
                                                       uint32_t noise_words,
                                                       size_t total_count)
{
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_count) {
        return;
    }
    const uint32_t batch = static_cast<uint32_t>(gid / noise_words);
    const uint32_t local = static_cast<uint32_t>(gid % noise_words);
    Element* base = storage + static_cast<size_t>(batch) * stride_words;
    base[local] = FromOracleMidstateOrFallback(
        seed_el,
        batch,
        OracleInputSeedMidstate(seed_midstates, batch_size, 0U, batch),
        local);
    base[noise_words + local] = FromOracleMidstateOrFallback(
        seed_er,
        batch,
        OracleInputSeedMidstate(seed_midstates, batch_size, 1U, batch),
        local);
    base[2U * noise_words + local] = FromOracleMidstateOrFallback(
        seed_fl,
        batch,
        OracleInputSeedMidstate(seed_midstates, batch_size, 2U, batch),
        local);
    base[3U * noise_words + local] = FromOracleMidstateOrFallback(
        seed_fr,
        batch,
        OracleInputSeedMidstate(seed_midstates, batch_size, 3U, batch),
        local);
}

__global__ void GenerateOracleVectorBatchMidstateKernel(const OracleSeedBytes* seed_cv,
                                                        const uint32_t* seed_midstates,
                                                        Element* storage,
                                                        uint32_t batch_size,
                                                        uint32_t stride_words,
                                                        uint32_t compress_offset_words,
                                                        uint32_t compress_words,
                                                        size_t total_count)
{
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_count) {
        return;
    }
    const uint32_t batch = static_cast<uint32_t>(gid / compress_words);
    const uint32_t local = static_cast<uint32_t>(gid % compress_words);
    Element* base = storage + static_cast<size_t>(batch) * stride_words + compress_offset_words;
    base[local] = FromOracleMidstateOrFallback(
        seed_cv,
        batch,
        OracleInputSeedMidstate(seed_midstates, batch_size, 4U, batch),
        local);
}

} // namespace

MatMulGeneratedInputsDevice::~MatMulGeneratedInputsDevice()
{
    if (device_index >= 0) {
        cudaSetDevice(device_index);
    }
    if (owns_ready_event && ready_event != nullptr) {
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ready_event));
    }
    if (owns_storage) {
        cudaFree(storage);
    }
}

MatMulInputGenerationProfile ProbeMatMulInputGenerationProfile()
{
    MatMulInputGenerationProfile profile;

    const auto runtime = ProbeCudaRuntime();
    if (!runtime.available) {
        profile.available = false;
        profile.pool_initialized = false;
        profile.library_source = "unavailable";
        profile.reason = runtime.reason;
        return profile;
    }

    profile.available = true;
    profile.pool_initialized = g_profile.pool_initialized.load(std::memory_order_relaxed);
    profile.samples = g_profile.samples.load(std::memory_order_relaxed);
    profile.allocation_events = g_profile.allocation_events.load(std::memory_order_relaxed);
    profile.reuse_events = g_profile.reuse_events.load(std::memory_order_relaxed);
    profile.library_source = "cuda_compiled";
    {
        std::lock_guard<std::mutex> lock(g_profile.mutex);
        profile.last_encode_noise_us = g_profile.last_encode_noise_us;
        profile.last_encode_compress_us = g_profile.last_encode_compress_us;
        profile.last_submit_wait_us = g_profile.last_submit_wait_us;
        profile.last_gpu_generation_ms = g_profile.last_gpu_generation_ms;
        profile.reason = g_profile.reason;
    }
    if (profile.reason.empty()) {
        profile.reason = profile.pool_initialized ? "cuda_oracle_ready" : "cuda_oracle_pool_uninitialized";
    }
    return profile;
}

MatMulInputGenerationResult GenerateMatMulInputsGPU(const MatMulInputGenerationRequest& request)
{
    MatMulInputGenerationResult result;
    const auto runtime = ProbeCudaRuntime();
    result.available = runtime.available;
    if (!runtime.available) {
        result.error = runtime.reason;
        return result;
    }

    uint32_t noise_words{0};
    uint32_t compress_words{0};
    if (!ValidateInputGenerationRequest(request, result.error, noise_words, compress_words)) {
        return result;
    }
    const auto seed_el = ToInternalSeedBytes(matmul::noise::DeriveNoiseSeed(matmul::noise::TAG_EL, request.sigma));
    const auto seed_er = ToInternalSeedBytes(matmul::noise::DeriveNoiseSeed(matmul::noise::TAG_ER, request.sigma));
    const auto seed_fl = ToInternalSeedBytes(matmul::noise::DeriveNoiseSeed(matmul::noise::TAG_FL, request.sigma));
    const auto seed_fr = ToInternalSeedBytes(matmul::noise::DeriveNoiseSeed(matmul::noise::TAG_FR, request.sigma));
    const auto seed_cv = ToInternalSeedBytes(DeriveCompressionSeed(request.sigma));

    auto& workspace = g_workspace;
    ResetWorkspaceForDevice(workspace, runtime.device_index);

    cudaError_t error = cudaSetDevice(runtime.device_index);
    if (error != cudaSuccess) {
        result.error = "cudaSetDevice failed:" + std::string(cudaGetErrorString(error));
        return result;
    }
    if (!EnsureWorkspaceStream(workspace, result.error)) {
        return result;
    }

    if (!EnsureOutputBuffers(workspace, noise_words, compress_words, result.error)) {
        return result;
    }

    const uint32_t noise_blocks = (noise_words + ORACLE_THREADS - 1) / ORACLE_THREADS;
    const uint32_t compress_blocks = (compress_words + ORACLE_THREADS - 1) / ORACLE_THREADS;
    const auto encode_noise_start = std::chrono::steady_clock::now();
    GenerateOracleNoiseKernel<<<noise_blocks, ORACLE_THREADS, 0, workspace.stream>>>(
        seed_el,
        seed_er,
        seed_fl,
        seed_fr,
        workspace.out_e_l,
        workspace.out_e_r,
        workspace.out_f_l,
        workspace.out_f_r,
        noise_words);
    double encode_noise_us = std::chrono::duration<double, std::micro>(
                                 std::chrono::steady_clock::now() - encode_noise_start)
                                 .count();

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        result.error = "CUDA oracle noise kernel failed:" + std::string(cudaGetErrorString(error));
        return result;
    }

    const auto encode_compress_start = std::chrono::steady_clock::now();
    GenerateOracleVectorKernel<<<compress_blocks, ORACLE_THREADS, 0, workspace.stream>>>(
        seed_cv,
        workspace.out_cv,
        compress_words);
    double encode_compress_us = std::chrono::duration<double, std::micro>(
                                    std::chrono::steady_clock::now() - encode_compress_start)
                                    .count();

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        result.error = "CUDA oracle compress kernel failed:" + std::string(cudaGetErrorString(error));
        return result;
    }

    const auto submit_wait_start = std::chrono::steady_clock::now();
    std::string staging_warning;
    if (!workspace.host_e_l.Ensure(noise_words, staging_warning) ||
        !workspace.host_e_r.Ensure(noise_words, staging_warning) ||
        !workspace.host_f_l.Ensure(noise_words, staging_warning) ||
        !workspace.host_f_r.Ensure(noise_words, staging_warning) ||
        !workspace.host_cv.Ensure(compress_words, staging_warning)) {
        result.error = staging_warning;
        return result;
    }
    error = cudaMemcpyAsync(workspace.host_e_l.data(), workspace.out_e_l, noise_words * sizeof(Element), cudaMemcpyDeviceToHost, workspace.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(workspace.host_e_r.data(), workspace.out_e_r, noise_words * sizeof(Element), cudaMemcpyDeviceToHost, workspace.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(workspace.host_f_l.data(), workspace.out_f_l, noise_words * sizeof(Element), cudaMemcpyDeviceToHost, workspace.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(workspace.host_f_r.data(), workspace.out_f_r, noise_words * sizeof(Element), cudaMemcpyDeviceToHost, workspace.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(workspace.host_cv.data(), workspace.out_cv, compress_words * sizeof(Element), cudaMemcpyDeviceToHost, workspace.stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(workspace.stream);
    const double submit_wait_us = std::chrono::duration<double, std::micro>(
                                      std::chrono::steady_clock::now() - submit_wait_start)
                                      .count();
    if (error != cudaSuccess) {
        result.error = "CUDA oracle stream completion failed:" + std::string(cudaGetErrorString(error));
        return result;
    }

    result.noise_e_l.assign(workspace.host_e_l.data(), workspace.host_e_l.data() + noise_words);
    result.noise_e_r.assign(workspace.host_e_r.data(), workspace.host_e_r.data() + noise_words);
    result.noise_f_l.assign(workspace.host_f_l.data(), workspace.host_f_l.data() + noise_words);
    result.noise_f_r.assign(workspace.host_f_r.data(), workspace.host_f_r.data() + noise_words);
    result.compress_vec.assign(workspace.host_cv.data(), workspace.host_cv.data() + compress_words);
    result.success = true;
    UpdateProfile(encode_noise_us, encode_compress_us, submit_wait_us, "cuda_noise4_plus_compress");
    return result;
}

MatMulInputGenerationDeviceResult GenerateMatMulInputsGPUDevice(const MatMulInputGenerationRequest& request)
{
    MatMulInputGenerationDeviceResult result;
    const auto runtime_probe = ResolveCudaRuntimeForNextSelectedDevice(result.error);
    result.available = runtime_probe.has_value();
    if (!runtime_probe.has_value()) {
        return result;
    }
    const auto runtime = *runtime_probe;

    uint32_t noise_words{0};
    uint32_t compress_words{0};
    if (!ValidateInputGenerationRequest(request, result.error, noise_words, compress_words)) {
        return result;
    }

    const auto seed_el = ToInternalSeedBytes(matmul::noise::DeriveNoiseSeed(matmul::noise::TAG_EL, request.sigma));
    const auto seed_er = ToInternalSeedBytes(matmul::noise::DeriveNoiseSeed(matmul::noise::TAG_ER, request.sigma));
    const auto seed_fl = ToInternalSeedBytes(matmul::noise::DeriveNoiseSeed(matmul::noise::TAG_FL, request.sigma));
    const auto seed_fr = ToInternalSeedBytes(matmul::noise::DeriveNoiseSeed(matmul::noise::TAG_FR, request.sigma));
    const auto seed_cv = ToInternalSeedBytes(DeriveCompressionSeed(request.sigma));

    auto& workspace = g_workspace;
    ResetWorkspaceForDevice(workspace, runtime.device_index);

    cudaError_t error = cudaSetDevice(runtime.device_index);
    if (error != cudaSuccess) {
        result.error = "cudaSetDevice failed:" + std::string(cudaGetErrorString(error));
        return result;
    }
    if (!EnsureWorkspaceStream(workspace, result.error)) {
        return result;
    }

    auto generated = AcquireGeneratedInputsDevice(
        runtime.device_index,
        request.n,
        request.b,
        request.r,
        noise_words,
        compress_words,
        result.error);
    if (!generated) {
        return result;
    }

    const uint32_t noise_blocks = (noise_words + ORACLE_THREADS - 1) / ORACLE_THREADS;
    const uint32_t compress_blocks = (compress_words + ORACLE_THREADS - 1) / ORACLE_THREADS;
    const auto encode_noise_start = std::chrono::steady_clock::now();
    GenerateOracleNoiseKernel<<<noise_blocks, ORACLE_THREADS, 0, workspace.stream>>>(
        seed_el,
        seed_er,
        seed_fl,
        seed_fr,
        generated->noise_e_l,
        generated->noise_e_r,
        generated->noise_f_l,
        generated->noise_f_r,
        noise_words);
    const double encode_noise_us = std::chrono::duration<double, std::micro>(
                                       std::chrono::steady_clock::now() - encode_noise_start)
                                       .count();

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        result.error = "CUDA oracle noise kernel failed:" + std::string(cudaGetErrorString(error));
        return result;
    }

    const auto encode_compress_start = std::chrono::steady_clock::now();
    GenerateOracleVectorKernel<<<compress_blocks, ORACLE_THREADS, 0, workspace.stream>>>(
        seed_cv,
        generated->compress_vec,
        compress_words);
    const double encode_compress_us = std::chrono::duration<double, std::micro>(
                                          std::chrono::steady_clock::now() - encode_compress_start)
                                          .count();

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        result.error = "CUDA oracle compress kernel failed:" + std::string(cudaGetErrorString(error));
        return result;
    }

    auto* generated_inputs = const_cast<MatMulGeneratedInputsDevice*>(generated.get());
    if (!EnsureGeneratedInputsReadyEvent(*generated_inputs, result.error)) {
        return result;
    }

    const auto submit_wait_start = std::chrono::steady_clock::now();
    error = cudaEventRecord(
        reinterpret_cast<cudaEvent_t>(generated_inputs->ready_event),
        workspace.stream);
    const double submit_wait_us = std::chrono::duration<double, std::micro>(
                                      std::chrono::steady_clock::now() - submit_wait_start)
                                      .count();
    if (error != cudaSuccess) {
        result.error = "CUDA oracle ready-event record failed:" + std::string(cudaGetErrorString(error));
        return result;
    }

    result.success = true;
    result.inputs = std::move(generated);
    UpdateProfile(encode_noise_us, encode_compress_us, submit_wait_us, "cuda_noise4_plus_compress_device");
    return result;
}

static MatMulInputGenerationDeviceBatchResult GenerateMatMulInputsGPUDeviceBatchOnDevice(
    const MatMulInputGenerationDeviceBatchRequest& request,
    int device_index)
{
    MatMulInputGenerationDeviceBatchResult result;
    const auto runtime_probe = ResolveCudaRuntimeForSelectedDevice(device_index, result.error);
    result.available = runtime_probe.has_value();
    if (!runtime_probe.has_value()) {
        return result;
    }
    const auto runtime = *runtime_probe;

    uint32_t noise_words{0};
    uint32_t compress_words{0};
    if (!ValidateInputGenerationDeviceBatchRequest(request, result.error, noise_words, compress_words)) {
        return result;
    }

    std::vector<OracleSeedBytes> seed_el(request.batch_size);
    std::vector<OracleSeedBytes> seed_er(request.batch_size);
    std::vector<OracleSeedBytes> seed_fl(request.batch_size);
    std::vector<OracleSeedBytes> seed_fr(request.batch_size);
    std::vector<OracleSeedBytes> seed_cv(request.batch_size);
    for (uint32_t i = 0; i < request.batch_size; ++i) {
        const uint256& sigma = request.sigmas[i];
        seed_el[i] = ToInternalSeedBytes(matmul::noise::DeriveNoiseSeed(matmul::noise::TAG_EL, sigma));
        seed_er[i] = ToInternalSeedBytes(matmul::noise::DeriveNoiseSeed(matmul::noise::TAG_ER, sigma));
        seed_fl[i] = ToInternalSeedBytes(matmul::noise::DeriveNoiseSeed(matmul::noise::TAG_FL, sigma));
        seed_fr[i] = ToInternalSeedBytes(matmul::noise::DeriveNoiseSeed(matmul::noise::TAG_FR, sigma));
        seed_cv[i] = ToInternalSeedBytes(DeriveCompressionSeed(sigma));
    }

    auto& workspace = g_workspace;
    ResetWorkspaceForDevice(workspace, runtime.device_index);

    cudaError_t error = cudaSetDevice(runtime.device_index);
    if (error != cudaSuccess) {
        result.error = "cudaSetDevice failed:" + std::string(cudaGetErrorString(error));
        return result;
    }
    if (!EnsureWorkspaceStream(workspace, result.error)) {
        return result;
    }
    if (!EnsureSeedBuffers(workspace, request.batch_size, result.error)) {
        return result;
    }
    const size_t seed_midstate_words =
        static_cast<size_t>(request.batch_size) *
        ORACLE_INPUT_SEED_KINDS *
        ORACLE_SEED_MIDSTATE_WORDS;
    if (!EnsureDeviceBuffer(
            workspace.batch_seed_midstates,
            workspace.batch_seed_midstates_capacity,
            seed_midstate_words,
            result.error)) {
        return result;
    }

    const uint32_t stride_words = noise_words * 4U + compress_words;
    auto owner = AcquireGeneratedInputsDeviceBatchSlot(
        runtime.device_index,
        request.batch_size,
        static_cast<size_t>(request.batch_size) * stride_words,
        result.error);
    if (owner == nullptr) {
        return result;
    }

    const auto seed_copy_start = std::chrono::steady_clock::now();
    error = cudaMemcpyAsync(workspace.batch_seed_el, seed_el.data(), request.batch_size * sizeof(OracleSeedBytes), cudaMemcpyHostToDevice, workspace.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(workspace.batch_seed_er, seed_er.data(), request.batch_size * sizeof(OracleSeedBytes), cudaMemcpyHostToDevice, workspace.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(workspace.batch_seed_fl, seed_fl.data(), request.batch_size * sizeof(OracleSeedBytes), cudaMemcpyHostToDevice, workspace.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(workspace.batch_seed_fr, seed_fr.data(), request.batch_size * sizeof(OracleSeedBytes), cudaMemcpyHostToDevice, workspace.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(workspace.batch_seed_cv, seed_cv.data(), request.batch_size * sizeof(OracleSeedBytes), cudaMemcpyHostToDevice, workspace.stream);
    const double seed_copy_us = std::chrono::duration<double, std::micro>(
                                    std::chrono::steady_clock::now() - seed_copy_start)
                                    .count();
    if (error != cudaSuccess) {
        result.error = "cudaMemcpy batch oracle seeds failed:" + std::string(cudaGetErrorString(error));
        return result;
    }

    const uint32_t seed_count = request.batch_size * ORACLE_INPUT_SEED_KINDS;
    const uint32_t seed_blocks = (seed_count + ORACLE_THREADS - 1U) / ORACLE_THREADS;
    const auto precompute_start = std::chrono::steady_clock::now();
    PrecomputeOracleSeedMidstatesBatchKernel<<<seed_blocks, ORACLE_THREADS, 0, workspace.stream>>>(
        workspace.batch_seed_el,
        workspace.batch_seed_er,
        workspace.batch_seed_fl,
        workspace.batch_seed_fr,
        workspace.batch_seed_cv,
        request.batch_size,
        workspace.batch_seed_midstates);
    const double precompute_us = std::chrono::duration<double, std::micro>(
                                     std::chrono::steady_clock::now() - precompute_start)
                                     .count();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        result.error = "CUDA oracle seed midstate precompute kernel failed:" + std::string(cudaGetErrorString(error));
        return result;
    }

    const size_t total_noise_words = static_cast<size_t>(request.batch_size) * noise_words;
    const size_t total_compress_words = static_cast<size_t>(request.batch_size) * compress_words;
    const uint32_t noise_blocks = static_cast<uint32_t>((total_noise_words + ORACLE_THREADS - 1U) / ORACLE_THREADS);
    const uint32_t compress_blocks = static_cast<uint32_t>((total_compress_words + ORACLE_THREADS - 1U) / ORACLE_THREADS);
    const auto encode_noise_start = std::chrono::steady_clock::now();
    GenerateOracleNoiseBatchMidstateKernel<<<noise_blocks, ORACLE_THREADS, 0, workspace.stream>>>(
        workspace.batch_seed_el,
        workspace.batch_seed_er,
        workspace.batch_seed_fl,
        workspace.batch_seed_fr,
        workspace.batch_seed_midstates,
        owner->storage,
        request.batch_size,
        stride_words,
        noise_words,
        total_noise_words);
    const double encode_noise_us = std::chrono::duration<double, std::micro>(
                                       std::chrono::steady_clock::now() - encode_noise_start)
                                       .count();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        result.error = "CUDA oracle batch noise kernel failed:" + std::string(cudaGetErrorString(error));
        return result;
    }

    const auto encode_compress_start = std::chrono::steady_clock::now();
    GenerateOracleVectorBatchMidstateKernel<<<compress_blocks, ORACLE_THREADS, 0, workspace.stream>>>(
        workspace.batch_seed_cv,
        workspace.batch_seed_midstates,
        owner->storage,
        request.batch_size,
        stride_words,
        noise_words * 4U,
        compress_words,
        total_compress_words);
    const double encode_compress_us = std::chrono::duration<double, std::micro>(
                                          std::chrono::steady_clock::now() - encode_compress_start)
                                          .count();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        result.error = "CUDA oracle batch compress kernel failed:" + std::string(cudaGetErrorString(error));
        return result;
    }

    const auto submit_wait_start = std::chrono::steady_clock::now();
    error = cudaEventRecord(owner->ready_event, workspace.stream);
    const double submit_wait_us = std::chrono::duration<double, std::micro>(
                                      std::chrono::steady_clock::now() - submit_wait_start)
                                      .count();
    if (error != cudaSuccess) {
        result.error = "CUDA oracle batch ready-event record failed:" + std::string(cudaGetErrorString(error));
        return result;
    }

    result.inputs.reserve(request.batch_size);
    for (uint32_t i = 0; i < request.batch_size; ++i) {
        auto& view = owner->views[i];
        view.device_index = runtime.device_index;
        view.n = request.n;
        view.b = request.b;
        view.r = request.r;
        view.noise_words = noise_words;
        view.compress_words = compress_words;
        view.storage = owner->storage + static_cast<size_t>(i) * stride_words;
        view.ready_event = owner->ready_event;
        view.noise_e_l = view.storage;
        view.noise_e_r = view.noise_e_l + noise_words;
        view.noise_f_l = view.noise_e_r + noise_words;
        view.noise_f_r = view.noise_f_l + noise_words;
        view.compress_vec = view.noise_f_r + noise_words;
        view.owns_storage = false;
        view.owns_ready_event = false;
        result.inputs.emplace_back(owner, &view);
    }

    result.success = true;
    UpdateProfile(
        seed_copy_us + precompute_us + encode_noise_us,
        encode_compress_us,
        submit_wait_us,
        "cuda_noise4_plus_compress_device_batch_midstate");
    g_profile.samples.fetch_add(request.batch_size - 1U, std::memory_order_relaxed);
    return result;
}

MatMulInputGenerationDeviceBatchResult GenerateMatMulInputsGPUDeviceBatch(
    const MatMulInputGenerationDeviceBatchRequest& request)
{
    MatMulInputGenerationDeviceBatchResult result;
    const auto topology = ProbeCudaTopology();
    result.available = topology.available;
    if (!topology.available) {
        result.error = topology.reason;
        return result;
    }

    [[maybe_unused]] uint32_t noise_words{0};
    [[maybe_unused]] uint32_t compress_words{0};
    if (!ValidateInputGenerationDeviceBatchRequest(request, result.error, noise_words, compress_words)) {
        return result;
    }

    const auto shards = PlanCudaBatchShards(topology.selected_devices, request.batch_size);
    if (shards.empty()) {
        result.error = "no_cuda_input_generation_shards_available";
        return result;
    }
    if (shards.size() == 1) {
        return GenerateMatMulInputsGPUDeviceBatchOnDevice(request, shards.front().device_index);
    }

    struct ShardResult {
        CudaBatchShard shard;
        MatMulInputGenerationDeviceBatchResult result;
    };

    std::vector<std::future<ShardResult>> futures;
    futures.reserve(shards.size());
    for (const auto& shard : shards) {
        futures.push_back(std::async(std::launch::async, [request, shard]() {
            const MatMulInputGenerationDeviceBatchRequest shard_request{
                .n = request.n,
                .b = request.b,
                .r = request.r,
                .batch_size = static_cast<uint32_t>(shard.count),
                .sigmas = request.sigmas + shard.start_index,
            };
            return ShardResult{
                .shard = shard,
                .result = GenerateMatMulInputsGPUDeviceBatchOnDevice(
                    shard_request,
                    shard.device_index),
            };
        }));
    }

    result.inputs.resize(request.batch_size);
    try {
        for (auto& future : futures) {
            auto shard_result = future.get();
            if (!shard_result.result.success) {
                result.available = shard_result.result.available;
                result.inputs.clear();
                result.error = "cuda_device_" + std::to_string(shard_result.shard.device_index) +
                    "_input_generation_failed:" +
                    (shard_result.result.error.empty() ? "unknown_error" : shard_result.result.error);
                return result;
            }
            if (shard_result.result.inputs.size() != shard_result.shard.count) {
                result.inputs.clear();
                result.error = "cuda_multi_device_input_generation_result_size_mismatch";
                return result;
            }

            for (size_t i = 0; i < shard_result.shard.count; ++i) {
                result.inputs[shard_result.shard.start_index + i] =
                    std::move(shard_result.result.inputs[i]);
            }
        }
    } catch (const std::exception& e) {
        result.inputs.clear();
        result.error = std::string{"cuda_multi_device_input_generation_exception:"} + e.what();
        return result;
    } catch (...) {
        result.inputs.clear();
        result.error = "cuda_multi_device_input_generation_unknown_exception";
        return result;
    }

    result.success = true;
    return result;
}

static MatMulNonceSeedPreHashScanResult ScanMatMulNonceSeedPreHashGPUOnDevice(
    const MatMulNonceSeedPreHashScanRequest& request,
    int device_index)
{
    MatMulNonceSeedPreHashScanResult result;
    const auto runtime_probe = ResolveCudaRuntimeForSelectedDevice(device_index, result.error);
    result.available = runtime_probe.has_value();
    if (!runtime_probe.has_value()) {
        return result;
    }
    const auto runtime = *runtime_probe;
    if (request.scan_count == 0) {
        result.success = true;
        result.scanned_count = 0;
        return result;
    }
    if (request.matmul_dim == 0) {
        result.error = "CUDA nonce-seed pre-hash scan requires non-zero matmul_dim";
        return result;
    }
    if (request.seed_version != 2U && request.seed_version != 3U) {
        result.error = "CUDA nonce-seed pre-hash scan requires seed_version 2 or 3";
        return result;
    }
    if (request.seed_version == 3U && request.parent_median_time_past < 0) {
        result.error = "CUDA seed-v3 nonce-seed pre-hash scan requires non-negative parent median time past";
        return result;
    }

    auto& workspace = g_workspace;
    ResetWorkspaceForDevice(workspace, runtime.device_index);

    cudaError_t error = cudaSetDevice(runtime.device_index);
    if (error != cudaSuccess) {
        result.error = "cudaSetDevice failed:" + std::string(cudaGetErrorString(error));
        return result;
    }
    if (!EnsureWorkspaceStream(workspace, result.error)) {
        return result;
    }

    const bool use_compact_offsets = request.max_selected_offsets != 0;
    if (use_compact_offsets) {
        if (!EnsureDeviceBuffer(
                workspace.out_scan_selected_offsets,
                workspace.scan_selected_offsets_capacity,
                request.max_selected_offsets,
                result.error) ||
            !EnsureDeviceBuffer(
                workspace.out_scan_selected_count,
                workspace.scan_selected_count_capacity,
                1U,
                result.error)) {
            return result;
        }
        error = cudaMemsetAsync(workspace.out_scan_selected_count, 0, sizeof(Element), workspace.stream);
        if (error != cudaSuccess) {
            result.error = "CUDA nonce-seed compact scan counter reset failed:" + std::string(cudaGetErrorString(error));
            return result;
        }
    } else if (!EnsureDeviceBuffer(
                   workspace.out_scan_flags,
                   workspace.scan_flags_capacity,
                   request.scan_count,
                   result.error)) {
        return result;
    }

    const uint32_t scan_blocks = (request.scan_count + ORACLE_THREADS - 1) / ORACLE_THREADS;
    if (request.seed_version == 3U) {
        ScanNonceSeedPreHashV3Kernel<<<scan_blocks, ORACLE_THREADS, 0, workspace.stream>>>(
            ToInternalSeedBytes(request.previous_block_hash),
            ToInternalSeedBytes(request.merkle_root),
            ToInternalSeedBytes(request.pre_hash_target),
            static_cast<uint32_t>(request.version),
            request.block_height,
            request.time,
            request.bits,
            request.start_nonce,
            request.matmul_dim,
            static_cast<uint64_t>(request.parent_median_time_past),
            use_compact_offsets ? nullptr : workspace.out_scan_flags,
            use_compact_offsets ? workspace.out_scan_selected_offsets : nullptr,
            use_compact_offsets ? workspace.out_scan_selected_count : nullptr,
            request.max_selected_offsets,
            request.scan_count);
    } else {
        ScanNonceSeedPreHashKernel<<<scan_blocks, ORACLE_THREADS, 0, workspace.stream>>>(
            ToInternalSeedBytes(request.previous_block_hash),
            ToInternalSeedBytes(request.merkle_root),
            ToInternalSeedBytes(request.pre_hash_target),
            static_cast<uint32_t>(request.version),
            request.block_height,
            request.time,
            request.bits,
            request.start_nonce,
            request.matmul_dim,
            request.seed_version,
            static_cast<uint64_t>(request.parent_median_time_past),
            use_compact_offsets ? nullptr : workspace.out_scan_flags,
            use_compact_offsets ? workspace.out_scan_selected_offsets : nullptr,
            use_compact_offsets ? workspace.out_scan_selected_count : nullptr,
            request.max_selected_offsets,
            request.scan_count);
    }
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        result.error = "CUDA nonce-seed pre-hash scan kernel failed:" + std::string(cudaGetErrorString(error));
        return result;
    }

    std::string staging_warning;
    if (use_compact_offsets) {
        if (!workspace.host_scan_selected_count.Ensure(1U, staging_warning) ||
            !workspace.host_scan_selected_offsets.Ensure(request.max_selected_offsets, staging_warning)) {
            result.error = staging_warning;
            return result;
        }
        error = cudaMemcpyAsync(
            workspace.host_scan_selected_count.data(),
            workspace.out_scan_selected_count,
            sizeof(Element),
            cudaMemcpyDeviceToHost,
            workspace.stream);
        if (error == cudaSuccess) {
            error = cudaMemcpyAsync(
                workspace.host_scan_selected_offsets.data(),
                workspace.out_scan_selected_offsets,
                request.max_selected_offsets * sizeof(Element),
                cudaMemcpyDeviceToHost,
                workspace.stream);
        }
        if (error == cudaSuccess) {
            error = cudaStreamSynchronize(workspace.stream);
        }
        if (error != cudaSuccess) {
            result.error = "CUDA nonce-seed compact pre-hash scan completion failed:" + std::string(cudaGetErrorString(error));
            return result;
        }
        result.pass_count = workspace.host_scan_selected_count.data()[0];
        if (result.pass_count > request.max_selected_offsets) {
            MatMulNonceSeedPreHashScanRequest full_flags_request{request};
            full_flags_request.max_selected_offsets = 0;
            return ScanMatMulNonceSeedPreHashGPUOnDevice(full_flags_request, device_index);
        }
        const uint32_t stored_count = std::min<uint32_t>(result.pass_count, request.max_selected_offsets);
        result.selected_offsets.reserve(stored_count);
        const Element* offsets = workspace.host_scan_selected_offsets.data();
        for (uint32_t i = 0; i < stored_count; ++i) {
            result.selected_offsets.push_back(offsets[i]);
        }
        std::sort(result.selected_offsets.begin(), result.selected_offsets.end());
    } else {
        if (!workspace.host_scan_flags.Ensure(request.scan_count, staging_warning)) {
            result.error = staging_warning;
            return result;
        }
        error = cudaMemcpyAsync(
            workspace.host_scan_flags.data(),
            workspace.out_scan_flags,
            request.scan_count * sizeof(Element),
            cudaMemcpyDeviceToHost,
            workspace.stream);
        if (error == cudaSuccess) {
            error = cudaStreamSynchronize(workspace.stream);
        }
        if (error != cudaSuccess) {
            result.error = "CUDA nonce-seed pre-hash scan completion failed:" + std::string(cudaGetErrorString(error));
            return result;
        }

        result.pass_flags.resize(request.scan_count);
        const Element* flags = workspace.host_scan_flags.data();
        for (uint32_t i = 0; i < request.scan_count; ++i) {
            result.pass_flags[i] = flags[i] != 0 ? 1U : 0U;
            result.pass_count += result.pass_flags[i] != 0 ? 1U : 0U;
        }
    }
    result.scanned_count = request.scan_count;
    result.success = true;
    return result;
}

MatMulNonceSeedPreHashScanResult ScanMatMulNonceSeedPreHashGPU(
    const MatMulNonceSeedPreHashScanRequest& request)
{
    MatMulNonceSeedPreHashScanResult result;
    const auto topology = ProbeCudaTopology();
    result.available = topology.available;
    if (!topology.available) {
        result.error = topology.reason;
        return result;
    }
    if (request.scan_count == 0) {
        result.success = true;
        result.scanned_count = 0;
        return result;
    }
    if (request.matmul_dim == 0) {
        result.error = "CUDA nonce-seed pre-hash scan requires non-zero matmul_dim";
        return result;
    }
    if (request.seed_version != 2U && request.seed_version != 3U) {
        result.error = "CUDA nonce-seed pre-hash scan requires seed_version 2 or 3";
        return result;
    }
    if (request.seed_version == 3U && request.parent_median_time_past < 0) {
        result.error = "CUDA seed-v3 nonce-seed pre-hash scan requires non-negative parent median time past";
        return result;
    }

    const auto shards = PlanCudaBatchShards(topology.selected_devices, request.scan_count);
    if (shards.empty()) {
        result.error = "no_cuda_prehash_scan_shards_available";
        return result;
    }
    if (shards.size() == 1) {
        return ScanMatMulNonceSeedPreHashGPUOnDevice(request, shards.front().device_index);
    }

    struct ShardResult {
        CudaBatchShard shard;
        MatMulNonceSeedPreHashScanResult result;
    };

    std::vector<std::future<ShardResult>> futures;
    futures.reserve(shards.size());
    for (const auto& shard : shards) {
        futures.push_back(std::async(std::launch::async, [request, shard]() {
            MatMulNonceSeedPreHashScanRequest shard_request{request};
            shard_request.start_nonce = request.start_nonce + static_cast<uint64_t>(shard.start_index);
            shard_request.scan_count = static_cast<uint32_t>(shard.count);
            return ShardResult{
                .shard = shard,
                .result = ScanMatMulNonceSeedPreHashGPUOnDevice(
                    shard_request,
                    shard.device_index),
            };
        }));
    }

    std::vector<ShardResult> shard_results;
    shard_results.reserve(shards.size());
    try {
        for (auto& future : futures) {
            shard_results.push_back(future.get());
        }
    } catch (const std::exception& e) {
        result.error = std::string{"cuda_multi_device_prehash_scan_exception:"} + e.what();
        return result;
    } catch (...) {
        result.error = "cuda_multi_device_prehash_scan_unknown_exception";
        return result;
    }

    std::stable_sort(
        shard_results.begin(),
        shard_results.end(),
        [](const ShardResult& a, const ShardResult& b) {
            return a.shard.start_index < b.shard.start_index;
        });

    const bool compact_offsets = request.max_selected_offsets != 0;
    if (!compact_offsets) {
        result.pass_flags.resize(request.scan_count);
    } else {
        result.selected_offsets.reserve(request.max_selected_offsets);
    }

    result.scanned_count = request.scan_count;
    for (auto& shard_result : shard_results) {
        if (!shard_result.result.success) {
            result.available = shard_result.result.available;
            result.pass_flags.clear();
            result.selected_offsets.clear();
            result.error = "cuda_device_" + std::to_string(shard_result.shard.device_index) +
                "_prehash_scan_failed:" +
                (shard_result.result.error.empty() ? "unknown_error" : shard_result.result.error);
            return result;
        }
        if (shard_result.result.scanned_count != shard_result.shard.count) {
            result.pass_flags.clear();
            result.selected_offsets.clear();
            result.error = "cuda_multi_device_prehash_scan_result_size_mismatch";
            return result;
        }

        result.pass_count += shard_result.result.pass_count;
        if (compact_offsets) {
            for (const uint32_t offset : shard_result.result.selected_offsets) {
                if (offset >= shard_result.shard.count) {
                    continue;
                }
                if (result.selected_offsets.size() >= request.max_selected_offsets) {
                    continue;
                }
                result.selected_offsets.push_back(
                    static_cast<uint32_t>(shard_result.shard.start_index + offset));
            }
        } else {
            if (shard_result.result.pass_flags.size() != shard_result.shard.count) {
                result.pass_flags.clear();
                result.error = "cuda_multi_device_prehash_scan_flag_size_mismatch";
                return result;
            }
            std::copy(
                shard_result.result.pass_flags.begin(),
                shard_result.result.pass_flags.end(),
                result.pass_flags.begin() + shard_result.shard.start_index);
        }
    }

    result.success = true;
    return result;
}

} // namespace btx::cuda

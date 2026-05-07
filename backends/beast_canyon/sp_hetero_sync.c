// Shannon-Prime Beast Canyon: Heterogeneous GPU Synchronization — Implementation
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "sp_hetero_sync.h"
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <time.h>
#  include <sched.h>
#endif

// ============================================================================
// Platform timing + yield
// ============================================================================

static uint64_t sp_time_us(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (uint64_t)(now.QuadPart * 1000000ULL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
#endif
}

static inline void sp_yield(void) {
#ifdef _WIN32
    SwitchToThread();
#else
    sched_yield();
#endif
}

// ============================================================================
// Lifecycle
// ============================================================================

int sp_hetero_barrier_init(sp_hetero_barrier_t *barrier) {
    memset(barrier, 0, sizeof(*barrier));
    barrier->spin_count = 1000;
    barrier->use_blocking = false;
    barrier->min_barrier_us = UINT64_MAX;
    return 0;
}

void sp_hetero_barrier_free(sp_hetero_barrier_t *barrier) {
    // Backend-specific cleanup would go here (destroy events/fences).
    // For now, the orchestrator owns the GPU handles.
    memset(barrier, 0, sizeof(*barrier));
}

// ============================================================================
// GPU Discovery
// ============================================================================

int sp_hetero_detect_gpus(sp_hetero_barrier_t *barrier) {
    barrier->n_gpus = 0;

    // --- CUDA detection ---
#ifdef SP_WITH_CUDA
    {
        int cuda_count = 0;
        // cudaGetDeviceCount(&cuda_count);
        // For each CUDA device, populate barrier->gpu[n_gpus++]
        // This is wired at link time when CUDA is available.
        fprintf(stderr, "[sp-sync] CUDA detection: compile with -DSP_WITH_CUDA\n");
    }
#endif

    // --- Vulkan detection ---
#ifdef SP_WITH_VULKAN
    {
        // vkEnumeratePhysicalDevices(...)
        // For Intel Xe iGPU: look for vendor ID 0x8086
        fprintf(stderr, "[sp-sync] Vulkan detection: compile with -DSP_WITH_VULKAN\n");
    }
#endif

    // If no GPUs detected, we run CPU-only
    if (barrier->n_gpus == 0) {
        fprintf(stderr, "[sp-sync] No GPUs detected — CPU-only mode\n");
    }

    return barrier->n_gpus;
}

int sp_hetero_add_gpu(sp_hetero_barrier_t *barrier,
                      sp_gpu_type_t type, int device_id)
{
    if (barrier->n_gpus >= 2) {
        fprintf(stderr, "[sp-sync] ERROR: max 2 GPUs supported\n");
        return -1;
    }

    int idx = barrier->n_gpus;
    sp_gpu_device_t *gpu = &barrier->gpu[idx];
    memset(gpu, 0, sizeof(*gpu));
    gpu->type = type;
    gpu->device_id = device_id;

    const char *type_str = "unknown";
    switch (type) {
    case SP_GPU_CUDA:       type_str = "CUDA"; break;
    case SP_GPU_VULKAN:     type_str = "Vulkan"; break;
    case SP_GPU_LEVEL_ZERO: type_str = "Level Zero"; break;
    case SP_GPU_CPU_ONLY:   type_str = "CPU"; break;
    default: break;
    }
    snprintf(gpu->name, sizeof(gpu->name), "%s device %d", type_str, device_id);

    barrier->n_gpus++;
    fprintf(stderr, "[sp-sync] Added GPU[%d]: %s\n", idx, gpu->name);
    return idx;
}

// ============================================================================
// Dispatch tracking
// ============================================================================

void sp_hetero_mark_dispatched(sp_hetero_barrier_t *barrier, int gpu_idx) {
    if (gpu_idx < 0 || gpu_idx >= barrier->n_gpus) return;
    barrier->gpu_done[gpu_idx] = false;
    barrier->gpu[gpu_idx].dispatch_count++;
}

void sp_hetero_mark_done(sp_hetero_barrier_t *barrier, int gpu_idx) {
    if (gpu_idx < 0 || gpu_idx >= barrier->n_gpus) return;
    barrier->gpu_done[gpu_idx] = true;
}

// ============================================================================
// The Barrier — the critical synchronization point
// ============================================================================

// Check if a specific GPU is done (backend-specific polling).
static bool sp_poll_gpu(sp_hetero_barrier_t *barrier, int idx) {
    if (barrier->gpu_done[idx]) return true;

    sp_gpu_device_t *gpu = &barrier->gpu[idx];

    switch (gpu->type) {
#ifdef SP_WITH_CUDA
    case SP_GPU_CUDA:
        if (sp_hetero_cuda_sync_event(gpu) == 0) {
            barrier->gpu_done[idx] = true;
        }
        break;
#endif
#ifdef SP_WITH_VULKAN
    case SP_GPU_VULKAN:
        if (sp_hetero_vulkan_wait_fence(gpu) == 0) {
            barrier->gpu_done[idx] = true;
        }
        break;
#endif
    case SP_GPU_CPU_ONLY:
        // CPU "GPU" is always done immediately
        barrier->gpu_done[idx] = true;
        break;
    default:
        // For unimplemented backends, assume done
        barrier->gpu_done[idx] = true;
        break;
    }

    return barrier->gpu_done[idx];
}

bool sp_hetero_barrier_poll(sp_hetero_barrier_t *barrier) {
    bool all_done = true;
    for (int i = 0; i < barrier->n_gpus; i++) {
        if (!sp_poll_gpu(barrier, i)) {
            all_done = false;
        }
    }
    return all_done;
}

uint64_t sp_hetero_barrier_wait(sp_hetero_barrier_t *barrier) {
    uint64_t t0 = sp_time_us();
    bool prefetch_done = false;

    // Phase 1: Spin for a while (low latency when GPUs finish fast)
    for (int spin = 0; spin < barrier->spin_count; spin++) {
        if (sp_hetero_barrier_poll(barrier)) goto done;

        // Use dead time for pre-shredding
        if (!prefetch_done && barrier->prefetch_callback &&
            barrier->next_expert_hint >= 0)
        {
            uint64_t pf0 = sp_time_us();
            barrier->prefetch_callback(barrier->prefetch_user_data,
                                       barrier->next_expert_hint);
            barrier->total_prefetch_us += sp_time_us() - pf0;
            prefetch_done = true;
        }
    }

    // Phase 2: Yield-based wait (save power, let other threads run)
    while (!sp_hetero_barrier_poll(barrier)) {
        sp_yield();

        // Pre-shred if we haven't yet
        if (!prefetch_done && barrier->prefetch_callback &&
            barrier->next_expert_hint >= 0)
        {
            uint64_t pf0 = sp_time_us();
            barrier->prefetch_callback(barrier->prefetch_user_data,
                                       barrier->next_expert_hint);
            barrier->total_prefetch_us += sp_time_us() - pf0;
            prefetch_done = true;
        }
    }

done:
    ;
    uint64_t elapsed = sp_time_us() - t0;

    // Update diagnostics
    barrier->last_barrier_us = elapsed;
    barrier->total_barrier_us += elapsed;
    barrier->total_barriers++;
    if (elapsed > barrier->max_barrier_us) barrier->max_barrier_us = elapsed;
    if (elapsed < barrier->min_barrier_us) barrier->min_barrier_us = elapsed;

    // Reset completion flags for next dispatch
    for (int i = 0; i < barrier->n_gpus; i++) {
        barrier->gpu_done[i] = false;
    }
    barrier->next_expert_hint = -1;

    return elapsed;
}

// ============================================================================
// CUDA event operations (stub — linked when SP_WITH_CUDA is defined)
// ============================================================================

#ifdef SP_WITH_CUDA
// These call through to the actual CUDA runtime.
// At link time, the engine provides the CUDA implementation.
// The signatures match cuda_runtime.h expectations.

int sp_hetero_cuda_record_event(sp_gpu_device_t *gpu) {
    // cudaEventRecord((cudaEvent_t)gpu->event, (cudaStream_t)gpu->stream);
    // return cudaSuccess check
    return 0;
}

int sp_hetero_cuda_sync_event(sp_gpu_device_t *gpu) {
    // cudaError_t err = cudaEventQuery((cudaEvent_t)gpu->event);
    // return (err == cudaSuccess) ? 0 : -1;
    return 0;
}
#endif

// ============================================================================
// Vulkan fence operations (stub)
// ============================================================================

#ifdef SP_WITH_VULKAN
int sp_hetero_vulkan_submit_fence(sp_gpu_device_t *gpu) {
    // vkQueueSubmit + fence signal
    return 0;
}

int sp_hetero_vulkan_wait_fence(sp_gpu_device_t *gpu) {
    // VkResult r = vkGetFenceStatus(device, fence);
    // return (r == VK_SUCCESS) ? 0 : -1;
    return 0;
}
#endif

// ============================================================================
// Diagnostics
// ============================================================================

void sp_hetero_barrier_print_status(const sp_hetero_barrier_t *barrier) {
    fprintf(stderr, "\n=== HETEROGENEOUS BARRIER STATUS ===\n");
    fprintf(stderr, "GPUs:           %d\n", barrier->n_gpus);
    for (int i = 0; i < barrier->n_gpus; i++) {
        const sp_gpu_device_t *g = &barrier->gpu[i];
        fprintf(stderr, "  GPU[%d]: %s\n", i, g->name);
        fprintf(stderr, "    Dispatches:  %llu\n", (unsigned long long)g->dispatch_count);
        fprintf(stderr, "    Dispatch us: %llu\n", (unsigned long long)g->total_dispatch_us);
        fprintf(stderr, "    Wait us:     %llu\n", (unsigned long long)g->total_wait_us);
    }
    fprintf(stderr, "Barriers:       %llu\n", (unsigned long long)barrier->total_barriers);
    fprintf(stderr, "Avg wait:       %.1f us\n", sp_hetero_avg_barrier_us(barrier));
    fprintf(stderr, "Min/Max wait:   %llu / %llu us\n",
            (unsigned long long)barrier->min_barrier_us,
            (unsigned long long)barrier->max_barrier_us);
    fprintf(stderr, "Pre-shred time: %.2f ms (overlapped with GPU)\n",
            (double)barrier->total_prefetch_us / 1000.0);
    fprintf(stderr, "====================================\n\n");
}

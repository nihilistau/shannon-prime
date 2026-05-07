// Shannon-Prime Beast Canyon: Shadow-Steal Speculative Dispatch
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "sp_shadow_steal.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#  include <windows.h>
static uint64_t sp_steal_time_us(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (uint64_t)(c.QuadPart * 1000000ULL / f.QuadPart);
}
#else
#  include <time.h>
static uint64_t sp_steal_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000;
}
#endif

// ---------------------------------------------------------------------------
//  Clamp tau to valid range
// ---------------------------------------------------------------------------
static float clamp_tau(float tau) {
    if (tau < SP_SHADOW_STEAL_TAU_MIN) return SP_SHADOW_STEAL_TAU_MIN;
    if (tau > SP_SHADOW_STEAL_TAU_MAX) return SP_SHADOW_STEAL_TAU_MAX;
    return tau;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_init
// ---------------------------------------------------------------------------
int sp_shadow_steal_init(sp_shadow_steal_t* ctx, size_t expert_dim, float tau) {
    if (!ctx) return -1;
    memset(ctx, 0, sizeof(*ctx));

    ctx->tau      = clamp_tau(tau);
    ctx->tau_base = ctx->tau;
    ctx->state    = SP_STEAL_IDLE;

    // Allocate shadow buffers — two slots for top-2 predicted experts.
    // In a full Level-Zero implementation these would be SVM allocations
    // on the Ring Bus. For now we use aligned CPU memory as the staging
    // area — the L0 dispatch path is stubbed below.
    size_t buf_bytes = expert_dim * sizeof(float);
    for (int i = 0; i < 2; i++) {
#ifdef _WIN32
        ctx->slots[i].data = _aligned_malloc(buf_bytes, 64);
#else
        ctx->slots[i].data = aligned_alloc(64, (buf_bytes + 63) & ~(size_t)63);
#endif
        if (!ctx->slots[i].data) {
            sp_shadow_steal_free(ctx);
            return -1;
        }
        memset(ctx->slots[i].data, 0, buf_bytes);
        ctx->slots[i].size      = buf_bytes;
        ctx->slots[i].expert_id = -1;
        ctx->slots[i].ready     = false;
        ctx->slots[i].target    = SP_GPU_SECONDARY;
    }

    // Probe for Level-Zero command queue.
    // TODO: zeDriverGet -> zeDeviceGet -> zeCommandQueueCreate for Intel UHD.
    // For now, ctx->l0_command_queue remains NULL and we operate in stub mode.
    ctx->l0_command_queue = NULL;

    // Enable speculation — prediction + hit/miss tracking is fully functional
    // even without L0 dispatch (the compute results are stubbed as instant).
    ctx->enabled = true;

    return 0;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_free
// ---------------------------------------------------------------------------
void sp_shadow_steal_free(sp_shadow_steal_t* ctx) {
    if (!ctx) return;
    for (int i = 0; i < 2; i++) {
        if (ctx->slots[i].data) {
#ifdef _WIN32
            _aligned_free(ctx->slots[i].data);
#else
            free(ctx->slots[i].data);
#endif
        }
    }
    // TODO: zeCommandQueueDestroy if l0_command_queue != NULL
    memset(ctx, 0, sizeof(*ctx));
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_speculate
// ---------------------------------------------------------------------------
int sp_shadow_steal_speculate(sp_shadow_steal_t* ctx,
                              sp_moe_curriculum_t* curriculum,
                              int next_layer) {
    if (!ctx || !ctx->enabled || !curriculum) return -1;
    if (ctx->state != SP_STEAL_IDLE) return -1;

    // Query the Curriculum EWMA heatmap for top-2 hottest experts.
    int top2[2] = { -1, -1 };
    sp_moe_get_hottest_k(curriculum, next_layer, 2, top2);
    if (top2[0] < 0) return 1;  // No prediction available

    // Check combined confidence of top-2 against tau.
    float combined = sp_moe_top_k_confidence(curriculum, next_layer, 2);
    if (!sp_shadow_steal_should_speculate(ctx, combined)) {
        return 1;  // Below confidence threshold
    }

    // Populate shadow buffer slots.
    ctx->slots[0].expert_id = top2[0];
    ctx->slots[0].ready     = false;
    ctx->slots[1].expert_id = top2[1];
    ctx->slots[1].ready     = false;

    ctx->state = SP_STEAL_SPECULATING;
    ctx->stats.total_steals++;

    // TODO: Level-Zero dispatch path.
    //   1. Load expert weights from Optane via sp_optane_prefetch_expert()
    //   2. Shred to fp16 staging via sp_shredder_process()
    //   3. zeCommandListAppendMemoryCopy -> UHD SVM
    //   4. zeCommandListAppendLaunchKernel -> expert matmul
    //   5. zeCommandQueueExecuteCommandLists (async)
    //
    // For now, mark slots as "ready" immediately (simulating instant compute).
    ctx->slots[0].ready = true;
    ctx->slots[1].ready = true;

    return 0;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_check
// ---------------------------------------------------------------------------
sp_steal_state_t sp_shadow_steal_check(sp_shadow_steal_t* ctx,
                                       const int* actual_experts,
                                       int n_actual) {
    if (!ctx) return SP_STEAL_IDLE;
    if (ctx->state != SP_STEAL_SPECULATING) return SP_STEAL_IDLE;

    // Compare shadow slots against the Router's actual selection.
    bool hit = false;
    for (int s = 0; s < 2; s++) {
        for (int a = 0; a < n_actual; a++) {
            if (ctx->slots[s].expert_id == actual_experts[a] && ctx->slots[s].ready) {
                hit = true;
                break;
            }
        }
        if (hit) break;
    }

    if (hit) {
        ctx->state = SP_STEAL_HIT;
        ctx->stats.hits++;
        // Approximate: typical expert matmul on UHD ~3ms saved on HIT.
        ctx->stats.total_steal_time_ms += 3.0f;
        ctx->stats.net_time_saved_ms   += 3.0f;
    } else {
        ctx->state = SP_STEAL_MISS;
        ctx->stats.misses++;
        // Approximate: Ring Bus flush cost ~2ms.
        ctx->stats.total_flush_time_ms += 2.0f;
        ctx->stats.net_time_saved_ms   -= 2.0f;
    }

    if (ctx->stats.total_steals > 0) {
        ctx->stats.hit_rate = (float)ctx->stats.hits / (float)ctx->stats.total_steals;
    }

    return ctx->state;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_accept
// ---------------------------------------------------------------------------
void sp_shadow_steal_accept(sp_shadow_steal_t* ctx, int slot) {
    if (!ctx || slot < 0 || slot > 1) return;
    // In full implementation: pointer swap into MoE accumulator. Cost: ~10ns.
    ctx->slots[slot].ready = false;
    ctx->state = SP_STEAL_IDLE;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_abort
// ---------------------------------------------------------------------------
void sp_shadow_steal_abort(sp_shadow_steal_t* ctx) {
    if (!ctx) return;
    if (ctx->state == SP_STEAL_SPECULATING) {
        ctx->stats.aborts++;
        // TODO: zeCommandListReset to flush UHD queue.
    }
    ctx->slots[0].ready     = false;
    ctx->slots[0].expert_id = -1;
    ctx->slots[1].ready     = false;
    ctx->slots[1].expert_id = -1;
    ctx->state = SP_STEAL_IDLE;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_set_tau
// ---------------------------------------------------------------------------
void sp_shadow_steal_set_tau(sp_shadow_steal_t* ctx, float tau) {
    if (!ctx) return;
    ctx->tau      = clamp_tau(tau);
    ctx->tau_base = ctx->tau;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_get_stats
// ---------------------------------------------------------------------------
sp_shadow_steal_stats_t sp_shadow_steal_get_stats(const sp_shadow_steal_t* ctx) {
    if (!ctx) {
        sp_shadow_steal_stats_t empty;
        memset(&empty, 0, sizeof(empty));
        return empty;
    }
    return ctx->stats;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_log_efficiency
// ---------------------------------------------------------------------------
void sp_shadow_steal_log_efficiency(const sp_shadow_steal_t* ctx) {
    if (!ctx) return;
    const sp_shadow_steal_stats_t* s = &ctx->stats;

    fprintf(stderr, "[shadow-steal] steals=%llu hits=%llu (%.1f%%) misses=%llu aborts=%llu\n",
            (unsigned long long)s->total_steals,
            (unsigned long long)s->hits,
            s->hit_rate * 100.0f,
            (unsigned long long)s->misses,
            (unsigned long long)s->aborts);

    fprintf(stderr, "[shadow-steal] net_saved=%+.1fms  steal_time=%.1fms  flush_time=%.1fms\n",
            s->net_time_saved_ms,
            s->total_steal_time_ms,
            s->total_flush_time_ms);

    fprintf(stderr, "[shadow-steal] UHD utilisation: %.1f%%  tau=%.3f\n",
            s->uhd_utilisation_pct,
            ctx->tau);
}

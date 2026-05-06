// Shannon-Prime Beast Canyon: Heterogeneous MoE Orchestrator — Implementation
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "sp_beast_canyon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <time.h>
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#endif

// ============================================================================
// Timing
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

// ============================================================================
// Config defaults
// ============================================================================

void sp_beast_config_init(sp_beast_config_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->cuda_device = -1;       // Auto-detect
    cfg->vulkan_device = -1;     // Auto-detect
    cfg->force_cpu_only = false;
    cfg->n_experts_per_token = 0; // Use model's default
    cfg->expert_round_robin = true;
    cfg->enable_sidecar = false;
    cfg->sidecar_port = 9876;
    cfg->shredder_prefetch = 8;
    cfg->staging_elements = 0;   // Auto-size at init
    cfg->optane_budget = 0;      // Use all available
    cfg->preload_hot_layers = false;
    cfg->enable_dashboard = true;
    cfg->dashboard_interval_ms = 500;

    // Shannon-Prime defaults
    sp_config_init(&cfg->sp_config, 128, 32, 8); // head_dim=128, n_layers=32, n_kv=8
}

// ============================================================================
// Ping-pong buffer management
// ============================================================================

static int sp_pingpong_init(sp_pingpong_t *pp, size_t elements) {
    pp->buf_elements = elements;
    pp->buf_size = elements * sizeof(uint16_t);
    pp->active = 0;
    pp->filling = 1;
    pp->fill_ready = false;

    for (int i = 0; i < 2; i++) {
#ifdef _WIN32
        pp->buffers[i] = (uint16_t *)_aligned_malloc(pp->buf_size, 64);
#else
        void *ptr = NULL;
        if (posix_memalign(&ptr, 64, pp->buf_size) != 0) ptr = NULL;
        pp->buffers[i] = (uint16_t *)ptr;
#endif
        if (!pp->buffers[i]) {
            fprintf(stderr, "[sp-beast] ERROR: failed to allocate ping-pong buffer %d\n", i);
            return -1;
        }
        // Pre-fault pages
        memset(pp->buffers[i], 0, pp->buf_size);
    }

    return 0;
}

static void sp_pingpong_free(sp_pingpong_t *pp) {
    for (int i = 0; i < 2; i++) {
        if (pp->buffers[i]) {
#ifdef _WIN32
            _aligned_free(pp->buffers[i]);
#else
            free(pp->buffers[i]);
#endif
            pp->buffers[i] = NULL;
        }
    }
}

// Flip: the buffer that was being filled becomes active for GPU,
//       and vice versa. This is the "zero-copy pointer swap".
static inline void sp_pingpong_flip(sp_pingpong_t *pp) {
    pp->active  = 1 - pp->active;
    pp->filling = 1 - pp->filling;
    pp->fill_ready = false;
}

// ============================================================================
// Boot sequence
// ============================================================================

int sp_beast_init(sp_beast_engine_t *engine, const sp_beast_config_t *cfg) {
    memset(engine, 0, sizeof(*engine));
    engine->config = *cfg;
    uint64_t t0 = sp_time_us();

    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  SHANNON-PRIME: BEAST CANYON ENGINE          ║\n");
    fprintf(stderr, "║  Heterogeneous MoE Orchestrator              ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════════╝\n\n");

    // ── Stage 1: Map the Reservoir ──────────────────────────────────
    fprintf(stderr, "[BOOT] Stage 1: Mapping Optane reservoir...\n");
    int rc = sp_optane_init(&engine->reservoir, cfg->gguf_path);
    if (rc != 0) {
        fprintf(stderr, "[BOOT] FATAL: Optane mapping failed (rc=%d)\n", rc);
        return rc;
    }

    // ── Stage 2: Initialize the Shredder ────────────────────────────
    fprintf(stderr, "[BOOT] Stage 2: Initializing AVX-512 Shredder...\n");
    sp_shredder_config_t shred_cfg;
    sp_shredder_config_init(&shred_cfg);
    shred_cfg.prefetch_pages = cfg->shredder_prefetch;

    // Auto-size staging: enough for one expert's largest tensor
    size_t staging_elems = cfg->staging_elements;
    if (staging_elems == 0) {
        // Find the largest expert projection tensor
        if (engine->reservoir.is_moe) {
            for (int e = 0; e < engine->reservoir.n_experts; e++) {
                const sp_optane_expert_t *exp = &engine->reservoir.experts[e];
                if (exp->gate_proj && exp->gate_proj->n_bytes > staging_elems * 2) {
                    // Estimate elements from bytes (Q4_0: 32 elems per 18 bytes)
                    staging_elems = exp->gate_proj->n_bytes * 32 / 18;
                }
            }
        }
        // Fallback: enough for a 14336-wide intermediate × 4096 hidden
        if (staging_elems == 0) {
            staging_elems = 14336 * 4096;
        }
        // Double for ping-pong
        staging_elems *= 2;
    }

    rc = sp_shredder_init(&engine->shredder, &shred_cfg, staging_elems);
    if (rc != 0) {
        sp_optane_free(&engine->reservoir);
        return rc;
    }

    // ── Stage 3: Detect and initialize GPUs ─────────────────────────
    fprintf(stderr, "[BOOT] Stage 3: Detecting GPU hardware...\n");
    sp_hetero_barrier_init(&engine->barrier);

    if (!cfg->force_cpu_only) {
        // Add CUDA GPU if requested
        if (cfg->cuda_device >= 0) {
            sp_hetero_add_gpu(&engine->barrier, SP_GPU_CUDA, cfg->cuda_device);
        } else {
            // Auto-detect: try device 0
            // TODO: actual CUDA runtime detection
            fprintf(stderr, "[BOOT] CUDA: will init at first dispatch (lazy)\n");
        }

        // Add Vulkan/Xe GPU if requested
        if (cfg->vulkan_device >= 0) {
            sp_hetero_add_gpu(&engine->barrier, SP_GPU_VULKAN, cfg->vulkan_device);
        } else {
            // Auto-detect Intel Xe iGPU
            fprintf(stderr, "[BOOT] Vulkan/Xe: will init at first dispatch (lazy)\n");
        }
    } else {
        fprintf(stderr, "[BOOT] CPU-only mode (GPUs disabled)\n");
    }

    // ── Stage 4: Initialize ping-pong buffers ───────────────────────
    fprintf(stderr, "[BOOT] Stage 4: Initializing ping-pong buffers...\n");
    size_t pp_elements = staging_elems / 2;  // Each half gets half the staging
    for (int i = 0; i < 2; i++) {
        rc = sp_pingpong_init(&engine->pingpong[i], pp_elements);
        if (rc != 0) {
            fprintf(stderr, "[BOOT] WARNING: ping-pong buffer %d failed, degraded mode\n", i);
        }
    }

    // ── Stage 5: Connect sidecar (optional) ─────────────────────────
    engine->sidecar.state = SP_SIDECAR_DISCONNECTED;
    if (cfg->enable_sidecar) {
        fprintf(stderr, "[BOOT] Stage 5: Connecting S22U sidecar...\n");
        sp_beast_sidecar_connect(engine);
    } else {
        fprintf(stderr, "[BOOT] Stage 5: Sidecar disabled (CPU handles Prime-PE)\n");
    }

    // ── Stage 6: Initialize KV cache ────────────────────────────────
    fprintf(stderr, "[BOOT] Stage 6: Initializing Shannon-Prime KV cache...\n");
    // The KV cache uses the model's actual dimensions
    if (engine->reservoir.n_layer > 0 && engine->reservoir.head_dim > 0) {
        sp_config_t kv_cfg = cfg->sp_config;
        kv_cfg.head_dim = engine->reservoir.head_dim;
        kv_cfg.n_layers = engine->reservoir.n_layer;
        kv_cfg.n_heads_kv = engine->reservoir.n_head_kv;
        // KV cache allocation happens at first inference (lazy)
        fprintf(stderr, "[BOOT] KV cache: %u layers × %u heads × hd=%u (lazy alloc)\n",
                kv_cfg.n_layers, kv_cfg.n_heads_kv, kv_cfg.head_dim);
    }

    // ── Boot complete ───────────────────────────────────────────────
    engine->boot_time_us = sp_time_us() - t0;

    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  BEAST CANYON ENGINE: ONLINE                 ║\n");
    fprintf(stderr, "║  Boot time: %8.2f ms                       ║\n",
            (double)engine->boot_time_us / 1000.0);
    fprintf(stderr, "║  Reservoir: %.1f MB (%s)%*s║\n",
            (double)engine->reservoir.file_size / (1024.0*1024.0),
            engine->reservoir.architecture,
            (int)(20 - strlen(engine->reservoir.architecture)), "");
    fprintf(stderr, "║  GPUs:      %d configured                     ║\n",
            engine->barrier.n_gpus);
    fprintf(stderr, "║  Sidecar:   %s                       ║\n",
            engine->sidecar.state == SP_SIDECAR_ONLINE ? "ONLINE " :
            engine->sidecar.state == SP_SIDECAR_ERROR  ? "ERROR  " : "OFFLINE");
    fprintf(stderr, "╚══════════════════════════════════════════════╝\n\n");

    return 0;
}

void sp_beast_free(sp_beast_engine_t *engine) {
    fprintf(stderr, "[sp-beast] Shutting down Beast Canyon engine...\n");

    // Order matters: Level Zero/Vulkan before CUDA (per safeguard spec)
    sp_beast_sidecar_disconnect(engine);

    // Free ping-pong buffers
    for (int i = 0; i < 2; i++) {
        sp_pingpong_free(&engine->pingpong[i]);
    }

    sp_hetero_barrier_free(&engine->barrier);
    sp_shredder_free(&engine->shredder);
    sp_optane_free(&engine->reservoir);

    fprintf(stderr, "[sp-beast] Engine shutdown complete.\n");
}

// ============================================================================
// Expert Routing (The Oracle)
// ============================================================================

void sp_beast_route_experts(sp_beast_engine_t *engine,
                            const float *router_logits, int n_experts)
{
    sp_expert_routing_t *r = &engine->routing;
    int top_k = engine->config.n_experts_per_token;
    if (top_k <= 0) top_k = engine->reservoir.n_experts_per_token;
    if (top_k <= 0) top_k = 2;  // Default top-2
    if (top_k > n_experts) top_k = n_experts;

    // Softmax over router logits (for scores)
    float max_logit = -1e30f;
    for (int i = 0; i < n_experts; i++) {
        if (router_logits[i] > max_logit) max_logit = router_logits[i];
    }
    float sum_exp = 0.0f;
    float exp_vals[SP_OPTANE_MAX_EXPERTS];
    for (int i = 0; i < n_experts; i++) {
        exp_vals[i] = expf(router_logits[i] - max_logit);
        sum_exp += exp_vals[i];
    }

    // Top-K selection (simple partial sort)
    bool selected[SP_OPTANE_MAX_EXPERTS];
    memset(selected, 0, sizeof(selected));

    r->n_selected = top_k;
    for (int k = 0; k < top_k; k++) {
        int best = -1;
        float best_score = -1e30f;
        for (int i = 0; i < n_experts; i++) {
            if (!selected[i] && exp_vals[i] / sum_exp > best_score) {
                best = i;
                best_score = exp_vals[i] / sum_exp;
            }
        }
        r->expert_ids[k] = best;
        r->expert_scores[k] = best_score;
        selected[best] = true;

        // GPU assignment: round-robin across available GPUs
        if (engine->barrier.n_gpus >= 2 && engine->config.expert_round_robin) {
            r->gpu_assignment[k] = k % engine->barrier.n_gpus;
        } else if (engine->barrier.n_gpus == 1) {
            r->gpu_assignment[k] = 0;
        } else {
            r->gpu_assignment[k] = -1;  // CPU-only
        }
    }
}

// ============================================================================
// MoE Forward — The Execution Pulse
// ============================================================================

int sp_beast_moe_forward(sp_beast_engine_t *engine,
                         const float *router_logits,
                         const float *hidden_states,
                         float *output,
                         int layer)
{
    uint64_t t0 = sp_time_us();
    int n_experts = engine->reservoir.n_experts;
    if (n_experts <= 0) return -1;  // Not an MoE model

    // ── Step 1: Oracle — route experts ──────────────────────────────
    sp_beast_route_experts(engine, router_logits, n_experts);
    const sp_expert_routing_t *r = &engine->routing;

    // ── Step 2: Shred — dequantize selected experts from Optane ─────
    for (int k = 0; k < r->n_selected; k++) {
        int eid = r->expert_ids[k];
        const sp_optane_expert_t *exp = sp_optane_expert(&engine->reservoir, eid);
        if (!exp || !exp->gate_proj) continue;

        // Prefetch the NEXT expert while shredding the current one
        if (k + 1 < r->n_selected) {
            sp_optane_prefetch_expert(&engine->reservoir, r->expert_ids[k+1]);
        }

        int gpu_idx = r->gpu_assignment[k];
        sp_pingpong_t *pp = (gpu_idx >= 0 && gpu_idx < 2)
                            ? &engine->pingpong[gpu_idx] : NULL;

        if (pp && pp->buffers[pp->filling]) {
            // Shred gate_proj into the filling half of the ping-pong buffer
            sp_shredder_auto(&engine->shredder,
                             exp->gate_proj->type,
                             exp->gate_proj->ptr,
                             pp->buffers[pp->filling],
                             exp->gate_proj->ne[0] * exp->gate_proj->ne[1]);
            pp->fill_ready = true;
        }
    }

    // ── Step 3: Dispatch — launch on GPUs ───────────────────────────
    for (int k = 0; k < r->n_selected; k++) {
        int gpu_idx = r->gpu_assignment[k];
        if (gpu_idx < 0 || gpu_idx >= engine->barrier.n_gpus) continue;

        sp_hetero_mark_dispatched(&engine->barrier, gpu_idx);

        // TODO: actual kernel launch (CUDA / Vulkan)
        // For now, mark as done immediately (CPU fallback)
        sp_hetero_mark_done(&engine->barrier, gpu_idx);
    }

    // ── Step 4: Barrier — wait for all GPUs ─────────────────────────
    // Set pre-shred hint for the barrier wait callback
    engine->barrier.next_expert_hint = -1; // No speculative prefetch yet
    uint64_t barrier_us = sp_hetero_barrier_wait(&engine->barrier);

    // ── Step 5: Sum — merge expert outputs in LLC via AVX-512 ───────
    // Initialize output to zero
    size_t hidden_dim = engine->reservoir.n_embd;
    memset(output, 0, hidden_dim * sizeof(float));

    // Weighted sum of expert outputs
    // TODO: actual GPU result readback + AVX-512 fused addition
    // For now, passthrough hidden states (identity for skeleton testing)
    for (size_t i = 0; i < hidden_dim; i++) {
        output[i] = hidden_states[i];
    }

    // ── Step 6: Flip ping-pong buffers ──────────────────────────────
    for (int i = 0; i < 2; i++) {
        if (engine->pingpong[i].fill_ready) {
            sp_pingpong_flip(&engine->pingpong[i]);
        }
    }

    // Update counters
    engine->total_tokens++;
    uint64_t elapsed = sp_time_us() - t0;
    engine->total_inference_us += elapsed;
    engine->total_barrier_us += barrier_us;

    return 0;
}

// ============================================================================
// Sidecar — S22U via ADB USB-C
// ============================================================================

int sp_beast_sidecar_connect(sp_beast_engine_t *engine) {
    sp_sidecar_t *sc = &engine->sidecar;
    sc->port = engine->config.sidecar_port;
    sc->state = SP_SIDECAR_CONNECTING;

    // Try to connect to localhost:<port> (ADB port-forwarded to S22U)
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);

    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET) {
        sc->state = SP_SIDECAR_DISCONNECTED;
        fprintf(stderr, "[sp-sidecar] Socket creation failed\n");
        return -1;
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons((u_short)sc->port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        closesocket(sock);
        sc->state = SP_SIDECAR_DISCONNECTED;
        fprintf(stderr, "[sp-sidecar] S22U not available at port %d (run: adb forward tcp:%d tcp:%d)\n",
                sc->port, sc->port, sc->port);
        return -1;
    }

    sc->socket_fd = (int)sock;
#else
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        sc->state = SP_SIDECAR_DISCONNECTED;
        return -1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(sc->port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        sc->state = SP_SIDECAR_DISCONNECTED;
        fprintf(stderr, "[sp-sidecar] S22U not available at port %d\n", sc->port);
        return -1;
    }

    sc->socket_fd = sock;
#endif

    sc->state = SP_SIDECAR_ONLINE;
    fprintf(stderr, "[sp-sidecar] S22U connected at port %d\n", sc->port);
    return 0;
}

void sp_beast_sidecar_disconnect(sp_beast_engine_t *engine) {
    sp_sidecar_t *sc = &engine->sidecar;
    if (sc->state == SP_SIDECAR_ONLINE || sc->socket_fd > 0) {
#ifdef _WIN32
        closesocket((SOCKET)sc->socket_fd);
        WSACleanup();
#else
        close(sc->socket_fd);
#endif
        sc->socket_fd = -1;
    }
    sc->state = SP_SIDECAR_DISCONNECTED;
}

int sp_beast_sidecar_prime_pe(sp_beast_engine_t *engine,
                              const float *hidden_states, int dim,
                              float *pe_output)
{
    if (engine->sidecar.state != SP_SIDECAR_ONLINE) {
        // CPU fallback: run Prime-PE on host
        // sp_prime_pe_forward(hidden_states, dim, pe_output, ...);
        memcpy(pe_output, hidden_states, dim * sizeof(float));
        return -1;  // Indicate fallback
    }

    // TODO: serialize hidden_states, send via socket, recv result
    // Protocol: [dim:u32][data:f32*dim] → [result:f32*dim]
    uint64_t t0 = sp_time_us();

    // For now, fallback
    memcpy(pe_output, hidden_states, dim * sizeof(float));

    engine->sidecar.total_offloads++;
    engine->sidecar.total_offload_us += sp_time_us() - t0;

    return 0;
}

// ============================================================================
// Attention forward (non-MoE layers)
// ============================================================================

int sp_beast_attention_forward(sp_beast_engine_t *engine,
                               const float *q, const float *k, const float *v,
                               float *output,
                               int layer, int pos, int kv_len)
{
    // Write K/V to Shannon-Prime compressed cache
    // sp_shadow_write_k(cache, layer, head, pos, k);
    // sp_shadow_write_v(cache, layer, head, pos, v);

    // Read back full KV for attention
    // Compute attention scores
    // Output = softmax(Q @ K^T / sqrt(d)) @ V

    // TODO: actual implementation using existing engine infrastructure
    // For skeleton testing, passthrough Q
    size_t head_dim = engine->reservoir.head_dim;
    memcpy(output, q, head_dim * sizeof(float));

    return 0;
}

// ============================================================================
// Full forward pass (skeleton)
// ============================================================================

int sp_beast_forward(sp_beast_engine_t *engine,
                     int input_token,
                     float *logits)
{
    // This is the skeleton for the full forward pass.
    // The real implementation will call into the engine's existing
    // forward path (forward.cpp or forward_native.cpp) but with
    // Optane-backed weights and dual-GPU dispatch for MoE layers.

    // For non-MoE models, this dispatches to the standard path.
    // For MoE models, attention layers use the standard path while
    // MoE MLP layers use sp_beast_moe_forward().

    fprintf(stderr, "[sp-beast] Forward: token=%d (skeleton)\n", input_token);
    return 0;
}

int sp_beast_generate(sp_beast_engine_t *engine,
                      const int *prompt_tokens, int n_prompt,
                      int *output_tokens, int max_tokens,
                      float temperature, float top_p)
{
    fprintf(stderr, "[sp-beast] Generate: prompt=%d tokens, max=%d\n",
            n_prompt, max_tokens);

    // Skeleton: will wire into engine's generate loop
    return 0;
}

// ============================================================================
// ASCII Dashboard
// ============================================================================

void sp_beast_dashboard(const sp_beast_engine_t *engine) {
    double tok_s = sp_beast_tok_per_sec(engine);
    double shred_gbps = sp_shredder_throughput_gbps(&engine->shredder);
    double avg_barrier = sp_hetero_avg_barrier_us(&engine->barrier);

    fprintf(stderr, "\r");
    fprintf(stderr, "[ OPTANE ] ────── ");
    fprintf(stderr, "[ SHREDDER %.1f GB/s ] ────── ", shred_gbps);
    fprintf(stderr, "[ LLC ] ────── ");

    if (engine->barrier.n_gpus >= 1) {
        fprintf(stderr, "[ GPU0: %s ] ", engine->barrier.gpu[0].name);
    }
    if (engine->barrier.n_gpus >= 2) {
        fprintf(stderr, "[ GPU1: %s ] ", engine->barrier.gpu[1].name);
    }

    fprintf(stderr, "  %.1f tok/s  barrier=%.0fus  ", tok_s, avg_barrier);

    if (engine->sidecar.state == SP_SIDECAR_ONLINE) {
        fprintf(stderr, "[ S22U: ONLINE ]");
    }

    fflush(stderr);
}

// ============================================================================
// Full status print
// ============================================================================

void sp_beast_print_status(const sp_beast_engine_t *engine) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  BEAST CANYON ENGINE STATUS                  ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Tokens:     %10llu                       ║\n",
            (unsigned long long)engine->total_tokens);
    fprintf(stderr, "║  Tok/s:      %10.1f                       ║\n",
            sp_beast_tok_per_sec(engine));
    fprintf(stderr, "║  Inference:  %10.2f ms                    ║\n",
            (double)engine->total_inference_us / 1000.0);
    fprintf(stderr, "║  Barrier:    %10.2f ms (avg %.0f us)       ║\n",
            (double)engine->total_barrier_us / 1000.0,
            sp_hetero_avg_barrier_us(&engine->barrier));
    fprintf(stderr, "╠══════════════════════════════════════════════╣\n");

    sp_optane_print_status(&engine->reservoir);
    sp_shredder_print_status(&engine->shredder);
    sp_hetero_barrier_print_status(&engine->barrier);
}

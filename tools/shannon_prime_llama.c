// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#include "shannon_prime_llama.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Internal context
// ============================================================================

struct sp_llama_ctx_s {
    sp_llama_params_t  params;
    sp_config_t        config;

    // Backend-specific cache (exactly one is active)
    sp_shadow_cache_t  cpu_cache;      // CPU backend
    // sp_cuda_cache_t    cuda_cache;  // CUDA backend (when linked)
    // sp_vulkan_cache_t *vulkan_cache; // Vulkan backend
    // sp_adreno_cache_t  adreno_cache; // Adreno backend

    int active_backend;  // Which backend is in use
    int n_positions;     // Current max written position
};

// ============================================================================
// Environment variable parsing
// ============================================================================

static int parse_env_bool(const char *name, int default_val) {
    const char *v = getenv(name);
    if (!v) return default_val;
    return (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

static void parse_env_bits(const char *name, int *bits, int n, const int *defaults) {
    const char *v = getenv(name);
    if (!v) {
        memcpy(bits, defaults, n * sizeof(int));
        return;
    }

    // Parse comma-separated: "5,5,4,3"
    int i = 0;
    const char *p = v;
    while (i < n && *p) {
        bits[i++] = atoi(p);
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    // Fill remaining with last value
    while (i < n) {
        bits[i] = bits[i-1];
        i++;
    }
}

// ============================================================================
// Lifecycle
// ============================================================================

sp_llama_ctx_t *sp_llama_init(const sp_llama_params_t *params) {
    if (!parse_env_bool("SHANNON_PRIME_ENABLED", 0)) {
        return NULL;
    }

    sp_config_t cfg;
    sp_config_init(&cfg, params->head_dim, params->n_layers, params->n_heads_kv);

    // Override from environment
    int k_defaults[] = {5, 5, 4, 3};
    int v_defaults[] = {3};
    parse_env_bits("SHANNON_PRIME_K_BITS", cfg.k_band_bits, 4, k_defaults);
    parse_env_bits("SHANNON_PRIME_V_BITS", cfg.v_band_bits, 1, v_defaults);
    cfg.use_mobius_mask = parse_env_bool("SHANNON_PRIME_MOBIUS", 1);

    return sp_llama_init_config(params, &cfg);
}

sp_llama_ctx_t *sp_llama_init_config(const sp_llama_params_t *params,
                                     const sp_config_t *cfg) {
    sp_llama_ctx_t *ctx = (sp_llama_ctx_t *)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    memcpy(&ctx->params, params, sizeof(*params));
    memcpy(&ctx->config, cfg, sizeof(*cfg));

    // Initialize the appropriate backend
    ctx->active_backend = params->backend;

    switch (params->backend) {
    case SP_BACKEND_CPU:
    default: {
        if (sp_shadow_cache_init(&ctx->cpu_cache, cfg) != 0) {
            free(ctx);
            return NULL;
        }

        // Allocate cache storage
        int n_slots = cfg->n_layers * cfg->n_heads_kv;
        ctx->cpu_cache.k_cache = (uint8_t **)calloc(n_slots, sizeof(uint8_t *));
        ctx->cpu_cache.v_cache = (uint8_t **)calloc(n_slots, sizeof(uint8_t *));

        for (int i = 0; i < n_slots; i++) {
            ctx->cpu_cache.k_cache[i] = (uint8_t *)calloc(
                params->max_seq_len, ctx->cpu_cache.k_bands.total_bytes);
            ctx->cpu_cache.v_cache[i] = (uint8_t *)calloc(
                params->max_seq_len, ctx->cpu_cache.v_bands.total_bytes);
        }
        break;
    }
    // case SP_BACKEND_CUDA: ...
    // case SP_BACKEND_VULKAN: ...
    // case SP_BACKEND_ADRENO: ...
    }

    if (parse_env_bool("SHANNON_PRIME_VERBOSE", 0)) {
        sp_llama_print_config(ctx);
    }

    return ctx;
}

void sp_llama_free(sp_llama_ctx_t *ctx) {
    if (!ctx) return;

    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default: {
        int n_slots = ctx->config.n_layers * ctx->config.n_heads_kv;
        for (int i = 0; i < n_slots; i++) {
            free(ctx->cpu_cache.k_cache[i]);
            free(ctx->cpu_cache.v_cache[i]);
        }
        free(ctx->cpu_cache.k_cache);
        free(ctx->cpu_cache.v_cache);
        sp_shadow_cache_free(&ctx->cpu_cache);
        break;
    }
    }

    free(ctx);
}

// ============================================================================
// Write path
// ============================================================================

void sp_llama_write_kv(sp_llama_ctx_t *ctx,
                       int layer, int head, int pos,
                       const float *k_vec, const float *v_vec) {
    sp_llama_write_k(ctx, layer, head, pos, k_vec);
    sp_llama_write_v(ctx, layer, head, pos, v_vec);
}

void sp_llama_write_k(sp_llama_ctx_t *ctx,
                      int layer, int head, int pos,
                      const float *k_vec) {
    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default:
        sp_shadow_write_k(&ctx->cpu_cache, layer, head, pos, k_vec);
        break;
    }
    if (pos >= ctx->n_positions) ctx->n_positions = pos + 1;
}

void sp_llama_write_v(sp_llama_ctx_t *ctx,
                      int layer, int head, int pos,
                      const float *v_vec) {
    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default:
        sp_shadow_write_v(&ctx->cpu_cache, layer, head, pos, v_vec);
        break;
    }
}

void sp_llama_write_k_batch(sp_llama_ctx_t *ctx,
                            int layer, int head,
                            int start_pos, int n_pos,
                            const float *k_vecs) {
    int hd = ctx->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_llama_write_k(ctx, layer, head, start_pos + i, k_vecs + i * hd);
    }
}

void sp_llama_write_v_batch(sp_llama_ctx_t *ctx,
                            int layer, int head,
                            int start_pos, int n_pos,
                            const float *v_vecs) {
    int hd = ctx->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_llama_write_v(ctx, layer, head, start_pos + i, v_vecs + i * hd);
    }
}

// ============================================================================
// Read path
// ============================================================================

void sp_llama_read_k(const sp_llama_ctx_t *ctx,
                     int layer, int head, int pos,
                     float *k_out) {
    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default:
        sp_shadow_read_k(&ctx->cpu_cache, layer, head, pos, k_out);
        break;
    }
}

void sp_llama_read_v(const sp_llama_ctx_t *ctx,
                     int layer, int head, int pos,
                     float *v_out) {
    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default:
        sp_shadow_read_v(&ctx->cpu_cache, layer, head, pos, v_out);
        break;
    }
}

void sp_llama_read_k_batch(const sp_llama_ctx_t *ctx,
                           int layer, int head,
                           int start_pos, int n_pos,
                           float *k_out) {
    int hd = ctx->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_llama_read_k(ctx, layer, head, start_pos + i, k_out + i * hd);
    }
}

void sp_llama_read_v_batch(const sp_llama_ctx_t *ctx,
                           int layer, int head,
                           int start_pos, int n_pos,
                           float *v_out) {
    int hd = ctx->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_llama_read_v(ctx, layer, head, start_pos + i, v_out + i * hd);
    }
}

// ============================================================================
// Cache management
// ============================================================================

void sp_llama_clear_range(sp_llama_ctx_t *ctx,
                          int start_pos, int end_pos) {
    // Zero out the compressed cache in the given range
    int n_slots = ctx->config.n_layers * ctx->config.n_heads_kv;

    for (int s = 0; s < n_slots; s++) {
        size_t k_off = (size_t)start_pos * ctx->cpu_cache.k_bands.total_bytes;
        size_t k_len = (size_t)(end_pos - start_pos) * ctx->cpu_cache.k_bands.total_bytes;
        memset(ctx->cpu_cache.k_cache[s] + k_off, 0, k_len);

        size_t v_off = (size_t)start_pos * ctx->cpu_cache.v_bands.total_bytes;
        size_t v_len = (size_t)(end_pos - start_pos) * ctx->cpu_cache.v_bands.total_bytes;
        memset(ctx->cpu_cache.v_cache[s] + v_off, 0, v_len);
    }
}

sp_llama_memory_t sp_llama_memory(const sp_llama_ctx_t *ctx) {
    sp_llama_memory_t mem;
    int n_slots = ctx->config.n_layers * ctx->config.n_heads_kv;
    int n = ctx->n_positions;
    int hd = ctx->config.head_dim;

    mem.compressed_bytes = (size_t)n_slots * n *
        (ctx->cpu_cache.k_bands.total_bytes + ctx->cpu_cache.v_bands.total_bytes);
    mem.baseline_bytes = (size_t)n_slots * n * hd * 2 * 2; // K+V × fp16
    mem.compression_ratio = (mem.compressed_bytes > 0)
        ? (float)mem.baseline_bytes / (float)mem.compressed_bytes
        : 0.0f;
    mem.n_positions = n;
    return mem;
}

// ============================================================================
// Diagnostics
// ============================================================================

float sp_llama_validate_k(sp_llama_ctx_t *ctx,
                          const float *k_vec, int head_dim) {
    float *recon = (float *)malloc(head_dim * sizeof(float));
    sp_llama_write_k(ctx, 0, 0, 0, k_vec);
    sp_llama_read_k(ctx, 0, 0, 0, recon);
    float corr = sp_correlation_f32(k_vec, recon, head_dim);
    free(recon);
    return corr;
}

void sp_llama_print_config(const sp_llama_ctx_t *ctx) {
    fprintf(stderr, "[Shannon-Prime] llama.cpp integration\n");
    fprintf(stderr, "  Backend:  %s\n",
            ctx->active_backend == SP_BACKEND_CPU    ? "CPU" :
            ctx->active_backend == SP_BACKEND_CUDA   ? "CUDA" :
            ctx->active_backend == SP_BACKEND_VULKAN  ? "Vulkan" :
            ctx->active_backend == SP_BACKEND_ADRENO  ? "Adreno" : "unknown");
    sp_config_print(&ctx->config);
}

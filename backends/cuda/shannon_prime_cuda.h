// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#ifndef SHANNON_PRIME_CUDA_H
#define SHANNON_PRIME_CUDA_H

#include "../../core/shannon_prime.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CUDA Shadow Cache
// ============================================================================
//
// GPU-resident compressed KV cache. The write path (WHT → Möbius → quantize)
// runs entirely on GPU. The read path (dequantize → unreorder → iWHT) runs
// on GPU and produces fp16 K/V vectors ready for attention.
//
// Integration point for llama.cpp:
//   ggml_backend_cuda_set_kv_cache_hooks(sp_cuda_write_k, sp_cuda_read_k, ...)

typedef struct {
    sp_config_t     config;
    sp_band_config_t k_bands;
    sp_band_config_t v_bands;

    // GPU-resident compressed storage
    void           *d_k_cache;       // Compressed K: [n_layers * n_heads][max_seq][k_bytes]
    void           *d_v_cache;       // Compressed V: [n_layers * n_heads][max_seq][v_bytes]
    int             max_seq_len;

    // GPU-resident Möbius permutation tables
    int            *d_mobius_order;   // [head_dim] forward permutation
    int            *d_mobius_inv;     // [head_dim] inverse permutation

    // GPU scratch (per-stream)
    float          *d_scratch;        // [head_dim] working buffer
    void           *stream;           // CUDA stream
} sp_cuda_cache_t;

// Initialize CUDA shadow cache. Allocates GPU memory.
// stream: CUDA stream for async operations (NULL for default stream).
int sp_cuda_cache_init(sp_cuda_cache_t *cc, const sp_config_t *cfg,
                       int max_seq_len, void *stream);
void sp_cuda_cache_free(sp_cuda_cache_t *cc);

// ============================================================================
// Write path: raw KV → GPU WHT → Möbius reorder → band quantize → store
// ============================================================================
//
// d_k_vec: device pointer to raw K vector (head_dim floats, already RoPE'd)
// All operations run on the cache's CUDA stream.

void sp_cuda_write_k(sp_cuda_cache_t *cc,
                     int layer, int head, int pos,
                     const float *d_k_vec);

void sp_cuda_write_v(sp_cuda_cache_t *cc,
                     int layer, int head, int pos,
                     const float *d_v_vec);

// ============================================================================
// Read path: load → band dequantize → Möbius unreorder → iWHT → KV
// ============================================================================
//
// d_k_out: device pointer for reconstructed K vector (head_dim floats)

void sp_cuda_read_k(const sp_cuda_cache_t *cc,
                    int layer, int head, int pos,
                    float *d_k_out);

void sp_cuda_read_v(const sp_cuda_cache_t *cc,
                    int layer, int head, int pos,
                    float *d_v_out);

// ============================================================================
// Batch operations — process entire sequence positions at once
// ============================================================================
//
// For prefill: compress all tokens in one kernel launch.
// n_pos: number of positions to process
// d_k_vecs: [n_pos][head_dim] contiguous K vectors on device

void sp_cuda_write_k_batch(sp_cuda_cache_t *cc,
                           int layer, int head,
                           int start_pos, int n_pos,
                           const float *d_k_vecs);

void sp_cuda_write_v_batch(sp_cuda_cache_t *cc,
                           int layer, int head,
                           int start_pos, int n_pos,
                           const float *d_v_vecs);

// Read batch: reconstruct n_pos K vectors into contiguous output
void sp_cuda_read_k_batch(const sp_cuda_cache_t *cc,
                          int layer, int head,
                          int start_pos, int n_pos,
                          float *d_k_out);

void sp_cuda_read_v_batch(const sp_cuda_cache_t *cc,
                          int layer, int head,
                          int start_pos, int n_pos,
                          float *d_v_out);

// ============================================================================
// CUDA kernel launchers (exposed for testing)
// ============================================================================

// In-place WHT butterfly on GPU. n must be power of 2.
// Processes n_vecs independent vectors of length n.
void sp_cuda_wht_inplace(float *d_data, int n, int n_vecs, void *stream);

// Inverse WHT (same kernel + 1/N scaling)
void sp_cuda_iwht_inplace(float *d_data, int n, int n_vecs, void *stream);

// Apply Möbius permutation on GPU
void sp_cuda_mobius_reorder(float *d_data, const int *d_order,
                            int n, int n_vecs, void *stream);
void sp_cuda_mobius_unreorder(float *d_data, const int *d_order,
                              int n, int n_vecs, void *stream);

// Banded quantize/dequantize on GPU
void sp_cuda_band_quantize(const float *d_input, void *d_output,
                           const sp_band_config_t *bc,
                           int n_vecs, void *stream);
void sp_cuda_band_dequantize(const void *d_input, float *d_output,
                             const sp_band_config_t *bc,
                             int n_vecs, void *stream);

// NaN guard on GPU
void sp_cuda_nan_guard(float *d_data, int n, int n_vecs,
                       float max_mag, void *stream);

// ============================================================================
// Diagnostics
// ============================================================================

// Compute correlation between two GPU vectors (returns on host)
float sp_cuda_correlation(const float *d_a, const float *d_b,
                          int n, void *stream);

// Print GPU memory usage
void sp_cuda_print_memory(const sp_cuda_cache_t *cc);

#ifdef __cplusplus
}
#endif

#endif // SHANNON_PRIME_CUDA_H

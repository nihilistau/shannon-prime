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
// GPU-resident compressed KV cache. The write path (VHT2 → Möbius → quantize)
// runs entirely on GPU. The read path (dequantize → unreorder → VHT2) runs
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
// Write path: raw KV → GPU VHT2 → Möbius reorder → band quantize → store
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
// Read path: load → band dequantize → Möbius unreorder → VHT2 (self-inverse) → KV
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

// In-place VHT2 on GPU. Self-inverse — call twice to recover the input.
// Processes n_vecs independent vectors of length n. At p=2 (power-of-2 n)
// this is the 1/√2-per-stage Hartley butterfly; non-power-of-2 dims
// dispatch the staged Vilenkin kernel from shannon_prime_sqfree.cu.
void sp_cuda_vht2_forward(float *d_data, int n, int n_vecs, void *stream);

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

// ============================================================================
// Diagnostics
// ============================================================================

// Compute correlation between two GPU vectors (returns on host)
float sp_cuda_correlation(const float *d_a, const float *d_b,
                          int n, void *stream);

// Print GPU memory usage
void sp_cuda_print_memory(const sp_cuda_cache_t *cc);

// ============================================================================
// Sqfree GPU cache (step 3 MVP — no spinor yet)
// ============================================================================
//
// GPU-resident variant of sp_sqfree_cache_t. Compressed K/V blocks live
// in VRAM; compress/decompress pipelines (sqfree_pad → Vilenkin →
// Knight extract → band quantize + Möbius predict + residual quantize)
// run as CUDA kernels. Spinor sheet bit storage is deferred (full scope
// in docs/STEP3-GPU-SQFREE-CACHE.md).

typedef struct {
    sp_config_t     config;
    int             pad_dim;
    int             sk_k;
    int             n_res;
    int             n_terms;
    int             residual_bits;
    int             use_spinor;
    int             use_skel_mobius;    // unused in MVP

    sp_band_config_t k_bands;
    sp_band_config_t v_bands;

    int            *d_skeleton_idx;
    int            *d_residual_idx;
    int            *d_csr_offsets;
    int            *d_csr_skel_slot;
    int            *d_csr_mu_sign;   // int32 on GPU (converted from int8_t on init)

    int            *d_vilenkin_factors;
    int             n_factors;

    unsigned char  *d_k_cache;
    unsigned char  *d_v_cache;
    int             bytes_per_pos_k;
    int             bytes_per_pos_v;
    int             max_seq_len;
    int             n_slots;

    float          *d_pad_scratch;
    float          *d_coeff_scratch;
    float          *d_skel_scratch;
    float          *d_pred_scratch;
    float          *d_dev_scratch;
    unsigned char  *d_levels_scratch;
    float          *d_mag_scratch;
    void           *stream;
} sp_cuda_sqfree_cache_t;

int  sp_cuda_sqfree_cache_init(sp_cuda_sqfree_cache_t *cc,
                                const sp_config_t *cfg,
                                int max_seq_len,
                                int residual_bits,
                                int use_spinor,
                                void *stream);
void sp_cuda_sqfree_cache_free(sp_cuda_sqfree_cache_t *cc);

void sp_cuda_sqfree_write_k(sp_cuda_sqfree_cache_t *cc,
                             int layer, int head, int pos,
                             const float *d_k_vec);
void sp_cuda_sqfree_write_v(sp_cuda_sqfree_cache_t *cc,
                             int layer, int head, int pos,
                             const float *d_v_vec);
void sp_cuda_sqfree_read_k(const sp_cuda_sqfree_cache_t *cc,
                            int layer, int head, int pos,
                            float *d_k_out);
void sp_cuda_sqfree_read_v(const sp_cuda_sqfree_cache_t *cc,
                            int layer, int head, int pos,
                            float *d_v_out);

#ifdef __cplusplus
}
#endif

#endif // SHANNON_PRIME_CUDA_H

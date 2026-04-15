// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#ifndef SHANNON_PRIME_CORE_H
#define SHANNON_PRIME_CORE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

// Maximum supported head dimensions (must be power of 2 for WHT)
#define SP_MAX_HEAD_DIM     256
#define SP_MAX_BANDS        8
#define SP_MAX_LAYERS       128
#define SP_MAX_HEADS        128

// Vilenkin basis: product of first k primes
// N=210 = 2*3*5*7 covers hd <= 128 with zero-padding
#define SP_VILENKIN_N_2P    6    // 2*3 = 6
#define SP_VILENKIN_N_3P    30   // 2*3*5 = 30
#define SP_VILENKIN_N_4P    210  // 2*3*5*7 = 210

// Ship-safe bit allocation (NaN-safe on all platforms)
#define SP_DEFAULT_K_BANDS  4
#define SP_DEFAULT_K_BITS   { 5, 5, 4, 3 }
#define SP_DEFAULT_V_BANDS  1
#define SP_DEFAULT_V_BITS   { 3 }

// ============================================================================
// Configuration
// ============================================================================

typedef struct {
    int      head_dim;           // Model head dimension (64, 128, 256)
    int      n_layers;           // Number of transformer layers
    int      n_heads_kv;         // Number of KV heads (GQA-aware)

    // VHT2 banded quantization
    int      k_n_bands;          // Number of K spectral bands (default 4)
    int      k_band_bits[SP_MAX_BANDS]; // Bits per K band (default 5,5,4,3)
    int      v_n_bands;          // Number of V bands (default 1)
    int      v_band_bits[SP_MAX_BANDS]; // Bits per V band (default 3)

    // Möbius partition mask
    bool     use_mobius_mask;    // Reorder coefficients squarefree-first
    int      skeleton_k;         // WHT skeleton size for K (must == head_dim)
    int      skeleton_v;         // WHT skeleton size for V (must == head_dim)

    // Vilenkin successive decomposition (research path)
    bool     use_vilenkin;       // Enable Vilenkin instead of pure WHT
    int      vilenkin_primes;    // Number of primes (2=6, 3=30, 4=210)
    float    energy_threshold;   // Energy retention (0.95 = 95%)
} sp_config_t;

// Initialize config with ship-safe defaults
void sp_config_init(sp_config_t *cfg, int head_dim, int n_layers, int n_heads_kv);

// ============================================================================
// WHT (Walsh-Hadamard Transform) — Z/2Z basis
// ============================================================================
//
// The WHT is self-inverse: WHT(WHT(x)) = N*x
// For vectors of length N (power of 2), in-place butterfly.
// This is the Z/2Z special case of the Vilenkin-Hartley basis.

// In-place WHT on float vector of length n (must be power of 2).
// After transform, divide by n for normalized inverse.
void sp_wht_inplace_f32(float *data, int n);

// In-place WHT on fp16 vector (for backends with native fp16).
// Provided as reference; backends may override with SIMD.
void sp_wht_inplace_f16(uint16_t *data, int n);

// ============================================================================
// Möbius mask — squarefree-first coefficient ordering
// ============================================================================
//
// μ(n) != 0 iff n is squarefree (no repeated prime factors).
// 61.4% of indices in N=210 are squarefree.
// Prioritizing squarefree indices during coefficient extraction
// improves quality by 0.14 PPL at identical coefficient budget.
//
// The mask is a property of the WHT spectrum, not the model.
// Cross-platform invariant: K correlation 0.997 on both hd=128 and hd=64.

typedef struct {
    int      n;                  // Dimension (head_dim)
    int     *order;              // Permutation: order[i] = original index
    int      n_squarefree;       // Count of squarefree indices
    int8_t  *mu;                 // Möbius function values μ(0..n-1)
} sp_mobius_mask_t;

// Build the Möbius mask for dimension n.
// Caller must free with sp_mobius_mask_free().
int sp_mobius_mask_init(sp_mobius_mask_t *mask, int n);
void sp_mobius_mask_free(sp_mobius_mask_t *mask);

// Apply Möbius reordering to WHT coefficients (in-place).
// Squarefree coefficients move to front; non-squarefree to back.
void sp_mobius_reorder(float *wht_coeffs, const sp_mobius_mask_t *mask);

// Inverse reorder (restore original index order after dequantization).
void sp_mobius_unreorder(float *wht_coeffs, const sp_mobius_mask_t *mask);

// Caller-owned-scratch variants. `scratch` must be at least mask->n floats
// and is writable. Used on the hot path to avoid malloc per KV vector.
void sp_mobius_reorder_ex(float *wht_coeffs, const sp_mobius_mask_t *mask,
                          float *scratch);
void sp_mobius_unreorder_ex(float *wht_coeffs, const sp_mobius_mask_t *mask,
                            float *scratch);

// ============================================================================
// VHT2 banded quantization
// ============================================================================
//
// After WHT + optional Möbius reorder:
//   1. Split N coefficients into k equal bands
//   2. Each band gets its own fp16 scale + packed integer values
//   3. Band allocation mirrors WHT energy decay:
//      high-energy bands get more bits, low-energy tail gets fewer
//
// Key findings from paper:
//   - 5/5/4/3 BEATS lossless fp16 by 0.04% (spectral regularization)
//   - 4/4/4/4 is off the Pareto frontier (4/4/4/3 is strictly better)
//   - 3-bit floor: 2-bit on any band is catastrophic
//   - Flat beats banded for V vectors (no exceptions)

typedef struct {
    int      n_bands;
    int      band_bits[SP_MAX_BANDS];
    int      band_size;          // coefficients per band (= head_dim / n_bands)
    int      total_bytes;        // compressed size per vector
} sp_band_config_t;

// Compute band configuration from config.
void sp_band_config_init(sp_band_config_t *bc, int head_dim,
                         int n_bands, const int *band_bits);

// Quantize WHT coefficients into banded format.
// wht_coeffs: input (head_dim floats, already WHT'd and optionally reordered)
// out:        output buffer (bc->total_bytes)
void sp_band_quantize(const float *wht_coeffs, uint8_t *out,
                      const sp_band_config_t *bc);

// Dequantize banded format back to WHT coefficients.
// in:         compressed buffer (bc->total_bytes)
// wht_coeffs: output (head_dim floats)
void sp_band_dequantize(const uint8_t *in, float *wht_coeffs,
                        const sp_band_config_t *bc);

// ============================================================================
// Vilenkin-Hartley Transform — multi-prime basis (research path)
// ============================================================================
//
// Generalizes WHT from Z/2Z to Z/p1Z × Z/p2Z × ... × Z/pkZ.
// Hartley kernel: cas(x) = cos(x) + sin(x)
// Self-inverse for ALL primes: V·V = N·I
// Round-trip error = 0.0000
//
// Progressive prime expansion monotonically increases correlation:
//   Walsh (Z/2Z):              0.9490
//   Z/2Z × Z/3Z:              0.9493
//   Z/2Z × Z/3Z × Z/5Z:      0.9500
//   Z/2Z × Z/3Z × Z/5Z × Z/7Z: 0.9513

typedef struct {
    int      n;                  // Basis dimension (product of primes)
    int      n_primes;           // Number of primes in factorization
    int      primes[8];          // The primes
    float   *basis;              // n×n Vilenkin-Hartley matrix (row-major)
} sp_vilenkin_basis_t;

// Initialize Vilenkin basis for given prime count.
// n_primes=2 → N=6, n_primes=3 → N=30, n_primes=4 → N=210
int sp_vilenkin_init(sp_vilenkin_basis_t *vb, int n_primes);
void sp_vilenkin_free(sp_vilenkin_basis_t *vb);

// Forward transform: project head_dim vector into Vilenkin space.
// Input is zero-padded from head_dim to vb->n if needed.
void sp_vilenkin_forward(const sp_vilenkin_basis_t *vb,
                         const float *input, int head_dim,
                         float *output);

// Inverse transform: reconstruct from Vilenkin coefficients.
// Output is truncated from vb->n back to head_dim.
void sp_vilenkin_inverse(const sp_vilenkin_basis_t *vb,
                         const float *input,
                         float *output, int head_dim);

// Three-pass successive extraction (Z/3Z skeleton, Z/5Z detail, Z/7Z texture)
typedef struct {
    int      n_coeffs;           // Number of retained coefficients
    int     *indices;            // Which Vilenkin indices are retained
    float   *values;             // Coefficient values
} sp_vilenkin_pass_t;

// Extract pass k from (residual) signal. Modifies residual in-place.
int sp_vilenkin_extract_pass(const sp_vilenkin_basis_t *vb,
                             float *residual, int head_dim,
                             float energy_threshold,
                             sp_vilenkin_pass_t *pass);

void sp_vilenkin_pass_free(sp_vilenkin_pass_t *pass);

// ============================================================================
// Shadow cache — the integration point
// ============================================================================
//
// The shadow cache intercepts KV writes, compresses via VHT2,
// and reconstructs on read. This is the interface backends implement.
//
// Architecture:
//   Write path: raw KV → WHT → Möbius reorder → band quantize → store
//   Read path:  load → band dequantize → Möbius unreorder → inverse WHT → KV

typedef struct {
    sp_config_t         config;
    sp_band_config_t    k_bands;
    sp_band_config_t    v_bands;
    sp_mobius_mask_t     mobius_mask;

    // Compressed storage (allocated by backend)
    uint8_t           **k_cache;     // [layer * n_heads + head][seq_pos] → compressed
    uint8_t           **v_cache;     // same layout
    int                *seq_len;     // per-layer current sequence length

    // Scratch buffers (callee-owned; callers must serialize across threads).
    // wht_scratch      primary WHT working buffer for write path
    // mobius_scratch   scratch for Möbius reorder tmp (hot path, avoids malloc)
    // read_scratch     WHT buffer for read path (avoids malloc per read)
    float              *wht_scratch;    // head_dim floats
    float              *mobius_scratch; // head_dim floats
    float              *read_scratch;   // head_dim floats
} sp_shadow_cache_t;

// Initialize shadow cache. Backend allocates compressed storage.
int sp_shadow_cache_init(sp_shadow_cache_t *sc, const sp_config_t *cfg);
void sp_shadow_cache_free(sp_shadow_cache_t *sc);

// Compress and store a K vector.
// layer, head: identifies the cache slot
// pos: sequence position
// k_vec: raw K vector (head_dim floats, already RoPE'd)
void sp_shadow_write_k(sp_shadow_cache_t *sc,
                       int layer, int head, int pos,
                       const float *k_vec);

// Compress and store a V vector.
void sp_shadow_write_v(sp_shadow_cache_t *sc,
                       int layer, int head, int pos,
                       const float *v_vec);

// Reconstruct a K vector from compressed storage.
// k_out: output buffer (head_dim floats)
void sp_shadow_read_k(const sp_shadow_cache_t *sc,
                      int layer, int head, int pos,
                      float *k_out);

// Reconstruct a V vector from compressed storage.
void sp_shadow_read_v(const sp_shadow_cache_t *sc,
                      int layer, int head, int pos,
                      float *v_out);

// Batch variants — process n_pos contiguous vectors with a single setup.
// k_vecs / v_vecs must be contiguous [n_pos × head_dim] arrays.
// These are the real batches: they run in a tight loop over the persistent
// scratch buffers (no per-vector malloc) so they amortize pipeline cost
// across the batch. start_pos + n_pos must fit within the cache's max_seq.
void sp_shadow_write_k_batch(sp_shadow_cache_t *sc,
                             int layer, int head,
                             int start_pos, int n_pos,
                             const float *k_vecs);
void sp_shadow_write_v_batch(sp_shadow_cache_t *sc,
                             int layer, int head,
                             int start_pos, int n_pos,
                             const float *v_vecs);
void sp_shadow_read_k_batch (const sp_shadow_cache_t *sc,
                             int layer, int head,
                             int start_pos, int n_pos,
                             float *k_out);
void sp_shadow_read_v_batch (const sp_shadow_cache_t *sc,
                             int layer, int head,
                             int start_pos, int n_pos,
                             float *v_out);

// ============================================================================
// NaN guard — clamp reconstructed values to safe range
// ============================================================================
//
// Aggressive compression can produce values at FP boundary after
// softmax over long context. Ship config (5/4/4) is NaN-safe,
// but the guard is defense-in-depth.

void sp_nan_guard_f32(float *data, int n, float max_magnitude);

// ============================================================================
// Diagnostics
// ============================================================================

// Compute correlation between original and reconstructed vector.
float sp_correlation_f32(const float *a, const float *b, int n);

// Compute compression ratio for current config.
float sp_compression_ratio(const sp_config_t *cfg);

// Print config summary to stderr.
void sp_config_print(const sp_config_t *cfg);

#ifdef __cplusplus
}
#endif

#endif // SHANNON_PRIME_CORE_H

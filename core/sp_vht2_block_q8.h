/* sp_vht2_block_q8.h — Strike 2: Band-as-Block scale fusion.
 *
 * Unifies Substrate A (VHT2 / Möbius / banded quant) with Substrate B
 * (sp_ok_q8_block Frobenius packing) into one storage object:
 *
 *   The Spectrally-Banded Cyclotomic Lattice.
 *
 * For a fp32 input vector of length head_dim:
 *
 *   1. VHT2 forward     → orthonormal spectral domain (energy concentrated)
 *   2. Möbius reorder   → squarefree indices to the front (high-info first)
 *   3. Split into bands → contiguous groups of 32 elements (1 block = 1 band-tile)
 *   4. Per-tile abs-max → int8 codepoints in [-127, +127]
 *   5. Fuse band-scale × π^k → (B_a, B_b) in O_K = Z[ω]
 *   6. Pack into sp_ok_q8_block_t (same struct the existing block-q8 matmul
 *      already understands — no new kernel needed for the inner integer ops)
 *
 * Math invariant: VHT2 is orthonormal, so <a, b> = <VHT2(a), VHT2(b)>.
 * The dot product computed from encoded blocks (sum across bands of the
 * tile-wise integer dot products, then multiplied by per-tile scales)
 * equals the original fp32 dot product up to int8 quantization error.
 *
 * Geometry contract: head_dim must be a multiple of (n_bands * 32).
 * Production cases:
 *   head_dim=128, n_bands=4  →  4 blocks (1 block per band)
 *   head_dim=256, n_bands=4  →  8 blocks (2 blocks per band)
 *   head_dim=64,  n_bands=2  →  2 blocks (1 block per band)
 *
 * Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
 */

#ifndef SP_VHT2_BLOCK_Q8_H
#define SP_VHT2_BLOCK_Q8_H

#include "sp_ok_arith.h"
#include "sp_ok_block_quant.h"  /* sp_ok_q8_block_t, SP_OK_BLOCK_SIZE */
#include "shannon_prime.h"      /* sp_mobius_mask_t */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Encoder context — holds the VHT2 + Möbius preprocessing parameters and
 * the pre-computed Frobenius π^k coordinates. Construct once per (head_dim,
 * n_bands, p, k) configuration; reuse across many encode calls.
 *
 * `mobius` may be NULL → skip Möbius reorder (test-only).
 */
typedef struct {
    int                     head_dim;
    int                     n_bands;
    int                     band_size;      /* head_dim / n_bands (multiple of 32) */
    int                     blocks_per_band; /* band_size / 32 */
    int                     total_blocks;   /* n_bands * blocks_per_band */
    const sp_mobius_mask_t* mobius;         /* may be NULL */
    int64_t                 scale_recip;    /* Q-format multiplier for B_a/B_b */
    sp_ok_t                 pi_pow;         /* (π_a, π_b) = π^k */
} sp_vht2_q8_ctx;

/* Initialize encoder context. Computes π^k from (p, k) once.
 * Returns 1 on success, 0 if geometry is invalid or π^k is unreachable
 * for the given prime. The mobius mask is borrowed (not owned). */
int sp_vht2_q8_ctx_init(sp_vht2_q8_ctx*         ctx,
                        int                     head_dim,
                        int                     n_bands,
                        const sp_mobius_mask_t* mobius,
                        int64_t                 scale_recip,
                        int64_t                 p,
                        int64_t                 k);

/* Encode one head_dim-sized fp32 vector into ctx->total_blocks blocks.
 * dst_blocks must have space for ctx->total_blocks entries.
 *
 * scratch must point to head_dim floats — used for VHT2 + Möbius staging.
 * Caller-owned to allow tight inner loops over many vectors.
 *
 * Returns 1 on success. */
int sp_vht2_q8_encode(sp_ok_q8_block_t*       dst_blocks,
                      const float*            src_fp32,
                      const sp_vht2_q8_ctx*   ctx,
                      float*                  scratch);

/* Decode the encoded blocks back to a fp32 vector. Lossy by the int8
 * quantization error per band. Used for parity checks and the inverse
 * path in attention dequant. */
int sp_vht2_q8_decode(float*                  dst_fp32,
                      const sp_ok_q8_block_t* src_blocks,
                      const sp_vht2_q8_ctx*   ctx,
                      float*                  scratch);

/* Spectral-domain dot product: compute <a, b> directly from encoded
 * blocks, without dequant/inverse-VHT2 round-trip. Exploits VHT2
 * orthonormality: <a, b> = sum over bands of <a_band, b_band> in the
 * spectral domain.
 *
 * max_bands controls progressive read:
 *   0 or ctx->n_bands → use every band (full inner product)
 *   1                  → System 1 fast path (Band 0 only, ~30% energy)
 *   2                  → Bands 0-1 (~86% energy)
 *
 * The integer per-tile dot product is computed in int32 (32 int8x int8
 * accumulate fits in int23), then scaled by the tile's (B_a, B_b) via
 * the decoded fp32 band-scale. */
float sp_vht2_q8_dot(const sp_ok_q8_block_t* a_blocks,
                     const sp_ok_q8_block_t* b_blocks,
                     const sp_vht2_q8_ctx*   ctx,
                     int                     max_bands);

/* ─── Strike 3: System 1 prefilter ─────────────────────────────────── */

/* Per-band L2-squared energy of an encoded vector. band_energy_out must
 * point to at least ctx->n_bands floats. The values are computed from
 * the stored Frobenius (B_a, B_b) per-tile scales and the int8 codepoints:
 *     E[band] = sum over tiles t in band of (tile_scale^2 · sum_i packed[i]^2)
 *
 * By VHT2 orthonormality, sum E[band] across all bands equals the L2^2
 * norm of the original (un-encoded) input vector, modulo int8 quant noise.
 *
 * Cost: O(head_dim) per call. Designed to be hoisted out of the K-cache
 * write path — compute once when a K vector is encoded and cache the
 * result, then read at attention time without recomputing. */
void sp_vht2_q8_band_energy(const sp_ok_q8_block_t* blocks,
                            const sp_vht2_q8_ctx*   ctx,
                            float*                  band_energy_out);

/* Gate function: returns the minimum number of (leading) bands needed to
 * capture `energy_fraction` of the vector's total energy. Walks bands
 * in natural order (Möbius has already reordered them so squarefree /
 * high-info indices live in Band 0).
 *
 * Used at attention time:
 *   int k = sp_vht2_q8_min_bands_for_energy(k_blocks, ctx, 0.86f);
 *   float score = sp_vht2_q8_dot(q_blocks, k_blocks, ctx, k);
 *
 * Returns a value in [1, ctx->n_bands]. Returns ctx->n_bands if the
 * vector has zero or unreachable energy (escalate to full reconstruction).
 *
 * Typical thresholds for System 1 / System 2 routing:
 *   0.30  →  Band 0 alone (fast path, may underestimate by ~70%)
 *   0.86  →  Bands 0-1 typically (production default — 86% energy)
 *   0.99  →  Full reconstruction (escalation only when nothing dominates)
 */
int sp_vht2_q8_min_bands_for_energy(const sp_ok_q8_block_t* blocks,
                                    const sp_vht2_q8_ctx*   ctx,
                                    float                   energy_fraction);

#ifdef __cplusplus
}
#endif

#endif /* SP_VHT2_BLOCK_Q8_H */

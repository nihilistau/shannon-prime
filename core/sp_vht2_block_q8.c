/* sp_vht2_block_q8.c — Strike 2 implementation.
 *
 * Fuses VHT2 + Möbius + banded quant + Frobenius (B_a, B_b) scale folding
 * into one storage object: sp_ok_q8_block_t tiles, one per 32-element
 * spectral sub-band.
 *
 * Encode pipeline (per input vector of length head_dim):
 *
 *   fp32 in ──► sp_vht2_forward_f32 ──► sp_mobius_reorder_ex ──┐
 *                                                              │
 *   ┌──────────────────────────────────────────────────────────┘
 *   ▼
 *   for each band b:
 *     for each 32-element tile t within band b:
 *       amax  = max |x[i]|       over tile
 *       scale = amax / 127       (zero-clamped)
 *       q[i]  = round(x[i] / scale)   in [-127, +127]
 *       B_a   = round(scale_recip · scale · π_a^k)
 *       B_b   = round(scale_recip · scale · π_b^k)
 *       write block t with packed int8 + (B_a, B_b)
 *
 * Decode is the algebraic inverse: reconstruct band-tile fp32 from
 * int8 × scale, undo Möbius, undo VHT2.
 *
 * Spectral dot: <a, b> = sum over tiles t of
 *   (sum_{i in tile} a_q[i] * b_q[i]) * a_scale[t] * b_scale[t]
 * Each tile sum is int32 (32 × int8^2 < 2^23). The per-tile scale comes
 * from B_a / (scale_recip · π_a) for the a-coordinate; we store enough
 * to recover it without inverting Frobenius at runtime (see decode).
 */

#include "sp_vht2_block_q8.h"
#include "sp_frobenius.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Pull just the prototypes we need from shannon_prime.h (already
 * included transitively, but be explicit for the linker symbols). */
extern void sp_vht2_forward_f32(float *data, int n);
extern void sp_mobius_reorder_ex  (float *vht2_coeffs,
                                   const sp_mobius_mask_t *mask, float *scratch);
extern void sp_mobius_unreorder_ex(float *vht2_coeffs,
                                   const sp_mobius_mask_t *mask, float *scratch);

/* Frobenius helpers (same logic as sp_ok_block_quant.c — kept local
 * so this module doesn't depend on a private header from there). */
static int sp_vbq_compute_pi_pow(int64_t p, int64_t k, sp_ok_t* out) {
    if (k == 0) { out->a = 1; out->b = 0; return 1; }
    if (sp_is_inert(p)) {
        int64_t m = k / 2;
        int64_t s = 1;
        for (int64_t i = 0; i < m; ++i) s *= (-p);
        out->a = s; out->b = 0;
        return 1;
    } else if (sp_is_split(p)) {
        sp_ok_t pi;
        if (!sp_find_element_of_norm(p, &pi)) return 0;
        *out = sp_ok_pow(pi, k);
        return 1;
    }
    return 0;
}

static inline int64_t sp_vbq_rint_i64(double v) { return (int64_t)llrint(v); }

/* ─── Context init ─────────────────────────────────────────────────── */

int sp_vht2_q8_ctx_init(sp_vht2_q8_ctx*         ctx,
                        int                     head_dim,
                        int                     n_bands,
                        const sp_mobius_mask_t* mobius,
                        int64_t                 scale_recip,
                        int64_t                 p,
                        int64_t                 k)
{
    if (!ctx) return 0;
    if (head_dim <= 0 || n_bands <= 0) return 0;
    if (head_dim % n_bands != 0) return 0;
    int band_size = head_dim / n_bands;
    if (band_size % SP_OK_BLOCK_SIZE != 0) return 0;  /* tile alignment */

    sp_ok_t pi_pow;
    if (!sp_vbq_compute_pi_pow(p, k, &pi_pow)) return 0;

    ctx->head_dim        = head_dim;
    ctx->n_bands         = n_bands;
    ctx->band_size       = band_size;
    ctx->blocks_per_band = band_size / SP_OK_BLOCK_SIZE;
    ctx->total_blocks    = n_bands * ctx->blocks_per_band;
    ctx->mobius          = mobius;  /* may be NULL */
    ctx->scale_recip     = scale_recip;
    ctx->pi_pow          = pi_pow;
    return 1;
}

/* ─── Encode ───────────────────────────────────────────────────────── */
/* Process one head_dim vector: VHT2 → Möbius → per-tile q8 + Frobenius. */

int sp_vht2_q8_encode(sp_ok_q8_block_t*       dst_blocks,
                      const float*            src_fp32,
                      const sp_vht2_q8_ctx*   ctx,
                      float*                  scratch)
{
    if (!dst_blocks || !src_fp32 || !ctx || !scratch) return 0;

    /* Stage into scratch, run forward transforms. */
    memcpy(scratch, src_fp32, (size_t)ctx->head_dim * sizeof(float));
    sp_vht2_forward_f32(scratch, ctx->head_dim);
    if (ctx->mobius) {
        /* sp_mobius_reorder_ex needs an extra float[head_dim] scratch.
         * Stash it on the stack — head_dim is bounded by typical model
         * dims (≤ 4096), so a few KB is fine.
         *
         * If callers ship larger head_dims, they should pre-pass a
         * 2*head_dim scratch and we'll partition; for now, stack-alloc
         * the small overflow. */
        float mscratch[8192];
        if (ctx->head_dim > 8192) return 0;
        sp_mobius_reorder_ex(scratch, ctx->mobius, mscratch);
    }

    const double pi_a_d = (double)ctx->pi_pow.a;
    const double pi_b_d = (double)ctx->pi_pow.b;
    const double S      = (double)ctx->scale_recip;

    int blk = 0;
    for (int b = 0; b < ctx->n_bands; ++b) {
        int band_off = b * ctx->band_size;
        for (int t = 0; t < ctx->blocks_per_band; ++t) {
            const float* tile = scratch + band_off + t * SP_OK_BLOCK_SIZE;

            /* Per-tile abs-max. */
            float amax = 0.0f;
            for (int i = 0; i < SP_OK_BLOCK_SIZE; ++i) {
                float a = fabsf(tile[i]);
                if (a > amax) amax = a;
            }

            /* int8 maps to [-127, +127]; scale = amax / 127 */
            float tile_scale = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
            float inv_scale  = (tile_scale > 0.0f) ? (1.0f / tile_scale) : 0.0f;

            sp_ok_q8_block_t* dst = dst_blocks + blk;

            /* Fuse tile_scale into (B_a, B_b) via Frobenius. */
            dst->B_a = sp_vbq_rint_i64(S * (double)tile_scale * pi_a_d);
            dst->B_b = sp_vbq_rint_i64(S * (double)tile_scale * pi_b_d);
            dst->reserved_block_min_a = 0;
            dst->reserved_block_min_b = 0;

            /* Quantize codepoints. */
            for (int i = 0; i < SP_OK_BLOCK_SIZE; ++i) {
                int q = (int)lrintf(tile[i] * inv_scale);
                if (q >  127) q =  127;
                if (q < -127) q = -127;
                dst->packed[i] = (int8_t)q;
            }
            ++blk;
        }
    }
    return 1;
}

/* ─── Decode (parity / inverse path) ───────────────────────────────── */
/* Reconstruct fp32 vector. The per-tile scale is recovered from B_a:
 *   scale = B_a / (scale_recip · π_a)
 * If π_a is zero (degenerate Frobenius — k=0 inert case is handled),
 * we fall back to π_b. This is exact algebra; quantization error
 * enters only at the codepoint × scale multiply. */

int sp_vht2_q8_decode(float*                  dst_fp32,
                      const sp_ok_q8_block_t* src_blocks,
                      const sp_vht2_q8_ctx*   ctx,
                      float*                  scratch)
{
    if (!dst_fp32 || !src_blocks || !ctx || !scratch) return 0;

    const double S      = (double)ctx->scale_recip;
    const double pi_a_d = (double)ctx->pi_pow.a;
    const double pi_b_d = (double)ctx->pi_pow.b;
    const int use_a = (pi_a_d != 0.0);

    int blk = 0;
    for (int b = 0; b < ctx->n_bands; ++b) {
        int band_off = b * ctx->band_size;
        for (int t = 0; t < ctx->blocks_per_band; ++t) {
            const sp_ok_q8_block_t* src = src_blocks + blk;

            double tile_scale_d;
            if (use_a) {
                tile_scale_d = (double)src->B_a / (S * pi_a_d);
            } else if (pi_b_d != 0.0) {
                tile_scale_d = (double)src->B_b / (S * pi_b_d);
            } else {
                tile_scale_d = 0.0;  /* π = 0 → degenerate; emit zeros */
            }
            float tile_scale = (float)tile_scale_d;

            float* dst_tile = scratch + band_off + t * SP_OK_BLOCK_SIZE;
            for (int i = 0; i < SP_OK_BLOCK_SIZE; ++i) {
                dst_tile[i] = (float)src->packed[i] * tile_scale;
            }
            ++blk;
        }
    }

    /* Inverse: undo Möbius then VHT2 (self-inverse, so forward again). */
    if (ctx->mobius) {
        float mscratch[8192];
        if (ctx->head_dim > 8192) return 0;
        sp_mobius_unreorder_ex(scratch, ctx->mobius, mscratch);
    }
    sp_vht2_forward_f32(scratch, ctx->head_dim);  /* VHT2 self-inverse */
    memcpy(dst_fp32, scratch, (size_t)ctx->head_dim * sizeof(float));
    return 1;
}

/* ─── Spectral-domain dot product ──────────────────────────────────── */
/* <a, b> = sum_t (int_dot_t * scale_a_t * scale_b_t)
 * where int_dot_t = sum_i a.packed[i] * b.packed[i]  (int32 accumulator)
 *
 * Orthonormality of VHT2 makes this equal to the original fp32 dot of
 * pre-transform vectors, up to int8 quantization error. The Möbius
 * permutation is the same on both sides (since we use the same ctx),
 * so it cancels out — no inverse needed for inner product.
 *
 * Frobenius / O_K observation: we collapse the (B_a, B_b) per-tile
 * coefficients back to fp32 scales for the float accumulation. A
 * pure-integer dot would require computing the dot in the O_K ring
 * (sum of products is itself an O_K element, then norm-to-real) — that's
 * the form the HVX kernel will use, but for the parity test against the
 * fp32 reference we evaluate the float result directly. */

float sp_vht2_q8_dot(const sp_ok_q8_block_t* a_blocks,
                     const sp_ok_q8_block_t* b_blocks,
                     const sp_vht2_q8_ctx*   ctx,
                     int                     max_bands)
{
    if (!a_blocks || !b_blocks || !ctx) return 0.0f;
    int nb = (max_bands == 0) ? ctx->n_bands : max_bands;
    if (nb < 0) nb = 0;
    if (nb > ctx->n_bands) nb = ctx->n_bands;

    const double S      = (double)ctx->scale_recip;
    const double pi_a_d = (double)ctx->pi_pow.a;
    const double pi_b_d = (double)ctx->pi_pow.b;
    const int    use_a  = (pi_a_d != 0.0);

    double acc = 0.0;
    int blk = 0;
    for (int b = 0; b < nb; ++b) {
        for (int t = 0; t < ctx->blocks_per_band; ++t) {
            const sp_ok_q8_block_t* sa = a_blocks + blk;
            const sp_ok_q8_block_t* sb = b_blocks + blk;
            /* Integer dot of two int8 vectors fits comfortably in int32:
             *   32 × (127 × 127) < 32 × 16384 = 524288 < 2^20. */
            int32_t int_dot = 0;
            for (int i = 0; i < SP_OK_BLOCK_SIZE; ++i) {
                int_dot += (int32_t)sa->packed[i] * (int32_t)sb->packed[i];
            }
            /* Recover per-tile scales from Frobenius coordinates. */
            double a_scale, b_scale;
            if (use_a) {
                a_scale = (double)sa->B_a / (S * pi_a_d);
                b_scale = (double)sb->B_a / (S * pi_a_d);
            } else if (pi_b_d != 0.0) {
                a_scale = (double)sa->B_b / (S * pi_b_d);
                b_scale = (double)sb->B_b / (S * pi_b_d);
            } else {
                a_scale = b_scale = 0.0;
            }
            acc += (double)int_dot * a_scale * b_scale;
            ++blk;
        }
    }
    return (float)acc;
}

/* ─── Strike 3: System 1 prefilter primitives ──────────────────────── */
/* Per-band L2² energy. Cost O(head_dim) — hoist out of attention hot
 * path by caching the result alongside each K-vector at write time. */

void sp_vht2_q8_band_energy(const sp_ok_q8_block_t* blocks,
                            const sp_vht2_q8_ctx*   ctx,
                            float*                  band_energy_out)
{
    if (!blocks || !ctx || !band_energy_out) return;

    const double S      = (double)ctx->scale_recip;
    const double pi_a_d = (double)ctx->pi_pow.a;
    const double pi_b_d = (double)ctx->pi_pow.b;
    const int    use_a  = (pi_a_d != 0.0);

    int blk = 0;
    for (int b = 0; b < ctx->n_bands; ++b) {
        double e = 0.0;
        for (int t = 0; t < ctx->blocks_per_band; ++t) {
            const sp_ok_q8_block_t* src = blocks + blk;
            double tile_scale_d;
            if (use_a) {
                tile_scale_d = (double)src->B_a / (S * pi_a_d);
            } else if (pi_b_d != 0.0) {
                tile_scale_d = (double)src->B_b / (S * pi_b_d);
            } else {
                tile_scale_d = 0.0;
            }
            /* sum of squared int8 codepoints, fits comfortably in int32:
             *   32 × 127^2 = 516128 < 2^20. */
            int32_t sumsq = 0;
            for (int i = 0; i < SP_OK_BLOCK_SIZE; ++i) {
                int32_t v = (int32_t)src->packed[i];
                sumsq += v * v;
            }
            e += (double)sumsq * tile_scale_d * tile_scale_d;
            ++blk;
        }
        band_energy_out[b] = (float)e;
    }
}

int sp_vht2_q8_min_bands_for_energy(const sp_ok_q8_block_t* blocks,
                                    const sp_vht2_q8_ctx*   ctx,
                                    float                   energy_fraction)
{
    if (!blocks || !ctx) return 1;
    /* SP_MAX_BANDS in shannon_prime.h is 32; n_bands ≤ that. Stack-alloc. */
    float be[32];
    if (ctx->n_bands > 32) return ctx->n_bands;
    sp_vht2_q8_band_energy(blocks, ctx, be);

    double total = 0.0;
    for (int b = 0; b < ctx->n_bands; ++b) total += (double)be[b];
    if (total <= 0.0) return ctx->n_bands;  /* pathological — escalate */

    /* Clamp fraction to [0, 1]; > 1 always returns full. */
    double f = (double)energy_fraction;
    if (f < 0.0) f = 0.0;
    if (f > 1.0) return ctx->n_bands;

    double target = f * total;
    double acc = 0.0;
    for (int b = 0; b < ctx->n_bands; ++b) {
        acc += (double)be[b];
        if (acc >= target) return b + 1;
    }
    return ctx->n_bands;
}

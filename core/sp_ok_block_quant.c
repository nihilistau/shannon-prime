/* sp_ok_block_quant.c — Phase 15: GGUF Q8_0 / Q4_0 importers.
 *
 * For each GGUF block:
 *   - Read fp16 block_scale.
 *   - Compute fused integers
 *       B_a = round(scale_recip * block_scale * π_a^k)
 *       B_b = round(scale_recip * block_scale * π_b^k)
 *   - Copy the int8 / int4 codepoints byte-for-byte into the output block.
 *
 * π^k = (π_a, π_b) is computed once per tensor (not per block) using
 * sp_find_element_of_norm + sp_ok_pow. For inert primes (e.g., p=2 with
 * k even) π_b is 0 and the b-coordinate of every weight stays zero; for
 * split primes (p=41 the production case) both coordinates are non-zero.
 *
 * The fp16-to-fp32 conversion is inline here to keep this TU standalone
 * (no engine-side includes).
 */

#include "sp_ok_block_quant.h"
#include "sp_frobenius.h"

#include <math.h>
#include <string.h>

/* IEEE half -> float, matches the helper in src/sp_ok_encode.cpp. */
static inline float sp_blkq_fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15)) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            /* subnormal */
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float r;
    memcpy(&r, &f, sizeof(r));
    return r;
}

/* Compute π^k for the given prime, write (a, b) coords into *pi_pow.
 * Returns 1 on success, 0 on failure (prime not split / element not
 * found, k==0 returns (1, 0)). */
static int sp_blkq_compute_pi_pow_k(int64_t p, int64_t k, sp_ok_t* pi_pow) {
    if (k == 0) {
        pi_pow->a = 1;
        pi_pow->b = 0;
        return 1;
    }
    if (sp_is_inert(p)) {
        /* For inert primes, "π" is just p itself (the prime is inert in O_K).
         * The Frobenius shim treats this as a real scalar p^(k/2) when k even.
         * For symmetric/scaled Frobenius, model it as (p^(k/2), 0). */
        int64_t m = k / 2;
        int64_t scale = 1;
        for (int64_t i = 0; i < m; ++i) scale *= (-p);
        /* (-p)^m carries the sign convention used in sp_ok_encode.cpp. */
        pi_pow->a = scale;
        pi_pow->b = 0;
        return 1;
    } else if (sp_is_split(p)) {
        sp_ok_t pi;
        if (!sp_find_element_of_norm(p, &pi)) return 0;
        *pi_pow = sp_ok_pow(pi, k);
        return 1;
    }
    return 0;
}

/* Round-half-to-even fp64 -> int64. */
static inline int64_t sp_blkq_rint_i64(double v) {
    return (int64_t)llrint(v);
}

/* ============================================================================
 * Q8_0 importer
 * ============================================================================ */

int sp_ok_block_q8_from_gguf_q8_0(
    sp_ok_block_q8_tensor* dst,
    const sp_gguf_block_q8_0* src,
    size_t n_blocks,
    int64_t scale_recip,
    int64_t p,
    int64_t k)
{
    if (!dst || !src || !dst->blocks) return 0;
    if (dst->n_blocks != n_blocks) return 0;
    if (dst->numel != n_blocks * SP_OK_BLOCK_SIZE) return 0;

    sp_ok_t pi_pow;
    if (!sp_blkq_compute_pi_pow_k(p, k, &pi_pow)) return 0;

    const double pi_a_d = (double)pi_pow.a;
    const double pi_b_d = (double)pi_pow.b;
    const double S      = (double)scale_recip;

    for (size_t b = 0; b < n_blocks; ++b) {
        const sp_gguf_block_q8_0* gsrc = src + b;
        sp_ok_q8_block_t*         gdst = dst->blocks + b;

        const float block_scale_f = sp_blkq_fp16_to_fp32(gsrc->d);
        const double bs           = (double)block_scale_f;

        /* B_a = round(scale_recip * block_scale * π_a^k) */
        gdst->B_a = sp_blkq_rint_i64(S * bs * pi_a_d);
        gdst->B_b = sp_blkq_rint_i64(S * bs * pi_b_d);
        gdst->reserved_block_min_a = 0;
        gdst->reserved_block_min_b = 0;

        /* Copy the 32 int8 codepoints byte-for-byte. */
        memcpy(gdst->packed, gsrc->qs, SP_OK_BLOCK_SIZE);
    }

    dst->frobenius_p = (int16_t)p;
    dst->frobenius_k = (int16_t)k;
    dst->reserved    = 0;
    return 1;
}

/* ============================================================================
 * Q4_0 importer
 * ============================================================================ */

int sp_ok_block_q4_from_gguf_q4_0(
    sp_ok_block_q4_tensor* dst,
    const sp_gguf_block_q4_0* src,
    size_t n_blocks,
    int64_t scale_recip,
    int64_t p,
    int64_t k)
{
    if (!dst || !src || !dst->blocks) return 0;
    if (dst->n_blocks != n_blocks) return 0;
    if (dst->numel != n_blocks * SP_OK_BLOCK_SIZE) return 0;

    sp_ok_t pi_pow;
    if (!sp_blkq_compute_pi_pow_k(p, k, &pi_pow)) return 0;

    const double pi_a_d = (double)pi_pow.a;
    const double pi_b_d = (double)pi_pow.b;
    const double S      = (double)scale_recip;

    for (size_t b = 0; b < n_blocks; ++b) {
        const sp_gguf_block_q4_0* gsrc = src + b;
        sp_ok_q4_block_t*         gdst = dst->blocks + b;

        const float block_scale_f = sp_blkq_fp16_to_fp32(gsrc->d);
        const double bs           = (double)block_scale_f;

        gdst->B_a = sp_blkq_rint_i64(S * bs * pi_a_d);
        gdst->B_b = sp_blkq_rint_i64(S * bs * pi_b_d);

        /* Copy the 16 packed-nybble bytes. Note: GGUF Q4_0 stores values
         * with a +8 bias; the bias is removed at decode-time via
         * sp_ok_block_q4_decode_codepoint, not here. */
        memcpy(gdst->packed, gsrc->qs, SP_OK_BLOCK_SIZE / 2);
    }

    dst->frobenius_p = (int16_t)p;
    dst->frobenius_k = (int16_t)k;
    dst->reserved    = 0;
    return 1;
}

/* ============================================================================
 * Q4_1 importer
 * ============================================================================
 *
 * GGUF Q4_1: W[k] = d * x_int[k] + m, where x_int[k] is UNSIGNED in
 * [0, 15] (no +8 bias). Fuses (d, m) with Frobenius π^k:
 *   B_a = round(S · d · π_a),  B_b = round(S · d · π_b)
 *   M_a = round(S · m · π_a),  M_b = round(S · m · π_b) */

int sp_ok_block_q4_1_from_gguf_q4_1(
    sp_ok_block_q4_1_tensor* dst,
    const sp_gguf_block_q4_1* src,
    size_t n_blocks,
    int64_t scale_recip,
    int64_t p,
    int64_t k)
{
    if (!dst || !src || !dst->blocks) return 0;
    if (dst->n_blocks != n_blocks) return 0;
    if (dst->numel != n_blocks * SP_OK_BLOCK_SIZE) return 0;

    sp_ok_t pi_pow;
    if (!sp_blkq_compute_pi_pow_k(p, k, &pi_pow)) return 0;

    const double pi_a_d = (double)pi_pow.a;
    const double pi_b_d = (double)pi_pow.b;
    const double S      = (double)scale_recip;

    for (size_t b = 0; b < n_blocks; ++b) {
        const sp_gguf_block_q4_1* gsrc = src + b;
        sp_ok_q4_1_block_t*       gdst = dst->blocks + b;

        const float bd_f = sp_blkq_fp16_to_fp32(gsrc->d);
        const float bm_f = sp_blkq_fp16_to_fp32(gsrc->m);
        const double bd  = (double)bd_f;
        const double bm  = (double)bm_f;

        gdst->B_a = sp_blkq_rint_i64(S * bd * pi_a_d);
        gdst->B_b = sp_blkq_rint_i64(S * bd * pi_b_d);
        gdst->M_a = sp_blkq_rint_i64(S * bm * pi_a_d);
        gdst->M_b = sp_blkq_rint_i64(S * bm * pi_b_d);

        memcpy(gdst->packed, gsrc->qs, SP_OK_BLOCK_SIZE / 2);
    }

    dst->frobenius_p = (int16_t)p;
    dst->frobenius_k = (int16_t)k;
    dst->reserved    = 0;
    return 1;
}

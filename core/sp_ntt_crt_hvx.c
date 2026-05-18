/* sp_ntt_crt_hvx.c — Hexagon V69 HVX kernel for dual-Proth CRT NTT.
 *
 * Strike 1 (Path A) skeleton. Mirrors the AVX-512 Barrett path from
 * sp_ntt_crt.c, expressed against HVX 1024-bit vectors (32 × uint32).
 * The 30-bit Proth residues fit natively in 32-bit lanes; Barrett
 * intermediates use HVX_VectorPair (32 × uint64 = 2048 bits).
 *
 * Build matrix:
 *   SP_HEXAGON_ENABLED  defined → real Q6_* intrinsics from hexagon_protos.h
 *                                 (compiled by hexagon-clang from Hexagon SDK 5.x+)
 *   SP_HEXAGON_ENABLED  undef   → scalar reference using uint32 lane math,
 *                                 produces bit-identical output to the
 *                                 AVX-512 path (Barrett sequence is the
 *                                 same six-op pattern).
 *
 * Numerical invariant: every value returned through this file is exactly
 * the same as sp_ntt_crt_forward / sp_ntt_crt_inverse / sp_ntt_crt_pointwise_mul
 * for the same input. Verified by test_sp_ntt_crt_hvx.cpp (forward-HVX
 * → inverse-AVX-512 round-trip must equal forward-AVX-512 → inverse-AVX-512
 * for the same int64 input).
 *
 * Twiddle tables: read from the uint32 mirrors in sp_ntt_crt_consts.h
 * (Phase HVX-1 addition — see scripts/gen_ntt_crt_consts.py). The
 * uint64 tables remain for the AVX-512 path; both are generated from
 * the same Python source-of-truth.
 *
 * Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
 */

#include "sp_ntt_crt_hvx.h"
#include "sp_ntt_crt.h"
#include "sp_ntt_crt_consts.h"

#include <stdint.h>
#include <string.h>

/* Portable 128-byte alignment for HVX vector loads. MSVC ignores
 * __attribute__((aligned)) with a warning; gcc/clang/hexagon-clang
 * honor it. We only need true alignment on the HVX path — the scalar
 * fallback is byte-addressable. */
#if defined(__GNUC__) || defined(__clang__)
#  define SP_HVX_ALIGNED __attribute__((aligned(128)))
#else
#  define SP_HVX_ALIGNED  /* MSVC: alignment unenforced; scalar path only */
#endif

/* ─── HVX intrinsic abstraction layer ──────────────────────────────────
 *
 * sp_hvx_* macros/functions present a uniform interface to the kernel
 * logic below. Under SP_HEXAGON_ENABLED they expand to real Q6_*
 * intrinsics; otherwise they expand to scalar loops over the 32 lanes
 * of a fixed-size buffer. This lets the algorithm read identically on
 * both targets and lets the parity test fire on x86.
 *
 * Lane width: 32 × uint32 per HVX vector. We use VectorPair to hold
 * 32 × uint64 of Barrett intermediates (low half + high half).
 */

#define SP_HVX_LANES 32  /* uint32 lanes per 1024-bit HVX vector */

#ifdef SP_HEXAGON_ENABLED

#include <hexagon_protos.h>
#include <hexagon_types.h>

typedef HVX_Vector       sp_hvx_v32;   /* 32 × uint32 */
typedef HVX_VectorPair   sp_hvx_v64;   /* 32 × uint64 (lo,hi 32-bit pair) */

#define SP_HVX_LOAD_U32(ptr)       (*(const HVX_Vector*)(ptr))
#define SP_HVX_STORE_U32(ptr, v)   (*(HVX_Vector*)(ptr) = (v))

/* HVX V69 unsigned 32x32 → 64 widening multiply, vector × vector.
 * Returns a VectorPair: low 32 bits of each product in one half, high
 * 32 bits in the other. Composed from halfword multiplies on V69 since
 * there is no direct 32x32 widening intrinsic on this arch. */
static inline sp_hvx_v64 sp_hvx_mul_u32_to_u64(sp_hvx_v32 a, sp_hvx_v32 b) {
    /* Decompose a = a_hi:16 | a_lo:16, b = b_hi:16 | b_lo:16.
     * (a_hi*b_hi)<<32 + (a_hi*b_lo + a_lo*b_hi)<<16 + a_lo*b_lo  */
    HVX_Vector       a_lo16   = Q6_V_vand_VV(a, Q6_V_vsplat_R(0x0000FFFF));
    HVX_Vector       a_hi16   = Q6_Vuw_vlsr_VuwR(a, 16);
    HVX_Vector       b_lo16   = Q6_V_vand_VV(b, Q6_V_vsplat_R(0x0000FFFF));
    HVX_Vector       b_hi16   = Q6_Vuw_vlsr_VuwR(b, 16);
    /* All four 16x16 → 32 products fit in one HVX_Vector each (32-bit lanes). */
    HVX_Vector p_lolo = Q6_Vuw_vmpyio_VuhVuh(a_lo16, b_lo16);
    HVX_Vector p_lohi = Q6_Vuw_vmpyio_VuhVuh(a_lo16, b_hi16);
    HVX_Vector p_hilo = Q6_Vuw_vmpyio_VuhVuh(a_hi16, b_lo16);
    HVX_Vector p_hihi = Q6_Vuw_vmpyio_VuhVuh(a_hi16, b_hi16);
    /* Sum cross terms and split into lo32 / hi32. */
    HVX_Vector cross  = Q6_Vw_vadd_VwVw(p_lohi, p_hilo);  /* < 2^31 (each <2^30) */
    HVX_Vector cross_lo16 = Q6_V_vand_VV(cross, Q6_V_vsplat_R(0x0000FFFF));
    HVX_Vector cross_hi16 = Q6_Vuw_vlsr_VuwR(cross, 16);
    /* lo32 = p_lolo + (cross_lo16 << 16). Carries into hi via sub-trick. */
    HVX_Vector cross_shl  = Q6_Vw_vasl_VwR(cross_lo16, 16);
    HVX_Vector lo32_pre   = Q6_Vw_vadd_VwVw(p_lolo, cross_shl);
    /* Carry detection: if lo32_pre < p_lolo, an overflow happened. */
    HVX_VectorPred carry  = Q6_Q_vcmp_gtu_VwVw(p_lolo, lo32_pre);
    HVX_Vector     carry_u32 = Q6_V_vmux_QVV(carry, Q6_V_vsplat_R(1), Q6_V_vzero());
    HVX_Vector hi32_pre  = Q6_Vw_vadd_VwVw(p_hihi, cross_hi16);
    HVX_Vector hi32      = Q6_Vw_vadd_VwVw(hi32_pre, carry_u32);
    return Q6_W_vcombine_VV(hi32, lo32_pre);  /* hi in [1], lo in [0] */
}

/* Extract lo/hi halves of a u64 vector pair as u32 vectors. */
#define SP_HVX_PAIR_LO(p)  Q6_V_lo_W(p)
#define SP_HVX_PAIR_HI(p)  Q6_V_hi_W(p)

/* 32-bit arithmetic helpers, all vector lanes parallel. */
#define SP_HVX_AND_U32(a, b)      Q6_V_vand_VV((a), (b))
#define SP_HVX_OR_U32(a, b)       Q6_V_vor_VV((a), (b))
#define SP_HVX_ADD_U32(a, b)      Q6_Vw_vadd_VwVw((a), (b))
#define SP_HVX_SUB_U32(a, b)      Q6_Vw_vsub_VwVw((a), (b))
#define SP_HVX_SHR_U32(a, n)      Q6_Vuw_vlsr_VuwR((a), (n))
#define SP_HVX_SPLAT_U32(scalar)  Q6_V_vsplat_R((scalar))
#define SP_HVX_MIN_U32(a, b)      Q6_Vuw_vmin_VuwVuw((a), (b))

#else  /* SP_HEXAGON_ENABLED not set — scalar reference path for x86 / CI */

typedef struct { uint32_t lane[SP_HVX_LANES]; } sp_hvx_v32;
typedef struct { uint64_t lane[SP_HVX_LANES]; } sp_hvx_v64;

static inline sp_hvx_v32 SP_HVX_LOAD_U32(const uint32_t* p) {
    sp_hvx_v32 v; memcpy(v.lane, p, sizeof(v)); return v;
}
static inline void SP_HVX_STORE_U32(uint32_t* p, sp_hvx_v32 v) {
    memcpy(p, v.lane, sizeof(v));
}
static inline sp_hvx_v64 sp_hvx_mul_u32_to_u64(sp_hvx_v32 a, sp_hvx_v32 b) {
    sp_hvx_v64 r;
    for (int i = 0; i < SP_HVX_LANES; ++i) {
        r.lane[i] = (uint64_t)a.lane[i] * (uint64_t)b.lane[i];
    }
    return r;
}
static inline sp_hvx_v32 SP_HVX_PAIR_LO(sp_hvx_v64 p) {
    sp_hvx_v32 r;
    for (int i = 0; i < SP_HVX_LANES; ++i) r.lane[i] = (uint32_t)(p.lane[i] & 0xFFFFFFFFu);
    return r;
}
static inline sp_hvx_v32 SP_HVX_PAIR_HI(sp_hvx_v64 p) {
    sp_hvx_v32 r;
    for (int i = 0; i < SP_HVX_LANES; ++i) r.lane[i] = (uint32_t)(p.lane[i] >> 32);
    return r;
}
#define SP_HVX_LANE_OP(out, in_a, in_b, OP) \
    do { for (int _i = 0; _i < SP_HVX_LANES; ++_i) (out).lane[_i] = (OP); } while (0)

static inline sp_hvx_v32 SP_HVX_AND_U32(sp_hvx_v32 a, sp_hvx_v32 b) {
    sp_hvx_v32 r; for (int i = 0; i < SP_HVX_LANES; ++i) r.lane[i] = a.lane[i] & b.lane[i]; return r;
}
static inline sp_hvx_v32 SP_HVX_OR_U32(sp_hvx_v32 a, sp_hvx_v32 b) {
    sp_hvx_v32 r; for (int i = 0; i < SP_HVX_LANES; ++i) r.lane[i] = a.lane[i] | b.lane[i]; return r;
}
static inline sp_hvx_v32 SP_HVX_ADD_U32(sp_hvx_v32 a, sp_hvx_v32 b) {
    sp_hvx_v32 r; for (int i = 0; i < SP_HVX_LANES; ++i) r.lane[i] = a.lane[i] + b.lane[i]; return r;
}
static inline sp_hvx_v32 SP_HVX_SUB_U32(sp_hvx_v32 a, sp_hvx_v32 b) {
    sp_hvx_v32 r; for (int i = 0; i < SP_HVX_LANES; ++i) r.lane[i] = a.lane[i] - b.lane[i]; return r;
}
static inline sp_hvx_v32 SP_HVX_SHR_U32(sp_hvx_v32 a, int n) {
    sp_hvx_v32 r; for (int i = 0; i < SP_HVX_LANES; ++i) r.lane[i] = a.lane[i] >> n; return r;
}
static inline sp_hvx_v32 SP_HVX_SPLAT_U32(uint32_t s) {
    sp_hvx_v32 r; for (int i = 0; i < SP_HVX_LANES; ++i) r.lane[i] = s; return r;
}
static inline sp_hvx_v32 SP_HVX_MIN_U32(sp_hvx_v32 a, sp_hvx_v32 b) {
    sp_hvx_v32 r; for (int i = 0; i < SP_HVX_LANES; ++i)
        r.lane[i] = (a.lane[i] < b.lane[i]) ? a.lane[i] : b.lane[i];
    return r;
}

#endif /* SP_HEXAGON_ENABLED */

/* ─── Barrett mod-multiply tile ────────────────────────────────────────
 *
 * 32-lane parallel Barrett reduction mirroring sp_ntt_crt.c::vbarrett_mul.
 * Inputs: 30-bit unsigned operands in u32 lanes; outputs: 30-bit result.
 * Six-op sequence, same as AVX-512:
 *   prod    = a * b              (32x32 → 64, vector pair)
 *   prod_hi = prod >> 32
 *   prod_lo = prod & 0xFFFFFFFF
 *   term1   = prod_hi * mu_u32   (operand widening)
 *   term2   = prod_lo * mu_u32
 *   q_hat   = (term1 >> 29) + (term2 >> 61)
 *   r       = prod - q_hat * q   (low 32 bits suffice since r < 2*q < 2^31)
 *   r = min(r, r - q)            (×2 to handle the worst case)
 *
 * NOTE: mu = floor(2^61 / q) ~ 2^31 so it fits in u32. q ~ 2^30 fits in u32.
 */
static inline sp_hvx_v32 sp_hvx_barrett_mul(sp_hvx_v32 a, sp_hvx_v32 b,
                                            sp_hvx_v32 v_q, sp_hvx_v32 v_mu) {
    sp_hvx_v64 prod    = sp_hvx_mul_u32_to_u64(a, b);
    sp_hvx_v32 prod_lo = SP_HVX_PAIR_LO(prod);
    sp_hvx_v32 prod_hi = SP_HVX_PAIR_HI(prod);

    /* term1 = prod_hi * mu  (both u32, product u64) */
    sp_hvx_v64 term1   = sp_hvx_mul_u32_to_u64(prod_hi, v_mu);
    /* term2 = prod_lo * mu  (both u32, product u64) */
    sp_hvx_v64 term2   = sp_hvx_mul_u32_to_u64(prod_lo, v_mu);

    /* q_hat = (term1 >> 29) + (term2 >> 61).
     * term1 >> 29: upper 35 bits of a 64-bit value. The low 3 bits of
     * term1_hi shifted by (29-32)=-3 → take term1_hi shifted LEFT 3, OR with
     * term1_lo shifted RIGHT 29. We're doing this manually because HVX
     * lacks 64-bit shifts; we shift the two halves separately.
     *
     * term2 >> 61: only the top 3 bits of term2_hi remain (bit 61,62,63),
     * so q_hat_term2 = term2_hi >> 29 (since term2_hi already holds bits 32-63).
     */
    sp_hvx_v32 t1_hi   = SP_HVX_PAIR_HI(term1);
    sp_hvx_v32 t1_lo   = SP_HVX_PAIR_LO(term1);
    /* term1 >> 29 across the 64-bit boundary:
     *   low_out  = (t1_lo >> 29) | (t1_hi << 3)   -- 32 useful bits
     *   high_out = (t1_hi >> 29)                  -- top 3 bits (small)
     * For Barrett q_hat we only need ~32-bit result (q_hat * q fits 60 bits). */
    sp_hvx_v32 t1_lo_sr = SP_HVX_SHR_U32(t1_lo, 29);
    /* Use SHL of t1_hi by 3 via mul-by-8; HVX has no general vasl_VwR for
     * left-shift but vasl_VwR exists; fall back to mul by 8 if needed. */
#ifdef SP_HEXAGON_ENABLED
    sp_hvx_v32 t1_hi_sl3 = Q6_Vw_vasl_VwR(t1_hi, 3);
#else
    sp_hvx_v32 t1_hi_sl3;
    for (int i = 0; i < SP_HVX_LANES; ++i) t1_hi_sl3.lane[i] = t1_hi.lane[i] << 3;
#endif
    sp_hvx_v32 q_hat_part1 = SP_HVX_OR_U32(t1_lo_sr, t1_hi_sl3);

    sp_hvx_v32 t2_hi   = SP_HVX_PAIR_HI(term2);
    sp_hvx_v32 q_hat_part2 = SP_HVX_SHR_U32(t2_hi, 29);  /* term2 >> 61 == t2_hi >> 29 */

    sp_hvx_v32 q_hat   = SP_HVX_ADD_U32(q_hat_part1, q_hat_part2);

    /* r = prod_lo - (q_hat * q)_lo32.  q_hat * q < q * (2^32 / q) ~ 2^32,
     * so the low 32 bits of the product suffice. */
#ifdef SP_HEXAGON_ENABLED
    sp_hvx_v32 q_hat_q_lo = Q6_Vw_vmpyi_VwVw(q_hat, v_q);  /* truncating 32-bit mul */
#else
    sp_hvx_v32 q_hat_q_lo;
    for (int i = 0; i < SP_HVX_LANES; ++i)
        q_hat_q_lo.lane[i] = (uint32_t)((uint64_t)q_hat.lane[i] * (uint64_t)v_q.lane[i]);
#endif
    sp_hvx_v32 r = SP_HVX_SUB_U32(prod_lo, q_hat_q_lo);
    /* Two conditional subtracts (worst case r is in [0, 3*q)). */
    sp_hvx_v32 r_minus_q = SP_HVX_SUB_U32(r, v_q);
    r = SP_HVX_MIN_U32(r, r_minus_q);
    r_minus_q = SP_HVX_SUB_U32(r, v_q);
    r = SP_HVX_MIN_U32(r, r_minus_q);
    return r;
}

/* ─── u64 ↔ u32 narrowing helpers ────────────────────────────────────── */
/* The public API takes uint64[N] (matching sp_ntt_crt.h); the HVX kernel
 * works on uint32[N] internally. We narrow at entry, widen at exit.
 * For 30-bit residues these are lossless. */

static void narrow_u64_to_u32(uint32_t* dst, const uint64_t* src, int n) {
    for (int i = 0; i < n; ++i) dst[i] = (uint32_t)src[i];
}
static void widen_u32_to_u64(uint64_t* dst, const uint32_t* src, int n) {
    for (int i = 0; i < n; ++i) dst[i] = (uint64_t)src[i];
}

/* In-place bit-reverse on a u32 buffer (replaces the u64 bitrev). */
static void bitrev_permute_u32(uint32_t* a) {
    for (int i = 0; i < SP_NTT_CRT_N; ++i) {
        uint32_t j = sp_ntt_crt_bitrev[i];
        if ((uint32_t)i < j) {
            uint32_t t = a[i]; a[i] = a[j]; a[j] = t;
        }
    }
}

/* Resolve which uint32 twiddle/psi table corresponds to a given context.
 * The HVX kernel needs uint32 mirrors of the tables stored in the
 * sp_ntt_ctx — these are emitted by gen_ntt_crt_consts.py alongside the
 * uint64 ones. We dispatch on the context pointer identity since C
 * doesn't easily let us extend the struct without rippling the AVX-512
 * path. */
static const uint32_t* psi_pow_u32_for(const sp_ntt_ctx* ctx) {
    if (ctx == &SP_NTT_CRT_CTX_Q1) return sp_ntt_crt_psi_pow1_u32;
    if (ctx == &SP_NTT_CRT_CTX_Q2) return sp_ntt_crt_psi_pow2_u32;
    return NULL;
}
static const uint32_t* psi_inv_pow_u32_for(const sp_ntt_ctx* ctx) {
    if (ctx == &SP_NTT_CRT_CTX_Q1) return sp_ntt_crt_psi_inv_pow1_u32;
    if (ctx == &SP_NTT_CRT_CTX_Q2) return sp_ntt_crt_psi_inv_pow2_u32;
    return NULL;
}
static const uint32_t* omega_pow_u32_for(const sp_ntt_ctx* ctx) {
    if (ctx == &SP_NTT_CRT_CTX_Q1) return sp_ntt_crt_omega_pow1_u32;
    if (ctx == &SP_NTT_CRT_CTX_Q2) return sp_ntt_crt_omega_pow2_u32;
    return NULL;
}
static const uint32_t* omega_inv_pow_u32_for(const sp_ntt_ctx* ctx) {
    if (ctx == &SP_NTT_CRT_CTX_Q1) return sp_ntt_crt_omega_inv_pow1_u32;
    if (ctx == &SP_NTT_CRT_CTX_Q2) return sp_ntt_crt_omega_inv_pow2_u32;
    return NULL;
}

/* ─── Butterfly NTT on uint32 lanes ────────────────────────────────────
 * Same algorithm as ntt_cyclic_fwd in sp_ntt_crt.c, but the SIMD width
 * is 32 (HVX) instead of 8 (AVX-512). Layer-flat omega twiddles. */

static void ntt_cyclic_fwd_hvx(uint32_t* a, const sp_ntt_ctx* ctx,
                               const uint32_t* w_tab_u32) {
    bitrev_permute_u32(a);
    const uint32_t q  = (uint32_t)ctx->q;
    const uint32_t mu = (uint32_t)ctx->barrett_mu;
    const sp_hvx_v32 v_q  = SP_HVX_SPLAT_U32(q);
    const sp_hvx_v32 v_mu = SP_HVX_SPLAT_U32(mu);

    int offset = 0;
    for (int length = 2; length <= SP_NTT_CRT_N; length <<= 1) {
        int half = length >> 1;
        if (half >= SP_HVX_LANES) {
            /* Wide-tile path: process 32 elements per pair-half. */
            for (int start = 0; start < SP_NTT_CRT_N; start += length) {
                for (int j = 0; j < half; j += SP_HVX_LANES) {
                    sp_hvx_v32 u_v = SP_HVX_LOAD_U32((const uint32_t*)(a + start + j));
                    sp_hvx_v32 x_v = SP_HVX_LOAD_U32((const uint32_t*)(a + start + half + j));
                    sp_hvx_v32 w_v = SP_HVX_LOAD_U32((const uint32_t*)(w_tab_u32 + offset + j));
                    sp_hvx_v32 t_v = sp_hvx_barrett_mul(x_v, w_v, v_q, v_mu);
                    /* sum = (u + t) mod q; both < q, sum < 2q < 2^31. */
                    sp_hvx_v32 sum = SP_HVX_ADD_U32(u_v, t_v);
                    sp_hvx_v32 sum_sub = SP_HVX_SUB_U32(sum, v_q);
                    sum = SP_HVX_MIN_U32(sum, sum_sub);
                    /* diff = (u + q - t) mod q. */
                    sp_hvx_v32 u_plus_q = SP_HVX_ADD_U32(u_v, v_q);
                    sp_hvx_v32 diff = SP_HVX_SUB_U32(u_plus_q, t_v);
                    sp_hvx_v32 diff_sub = SP_HVX_SUB_U32(diff, v_q);
                    diff = SP_HVX_MIN_U32(diff, diff_sub);
                    SP_HVX_STORE_U32((uint32_t*)(a + start + j), sum);
                    SP_HVX_STORE_U32((uint32_t*)(a + start + half + j), diff);
                }
            }
        } else {
            /* Narrow layers (half < 32): use scalar Barrett. The first
             * few NTT layers touch tiny tiles where vector dispatch is
             * pure overhead. Same fallback shape as AVX-512 path. */
            for (int start = 0; start < SP_NTT_CRT_N; start += length) {
                for (int j = 0; j < half; ++j) {
                    uint64_t w = (uint64_t)w_tab_u32[offset + j];
                    uint64_t u = a[start + j];
                    uint64_t x = a[start + half + j];
                    /* Inline scalar Barrett — identical to mulmod_barrett. */
                    uint64_t prod = x * w;
                    uint64_t hi = prod >> 32;
                    uint64_t lo = prod & 0xFFFFFFFFULL;
                    uint64_t term1 = hi * mu;
                    uint64_t term2 = lo * mu;
                    uint64_t q_hat = (term1 >> 29) + (term2 >> 61);
                    uint64_t r = prod - q_hat * q;
                    if (r >= q) r -= q;
                    if (r >= q) r -= q;
                    uint64_t sum = u + r;  if (sum >= q) sum -= q;
                    uint64_t diff = u + q - r;  if (diff >= q) diff -= q;
                    a[start + j]        = (uint32_t)sum;
                    a[start + half + j] = (uint32_t)diff;
                }
            }
        }
        offset += half;
    }
}

static void ntt_cyclic_inv_hvx(uint32_t* a, const sp_ntt_ctx* ctx,
                               const uint32_t* w_tab_u32) {
    bitrev_permute_u32(a);
    const uint32_t q  = (uint32_t)ctx->q;
    const uint32_t mu = (uint32_t)ctx->barrett_mu;
    const sp_hvx_v32 v_q  = SP_HVX_SPLAT_U32(q);
    const sp_hvx_v32 v_mu = SP_HVX_SPLAT_U32(mu);

    int offset = 0;
    for (int length = 2; length <= SP_NTT_CRT_N; length <<= 1) {
        int half = length >> 1;
        if (half >= SP_HVX_LANES) {
            for (int start = 0; start < SP_NTT_CRT_N; start += length) {
                for (int j = 0; j < half; j += SP_HVX_LANES) {
                    sp_hvx_v32 u_v = SP_HVX_LOAD_U32((const uint32_t*)(a + start + j));
                    sp_hvx_v32 x_v = SP_HVX_LOAD_U32((const uint32_t*)(a + start + half + j));
                    sp_hvx_v32 w_v = SP_HVX_LOAD_U32((const uint32_t*)(w_tab_u32 + offset + j));
                    sp_hvx_v32 t_v = sp_hvx_barrett_mul(x_v, w_v, v_q, v_mu);
                    sp_hvx_v32 sum = SP_HVX_ADD_U32(u_v, t_v);
                    sp_hvx_v32 sum_sub = SP_HVX_SUB_U32(sum, v_q);
                    sum = SP_HVX_MIN_U32(sum, sum_sub);
                    sp_hvx_v32 u_plus_q = SP_HVX_ADD_U32(u_v, v_q);
                    sp_hvx_v32 diff = SP_HVX_SUB_U32(u_plus_q, t_v);
                    sp_hvx_v32 diff_sub = SP_HVX_SUB_U32(diff, v_q);
                    diff = SP_HVX_MIN_U32(diff, diff_sub);
                    SP_HVX_STORE_U32((uint32_t*)(a + start + j), sum);
                    SP_HVX_STORE_U32((uint32_t*)(a + start + half + j), diff);
                }
            }
        } else {
            for (int start = 0; start < SP_NTT_CRT_N; start += length) {
                for (int j = 0; j < half; ++j) {
                    uint64_t w = (uint64_t)w_tab_u32[offset + j];
                    uint64_t u = a[start + j];
                    uint64_t x = a[start + half + j];
                    uint64_t prod = x * w;
                    uint64_t hi = prod >> 32;
                    uint64_t lo = prod & 0xFFFFFFFFULL;
                    uint64_t term1 = hi * mu;
                    uint64_t term2 = lo * mu;
                    uint64_t q_hat = (term1 >> 29) + (term2 >> 61);
                    uint64_t r = prod - q_hat * q;
                    if (r >= q) r -= q;
                    if (r >= q) r -= q;
                    uint64_t sum = u + r;  if (sum >= q) sum -= q;
                    uint64_t diff = u + q - r;  if (diff >= q) diff -= q;
                    a[start + j]        = (uint32_t)sum;
                    a[start + half + j] = (uint32_t)diff;
                }
            }
        }
        offset += half;
    }

    /* Scale every coefficient by N^-1 mod q.  N^-1 ~ 30 bits, fits u32. */
    const sp_hvx_v32 v_ninv = SP_HVX_SPLAT_U32((uint32_t)ctx->n_inv);
    for (int i = 0; i < SP_NTT_CRT_N; i += SP_HVX_LANES) {
        sp_hvx_v32 x = SP_HVX_LOAD_U32((const uint32_t*)(a + i));
        sp_hvx_v32 r = sp_hvx_barrett_mul(x, v_ninv, v_q, v_mu);
        SP_HVX_STORE_U32((uint32_t*)(a + i), r);
    }
}

/* ─── Public API ───────────────────────────────────────────────────── */

void sp_ntt_crt_hvx_forward(uint64_t* a, const sp_ntt_ctx* ctx) {
    const uint32_t* psi_u32   = psi_pow_u32_for(ctx);
    const uint32_t* omega_u32 = omega_pow_u32_for(ctx);
    if (!psi_u32 || !omega_u32) {
        /* Unknown context — defer to the scalar reference. */
        sp_ntt_crt_forward(a, ctx);
        return;
    }
    /* Narrow uint64 input → uint32 lanes. */
    SP_HVX_ALIGNED uint32_t buf[SP_NTT_CRT_N];
    narrow_u64_to_u32(buf, a, SP_NTT_CRT_N);

    /* Pre-twist a[i] *= psi^i mod q  (32-lane Barrett vectorised). */
    const uint32_t q  = (uint32_t)ctx->q;
    const uint32_t mu = (uint32_t)ctx->barrett_mu;
    const sp_hvx_v32 v_q  = SP_HVX_SPLAT_U32(q);
    const sp_hvx_v32 v_mu = SP_HVX_SPLAT_U32(mu);
    for (int i = 0; i < SP_NTT_CRT_N; i += SP_HVX_LANES) {
        sp_hvx_v32 ai  = SP_HVX_LOAD_U32(buf + i);
        sp_hvx_v32 psi = SP_HVX_LOAD_U32(psi_u32 + i);
        sp_hvx_v32 r   = sp_hvx_barrett_mul(ai, psi, v_q, v_mu);
        SP_HVX_STORE_U32(buf + i, r);
    }
    /* Butterfly forward NTT. */
    ntt_cyclic_fwd_hvx(buf, ctx, omega_u32);

    /* Widen back to uint64 for the caller. */
    widen_u32_to_u64(a, buf, SP_NTT_CRT_N);
}

void sp_ntt_crt_hvx_inverse(uint64_t* a, const sp_ntt_ctx* ctx) {
    const uint32_t* psi_inv_u32   = psi_inv_pow_u32_for(ctx);
    const uint32_t* omega_inv_u32 = omega_inv_pow_u32_for(ctx);
    if (!psi_inv_u32 || !omega_inv_u32) {
        sp_ntt_crt_inverse(a, ctx);
        return;
    }
    SP_HVX_ALIGNED uint32_t buf[SP_NTT_CRT_N];
    narrow_u64_to_u32(buf, a, SP_NTT_CRT_N);

    /* Inverse butterfly first (mirrors the AVX-512 path). */
    ntt_cyclic_inv_hvx(buf, ctx, omega_inv_u32);

    /* Post-twist a[i] *= psi^-i mod q. */
    const uint32_t q  = (uint32_t)ctx->q;
    const uint32_t mu = (uint32_t)ctx->barrett_mu;
    const sp_hvx_v32 v_q  = SP_HVX_SPLAT_U32(q);
    const sp_hvx_v32 v_mu = SP_HVX_SPLAT_U32(mu);
    for (int i = 0; i < SP_NTT_CRT_N; i += SP_HVX_LANES) {
        sp_hvx_v32 ai  = SP_HVX_LOAD_U32(buf + i);
        sp_hvx_v32 psi = SP_HVX_LOAD_U32(psi_inv_u32 + i);
        sp_hvx_v32 r   = sp_hvx_barrett_mul(ai, psi, v_q, v_mu);
        SP_HVX_STORE_U32(buf + i, r);
    }

    widen_u32_to_u64(a, buf, SP_NTT_CRT_N);
}

void sp_ntt_crt_hvx_pointwise_mul(uint64_t* c,
                                  const uint64_t* a,
                                  const uint64_t* b,
                                  const sp_ntt_ctx* ctx) {
    SP_HVX_ALIGNED uint32_t a_u32[SP_NTT_CRT_N];
    SP_HVX_ALIGNED uint32_t b_u32[SP_NTT_CRT_N];
    SP_HVX_ALIGNED uint32_t c_u32[SP_NTT_CRT_N];
    narrow_u64_to_u32(a_u32, a, SP_NTT_CRT_N);
    narrow_u64_to_u32(b_u32, b, SP_NTT_CRT_N);

    const sp_hvx_v32 v_q  = SP_HVX_SPLAT_U32((uint32_t)ctx->q);
    const sp_hvx_v32 v_mu = SP_HVX_SPLAT_U32((uint32_t)ctx->barrett_mu);
    for (int i = 0; i < SP_NTT_CRT_N; i += SP_HVX_LANES) {
        sp_hvx_v32 va = SP_HVX_LOAD_U32(a_u32 + i);
        sp_hvx_v32 vb = SP_HVX_LOAD_U32(b_u32 + i);
        sp_hvx_v32 r  = sp_hvx_barrett_mul(va, vb, v_q, v_mu);
        SP_HVX_STORE_U32(c_u32 + i, r);
    }
    widen_u32_to_u64(c, c_u32, SP_NTT_CRT_N);
}

/* ─── Engine integration entry point ───────────────────────────────── */
/* Same shape as sp_poly_dot_product_ntt_crt_qk_cached in sp_ntt_crt.c —
 * just routes the inner ops through the HVX kernel. The CRT-combine
 * and fp32 decode are identical (scalar, one-shot per token). */
float sp_poly_dot_product_ntt_crt_qk_cached_hvx(
        const uint64_t* Q_ntt_q1,
        const uint64_t* Q_ntt_q2,
        const uint64_t* K_ntt_q1,
        const uint64_t* K_ntt_q2,
        int d, double delta,
        uint64_t* c_q1_scratch,
        uint64_t* c_q2_scratch,
        int* ok)
{
    if (d > SP_NTT_CRT_N) { if (ok) *ok = 0; return 0.0f; }
    sp_ntt_crt_hvx_pointwise_mul(c_q1_scratch, Q_ntt_q1, K_ntt_q1, &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_hvx_pointwise_mul(c_q2_scratch, Q_ntt_q2, K_ntt_q2, &SP_NTT_CRT_CTX_Q2);
    sp_ntt_crt_hvx_inverse(c_q1_scratch, &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_hvx_inverse(c_q2_scratch, &SP_NTT_CRT_CTX_Q2);
    const uint64_t u1 = c_q1_scratch[d - 1];
    const uint64_t u2 = c_q2_scratch[d - 1];
    const uint64_t x  = sp_ntt_crt_combine(u1, u2);
    const uint64_t M    = SP_NTT_CRT_Q1 * SP_NTT_CRT_Q2;
    const uint64_t HALF = M >> 1;
    const int64_t  coeff = (x > HALF) ? -(int64_t)(M - x) : (int64_t)x;
    if (ok) *ok = 1;
    return (float)((double)coeff / (delta * delta));
}

int sp_ntt_crt_hvx_available(void) {
#ifdef SP_HEXAGON_ENABLED
    return 1;
#else
    return 0;
#endif
}

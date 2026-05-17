/* sp_ntt_crt.c — Phase 10 SIMD-accelerated dual-prime CRT NTT.
 *
 * Mathematical kernel: forward + pointwise + inverse NTT under the
 * negacyclic ring Z_q[x]/(x^N+1) for two 30-bit Proth primes q1, q2.
 * Combined modulus M = q1*q2 ~ 2^60. All intermediates fit in uint64.
 *
 * Phase 10 SIMD additions (this revision):
 *   - Layer-flat omega twiddle tables (gen_ntt_crt_consts.py emits these).
 *   - Vectorized 64-bit modular multiply (Barrett, 8 lanes / 512-bit reg).
 *   - sp_ntt_crt_pointwise_mul: AVX-512 path with scalar fallback.
 *   - ntt_cyclic_fwd / ntt_cyclic_inv: AVX-512 butterfly path at half>=8,
 *     scalar (Barrett) path at length<16.
 *
 * Verified by tests/test_sp_ntt_crt.cpp Tests 1-6 against scalar baseline.
 * Test 7 captures microbench numbers for the vectorized kernel.
 */

#include "sp_ntt_crt.h"
#include "sp_ntt_crt_consts.h"

#include <string.h>

/* All intrinsics used here (_mm512_mul_epu32, _mm512_min_epu64,
 * _mm512_set1_epi64, etc.) are AVX-512F only -- no DQ/BW/VL required.
 * Engine builds with /arch:AVX512 (MSVC) or -mavx512f (gcc) so this
 * enables on Tiger Lake+ and Skylake-X+. */
#if defined(__AVX512F__)
#  include <immintrin.h>
#  define SP_NTT_CRT_AVX512 1
#endif

/* Predefined contexts for the two production primes. */
const sp_ntt_ctx SP_NTT_CRT_CTX_Q1 = {
    SP_NTT_CRT_Q1,
    SP_NTT_CRT_N_INV1,
    sp_ntt_crt_psi_pow1,
    sp_ntt_crt_psi_inv_pow1,
    SP_NTT_CRT_PSI1,
    SP_NTT_CRT_PSI_INV1,
    SP_NTT_CRT_BARRETT_MU1,
    sp_ntt_crt_omega_pow1,
    sp_ntt_crt_omega_inv_pow1
};

const sp_ntt_ctx SP_NTT_CRT_CTX_Q2 = {
    SP_NTT_CRT_Q2,
    SP_NTT_CRT_N_INV2,
    sp_ntt_crt_psi_pow2,
    sp_ntt_crt_psi_inv_pow2,
    SP_NTT_CRT_PSI2,
    SP_NTT_CRT_PSI_INV2,
    SP_NTT_CRT_BARRETT_MU2,
    sp_ntt_crt_omega_pow2,
    sp_ntt_crt_omega_inv_pow2
};

/* ---- Scalar primitives (parity reference + fallback) ----------------- */

/* Scalar Barrett: q < 2^30, prod = a*b < 2^60, mu = floor(2^61 / q).
 * Matches the AVX-512 sequence below bit-for-bit. */
static inline uint64_t mulmod_barrett(uint64_t a, uint64_t b,
                                      uint64_t q, uint64_t mu) {
    uint64_t prod   = a * b;
    uint64_t hi     = prod >> 32;
    uint64_t lo     = prod & 0xFFFFFFFFULL;
    uint64_t term1  = hi * mu;
    uint64_t term2  = lo * mu;
    uint64_t q_hat  = (term1 >> 29) + (term2 >> 61);
    uint64_t r      = prod - q_hat * q;
    if (r >= q) r -= q;
    if (r >= q) r -= q;
    return r;
}

static inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t q) {
    return (a * b) % q;
}

static inline uint64_t addmod(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t s = a + b;
    return (s >= q) ? (s - q) : s;
}

static inline uint64_t submod(uint64_t a, uint64_t b, uint64_t q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

/* In-place bit-reverse permutation. */
static void bitrev_permute(uint64_t* a) {
    for (int i = 0; i < SP_NTT_CRT_N; ++i) {
        uint32_t j = sp_ntt_crt_bitrev[i];
        if ((uint32_t)i < j) {
            uint64_t t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }
}

/* ---- AVX-512 vectorized Barrett primitive --------------------------- */

#if SP_NTT_CRT_AVX512
/* 8-lane Barrett mod-multiply. Each 64-bit lane of va,vb holds a u64
 * value in [0, q). Returns 8 u64 values in [0, q).
 *
 * Operand contract: va_lo32 and vb_lo32 each contain the full operand
 * (< 2^30, fits in low 32 bits of each lane). `_mm512_mul_epu32` thus
 * yields the correct 32x32->64 product per lane. */
static inline __m512i vbarrett_mul(__m512i va, __m512i vb,
                                   __m512i v_q, __m512i v_mu,
                                   __m512i v_mask32) {
    __m512i prod    = _mm512_mul_epu32(va, vb);
    __m512i prod_hi = _mm512_srli_epi64(prod, 32);
    __m512i prod_lo = _mm512_and_si512(prod, v_mask32);
    __m512i term1   = _mm512_mul_epu32(prod_hi, v_mu);
    __m512i term2   = _mm512_mul_epu32(prod_lo, v_mu);
    __m512i q_hat   = _mm512_add_epi64(
                          _mm512_srli_epi64(term1, 29),
                          _mm512_srli_epi64(term2, 61));
    __m512i q_hat_q = _mm512_mul_epu32(q_hat, v_q);
    __m512i r       = _mm512_sub_epi64(prod, q_hat_q);
    __m512i r_sub   = _mm512_sub_epi64(r, v_q);
    r = _mm512_min_epu64(r, r_sub);
    r_sub = _mm512_sub_epi64(r, v_q);
    r = _mm512_min_epu64(r, r_sub);
    return r;
}
#endif

/* ---- Butterfly NTT (CT-DIT with bit-reverse-first, layer-flat twiddles) */

static void ntt_cyclic_fwd(uint64_t* a, const sp_ntt_ctx* ctx) {
    bitrev_permute(a);
    const uint64_t q  = ctx->q;
    const uint64_t mu = ctx->barrett_mu;
    const uint64_t* w_tab = ctx->omega_pow;
    int offset = 0;
    for (int length = 2; length <= SP_NTT_CRT_N; length <<= 1) {
        int half = length >> 1;
#if SP_NTT_CRT_AVX512
        if (half >= 8) {
            const __m512i v_q      = _mm512_set1_epi64((long long)q);
            const __m512i v_mu     = _mm512_set1_epi64((long long)mu);
            const __m512i v_mask32 = _mm512_set1_epi64((long long)0xFFFFFFFFULL);
            for (int start = 0; start < SP_NTT_CRT_N; start += length) {
                for (int j = 0; j < half; j += 8) {
                    __m512i u_v = _mm512_loadu_si512((const __m512i*)(a + start + j));
                    __m512i x_v = _mm512_loadu_si512((const __m512i*)(a + start + half + j));
                    __m512i w_v = _mm512_loadu_si512((const __m512i*)(w_tab + offset + j));
                    __m512i t_v = vbarrett_mul(x_v, w_v, v_q, v_mu, v_mask32);
                    /* sum = (u + t) mod q.  Both < q, sum < 2*q < 2^31. */
                    __m512i sum     = _mm512_add_epi64(u_v, t_v);
                    __m512i sum_sub = _mm512_sub_epi64(sum, v_q);
                    sum = _mm512_min_epu64(sum, sum_sub);
                    /* diff = (u - t) mod q via (u + q - t). */
                    __m512i diff     = _mm512_sub_epi64(_mm512_add_epi64(u_v, v_q), t_v);
                    __m512i diff_sub = _mm512_sub_epi64(diff, v_q);
                    diff = _mm512_min_epu64(diff, diff_sub);
                    _mm512_storeu_si512((__m512i*)(a + start + j), sum);
                    _mm512_storeu_si512((__m512i*)(a + start + half + j), diff);
                }
            }
        } else
#endif
        {
            for (int start = 0; start < SP_NTT_CRT_N; start += length) {
                for (int j = 0; j < half; ++j) {
                    uint64_t w = w_tab[offset + j];
                    uint64_t u = a[start + j];
                    uint64_t v = mulmod_barrett(a[start + half + j], w, q, mu);
                    a[start + j]        = addmod(u, v, q);
                    a[start + half + j] = submod(u, v, q);
                }
            }
        }
        offset += half;
    }
}

static void ntt_cyclic_inv(uint64_t* a, const sp_ntt_ctx* ctx) {
    bitrev_permute(a);
    const uint64_t q  = ctx->q;
    const uint64_t mu = ctx->barrett_mu;
    const uint64_t* w_tab = ctx->omega_inv_pow;
    int offset = 0;
    for (int length = 2; length <= SP_NTT_CRT_N; length <<= 1) {
        int half = length >> 1;
#if SP_NTT_CRT_AVX512
        if (half >= 8) {
            const __m512i v_q      = _mm512_set1_epi64((long long)q);
            const __m512i v_mu     = _mm512_set1_epi64((long long)mu);
            const __m512i v_mask32 = _mm512_set1_epi64((long long)0xFFFFFFFFULL);
            for (int start = 0; start < SP_NTT_CRT_N; start += length) {
                for (int j = 0; j < half; j += 8) {
                    __m512i u_v = _mm512_loadu_si512((const __m512i*)(a + start + j));
                    __m512i x_v = _mm512_loadu_si512((const __m512i*)(a + start + half + j));
                    __m512i w_v = _mm512_loadu_si512((const __m512i*)(w_tab + offset + j));
                    __m512i t_v = vbarrett_mul(x_v, w_v, v_q, v_mu, v_mask32);
                    __m512i sum     = _mm512_add_epi64(u_v, t_v);
                    __m512i sum_sub = _mm512_sub_epi64(sum, v_q);
                    sum = _mm512_min_epu64(sum, sum_sub);
                    __m512i diff     = _mm512_sub_epi64(_mm512_add_epi64(u_v, v_q), t_v);
                    __m512i diff_sub = _mm512_sub_epi64(diff, v_q);
                    diff = _mm512_min_epu64(diff, diff_sub);
                    _mm512_storeu_si512((__m512i*)(a + start + j), sum);
                    _mm512_storeu_si512((__m512i*)(a + start + half + j), diff);
                }
            }
        } else
#endif
        {
            for (int start = 0; start < SP_NTT_CRT_N; start += length) {
                for (int j = 0; j < half; ++j) {
                    uint64_t w = w_tab[offset + j];
                    uint64_t u = a[start + j];
                    uint64_t v = mulmod_barrett(a[start + half + j], w, q, mu);
                    a[start + j]        = addmod(u, v, q);
                    a[start + half + j] = submod(u, v, q);
                }
            }
        }
        offset += half;
    }
    /* Scale by N^-1.  Vectorize with Barrett broadcast on n_inv. */
#if SP_NTT_CRT_AVX512
    {
        const __m512i v_q      = _mm512_set1_epi64((long long)q);
        const __m512i v_mu     = _mm512_set1_epi64((long long)mu);
        const __m512i v_mask32 = _mm512_set1_epi64((long long)0xFFFFFFFFULL);
        const __m512i v_ninv   = _mm512_set1_epi64((long long)ctx->n_inv);
        for (int i = 0; i < SP_NTT_CRT_N; i += 8) {
            __m512i x = _mm512_loadu_si512((const __m512i*)(a + i));
            __m512i r = vbarrett_mul(x, v_ninv, v_q, v_mu, v_mask32);
            _mm512_storeu_si512((__m512i*)(a + i), r);
        }
    }
#else
    for (int i = 0; i < SP_NTT_CRT_N; ++i) {
        a[i] = mulmod_barrett(a[i], ctx->n_inv, q, mu);
    }
#endif
}

/* ---- Public API ----------------------------------------------------- */

void sp_ntt_crt_forward(uint64_t* a, const sp_ntt_ctx* ctx) {
    const uint64_t q  = ctx->q;
    const uint64_t mu = ctx->barrett_mu;
    /* Pre-twist a[i] *= psi^i mod q.  Vectorize via Barrett pointwise. */
#if SP_NTT_CRT_AVX512
    {
        const __m512i v_q      = _mm512_set1_epi64((long long)q);
        const __m512i v_mu     = _mm512_set1_epi64((long long)mu);
        const __m512i v_mask32 = _mm512_set1_epi64((long long)0xFFFFFFFFULL);
        for (int i = 0; i < SP_NTT_CRT_N; i += 8) {
            __m512i ai  = _mm512_loadu_si512((const __m512i*)(a + i));
            __m512i psi = _mm512_loadu_si512((const __m512i*)(ctx->psi_pow + i));
            __m512i r   = vbarrett_mul(ai, psi, v_q, v_mu, v_mask32);
            _mm512_storeu_si512((__m512i*)(a + i), r);
        }
    }
#else
    for (int i = 0; i < SP_NTT_CRT_N; ++i) {
        a[i] = mulmod_barrett(a[i], ctx->psi_pow[i], q, mu);
    }
#endif
    ntt_cyclic_fwd(a, ctx);
}

void sp_ntt_crt_inverse(uint64_t* a, const sp_ntt_ctx* ctx) {
    ntt_cyclic_inv(a, ctx);
    const uint64_t q  = ctx->q;
    const uint64_t mu = ctx->barrett_mu;
#if SP_NTT_CRT_AVX512
    {
        const __m512i v_q      = _mm512_set1_epi64((long long)q);
        const __m512i v_mu     = _mm512_set1_epi64((long long)mu);
        const __m512i v_mask32 = _mm512_set1_epi64((long long)0xFFFFFFFFULL);
        for (int i = 0; i < SP_NTT_CRT_N; i += 8) {
            __m512i ai  = _mm512_loadu_si512((const __m512i*)(a + i));
            __m512i psi = _mm512_loadu_si512((const __m512i*)(ctx->psi_inv_pow + i));
            __m512i r   = vbarrett_mul(ai, psi, v_q, v_mu, v_mask32);
            _mm512_storeu_si512((__m512i*)(a + i), r);
        }
    }
#else
    for (int i = 0; i < SP_NTT_CRT_N; ++i) {
        a[i] = mulmod_barrett(a[i], ctx->psi_inv_pow[i], q, mu);
    }
#endif
}

void sp_ntt_crt_pointwise_mul(uint64_t* c,
                              const uint64_t* a,
                              const uint64_t* b,
                              const sp_ntt_ctx* ctx) {
    const uint64_t q  = ctx->q;
    const uint64_t mu = ctx->barrett_mu;
#if SP_NTT_CRT_AVX512
    {
        const __m512i v_q      = _mm512_set1_epi64((long long)q);
        const __m512i v_mu     = _mm512_set1_epi64((long long)mu);
        const __m512i v_mask32 = _mm512_set1_epi64((long long)0xFFFFFFFFULL);
        for (int i = 0; i < SP_NTT_CRT_N; i += 8) {
            __m512i va = _mm512_loadu_si512((const __m512i*)(a + i));
            __m512i vb = _mm512_loadu_si512((const __m512i*)(b + i));
            __m512i r  = vbarrett_mul(va, vb, v_q, v_mu, v_mask32);
            _mm512_storeu_si512((__m512i*)(c + i), r);
        }
    }
#else
    for (int i = 0; i < SP_NTT_CRT_N; ++i) {
        c[i] = mulmod_barrett(a[i], b[i], q, mu);
    }
#endif
}

void sp_ntt_crt_coeffs_from_int64(uint64_t* out,
                                  const int64_t* in,
                                  int len,
                                  uint64_t q) {
    for (int i = 0; i < len; ++i) {
        int64_t v = in[i];
        if (v >= 0) {
            uint64_t uv = (uint64_t)v;
            out[i] = (uv >= q) ? (uv % q) : uv;
        } else {
            uint64_t mag = (uint64_t)(-v);
            uint64_t r = (mag >= q) ? (mag % q) : mag;
            out[i] = (r == 0) ? 0 : (q - r);
        }
    }
}

int sp_ntt_crt_poly_mul(int64_t* c,
                        const int64_t* a,
                        const int64_t* b,
                        int N,
                        uint64_t* workspace) {
    if (N != SP_NTT_CRT_N) return -1;

    uint64_t* A_q1 = workspace + 0 * SP_NTT_CRT_N;
    uint64_t* B_q1 = workspace + 1 * SP_NTT_CRT_N;
    uint64_t* C_q1 = workspace + 2 * SP_NTT_CRT_N;
    uint64_t* A_q2 = workspace + 3 * SP_NTT_CRT_N;
    uint64_t* B_q2 = workspace + 4 * SP_NTT_CRT_N;
    uint64_t* C_q2 = workspace + 5 * SP_NTT_CRT_N;

    /* Universe 1 (mod q1) */
    sp_ntt_crt_coeffs_from_int64(A_q1, a, SP_NTT_CRT_N, SP_NTT_CRT_Q1);
    sp_ntt_crt_coeffs_from_int64(B_q1, b, SP_NTT_CRT_N, SP_NTT_CRT_Q1);
    sp_ntt_crt_forward(A_q1, &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_forward(B_q1, &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_pointwise_mul(C_q1, A_q1, B_q1, &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_inverse(C_q1, &SP_NTT_CRT_CTX_Q1);

    /* Universe 2 (mod q2) */
    sp_ntt_crt_coeffs_from_int64(A_q2, a, SP_NTT_CRT_N, SP_NTT_CRT_Q2);
    sp_ntt_crt_coeffs_from_int64(B_q2, b, SP_NTT_CRT_N, SP_NTT_CRT_Q2);
    sp_ntt_crt_forward(A_q2, &SP_NTT_CRT_CTX_Q2);
    sp_ntt_crt_forward(B_q2, &SP_NTT_CRT_CTX_Q2);
    sp_ntt_crt_pointwise_mul(C_q2, A_q2, B_q2, &SP_NTT_CRT_CTX_Q2);
    sp_ntt_crt_inverse(C_q2, &SP_NTT_CRT_CTX_Q2);

    /* CRT-stitch and map to signed int64. M = q1 * q2 fits in uint64. */
    const uint64_t M = SP_NTT_CRT_Q1 * SP_NTT_CRT_Q2;
    const uint64_t HALF = M >> 1;
    for (int i = 0; i < SP_NTT_CRT_N; ++i) {
        uint64_t x = sp_ntt_crt_combine(C_q1[i], C_q2[i]);
        c[i] = (x > HALF) ? -(int64_t)(M - x) : (int64_t)x;
    }
    return 0;
}

/* ---- Phase 9b: engine integration helpers --------------------------- */

void sp_poly_encode_ntt_q_crt(uint64_t* Q_ntt_q1,
                              uint64_t* Q_ntt_q2,
                              const float* q_vec, int d, double delta,
                              int64_t* int_scratch) {
    sp_poly Qp = { int_scratch, SP_NTT_CRT_N };
    sp_poly_zero(&Qp);
    sp_poly_encode_fp32(&Qp, q_vec, d, delta, /*reversed=*/false);
    sp_ntt_crt_coeffs_from_int64(Q_ntt_q1, int_scratch, SP_NTT_CRT_N, SP_NTT_CRT_Q1);
    sp_ntt_crt_coeffs_from_int64(Q_ntt_q2, int_scratch, SP_NTT_CRT_N, SP_NTT_CRT_Q2);
    sp_ntt_crt_forward(Q_ntt_q1, &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_forward(Q_ntt_q2, &SP_NTT_CRT_CTX_Q2);
}

void sp_poly_encode_ntt_k_reversed_crt(uint64_t* K_ntt_q1,
                                       uint64_t* K_ntt_q2,
                                       const float* k_vec, int d, double delta,
                                       int64_t* int_scratch) {
    sp_poly Kp = { int_scratch, SP_NTT_CRT_N };
    sp_poly_zero(&Kp);
    sp_poly_encode_fp32(&Kp, k_vec, d, delta, /*reversed=*/true);
    sp_ntt_crt_coeffs_from_int64(K_ntt_q1, int_scratch, SP_NTT_CRT_N, SP_NTT_CRT_Q1);
    sp_ntt_crt_coeffs_from_int64(K_ntt_q2, int_scratch, SP_NTT_CRT_N, SP_NTT_CRT_Q2);
    sp_ntt_crt_forward(K_ntt_q1, &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_forward(K_ntt_q2, &SP_NTT_CRT_CTX_Q2);
}

float sp_poly_dot_product_ntt_crt_qk_cached(const uint64_t* Q_ntt_q1,
                                            const uint64_t* Q_ntt_q2,
                                            const uint64_t* K_ntt_q1,
                                            const uint64_t* K_ntt_q2,
                                            int d, double delta,
                                            uint64_t* c_q1_scratch,
                                            uint64_t* c_q2_scratch,
                                            int* ok) {
    if (d > SP_NTT_CRT_N) {
        if (ok) *ok = 0;
        return 0.0f;
    }
    sp_ntt_crt_pointwise_mul(c_q1_scratch, Q_ntt_q1, K_ntt_q1, &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_pointwise_mul(c_q2_scratch, Q_ntt_q2, K_ntt_q2, &SP_NTT_CRT_CTX_Q2);
    sp_ntt_crt_inverse(c_q1_scratch, &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_inverse(c_q2_scratch, &SP_NTT_CRT_CTX_Q2);
    const uint64_t u1 = c_q1_scratch[d - 1];
    const uint64_t u2 = c_q2_scratch[d - 1];
    const uint64_t x  = sp_ntt_crt_combine(u1, u2);
    const uint64_t M    = SP_NTT_CRT_Q1 * SP_NTT_CRT_Q2;
    const uint64_t HALF = M >> 1;
    const int64_t  coeff = (x > HALF) ? -(int64_t)(M - x) : (int64_t)x;
    if (ok) *ok = 1;
    return (float)((double)coeff / (delta * delta));
}

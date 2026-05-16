// Shannon-Prime — Number Theoretic Transform implementation.
// Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.

#include "sp_ntt.h"
#include "sp_ntt_consts.h"

#include <string.h>
#include <assert.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

// ----- 60-bit modular multiply --------------------------------------------
//
// q < 2^60, so a*b < 2^120. We need 128-bit intermediate arithmetic, then
// reduce mod q.

// Barrett: mu = floor(2^120 / q) ~ 2^61. For x = a*b in [0, q^2):
//   q_hat = ((x_hi * mu) >> 56) + ((x_lo * mu) >> 120)
//   r     = x_lo - q_hat * q   (in [0, 2q) under 64-bit wrap)
//   one conditional subtract brings r into [0, q).
// Replaces ~20-cycle DIV with ~5-cycle MUL+SHIFT+SUB.
uint64_t sp_ntt_mulmod(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    const unsigned __int128 x   = (unsigned __int128)a * (unsigned __int128)b;
    const uint64_t          x_lo = (uint64_t)x;
    const uint64_t          x_hi = (uint64_t)(x >> 64);
    const unsigned __int128 h   = (unsigned __int128)x_hi * SP_NTT_BARRETT_MU;
    const unsigned __int128 l   = (unsigned __int128)x_lo * SP_NTT_BARRETT_MU;
    const uint64_t          q_hat = (uint64_t)(h >> 56) + (uint64_t)(l >> 120);
    uint64_t                r   = x_lo - q_hat * SP_NTT_Q;
    if (r >= SP_NTT_Q) r -= SP_NTT_Q;
    return r;
#elif defined(_MSC_VER) && defined(_M_X64)
    uint64_t x_hi;
    const uint64_t x_lo = _umul128(a, b, &x_hi);
    uint64_t h_hi;
    const uint64_t h_lo = _umul128(x_hi, SP_NTT_BARRETT_MU, &h_hi);
    const uint64_t h_top = (h_hi << 8) | (h_lo >> 56);
    uint64_t l_hi;
    (void)_umul128(x_lo, SP_NTT_BARRETT_MU, &l_hi);
    const uint64_t l_top = l_hi >> 56;
    const uint64_t q_hat = h_top + l_top;
    uint64_t r = x_lo - q_hat * SP_NTT_Q;
    if (r >= SP_NTT_Q) r -= SP_NTT_Q;
    return r;
#else
    // Portable fallback: schoolbook split. Splits a into two 30-bit halves
    // so all intermediates fit in 64 bits.
    //
    // a = a_hi * 2^30 + a_lo  (a_hi, a_lo < 2^30)
    // a * b = (a_hi * b) << 30 + (a_lo * b)
    //
    // a_hi * b < 2^30 * 2^60 = 2^90 — too big. We instead reduce after
    // each shift step using a 32-bit Barrett-ish accumulation.
    //
    // For correctness without performance pressure (this path only fires on
    // exotic toolchains): use Russian-peasant mulmod.
    uint64_t result = 0;
    uint64_t aa = a % SP_NTT_Q;
    uint64_t bb = b % SP_NTT_Q;
    while (bb > 0) {
        if (bb & 1) {
            result = (result + aa) % SP_NTT_Q;
            // Watch for overflow: aa, result both < q < 2^60, so sum < 2^61.
        }
        aa = (aa << 1) % SP_NTT_Q;
        bb >>= 1;
    }
    return result;
#endif
}

// ----- Bit-reverse permutation ---------------------------------------------

static void bitrev_permute(uint64_t* a) {
    for (int i = 0; i < SP_NTT_N; ++i) {
        uint32_t j = sp_ntt_bitrev[i];
        if ((uint32_t)i < j) {
            uint64_t tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
}

// ----- Forward / inverse NTT (cyclic core; called after pre-twist) --------

static void ntt_cyclic_forward(uint64_t* a) {
    bitrev_permute(a);
    for (int length = 2; length <= SP_NTT_N; length <<= 1) {
        // w_step = omega^(N / length) mod q
        uint64_t w_step = 1;
        {
            uint64_t base = SP_NTT_OMEGA;
            uint64_t exp = SP_NTT_N / (uint64_t)length;
            while (exp > 0) {
                if (exp & 1) w_step = sp_ntt_mulmod(w_step, base);
                base = sp_ntt_mulmod(base, base);
                exp >>= 1;
            }
        }
        int half = length / 2;
        for (int start = 0; start < SP_NTT_N; start += length) {
            uint64_t w = 1;
            for (int j = start; j < start + half; ++j) {
                uint64_t t = sp_ntt_mulmod(w, a[j + half]);
                a[j + half] = sp_ntt_submod(a[j], t);
                a[j]        = sp_ntt_addmod(a[j], t);
                w = sp_ntt_mulmod(w, w_step);
            }
        }
    }
}

static void ntt_cyclic_inverse(uint64_t* a) {
    bitrev_permute(a);
    for (int length = 2; length <= SP_NTT_N; length <<= 1) {
        uint64_t w_step = 1;
        {
            uint64_t base = SP_NTT_OMEGA_INV;
            uint64_t exp = SP_NTT_N / (uint64_t)length;
            while (exp > 0) {
                if (exp & 1) w_step = sp_ntt_mulmod(w_step, base);
                base = sp_ntt_mulmod(base, base);
                exp >>= 1;
            }
        }
        int half = length / 2;
        for (int start = 0; start < SP_NTT_N; start += length) {
            uint64_t w = 1;
            for (int j = start; j < start + half; ++j) {
                uint64_t t = sp_ntt_mulmod(w, a[j + half]);
                a[j + half] = sp_ntt_submod(a[j], t);
                a[j]        = sp_ntt_addmod(a[j], t);
                w = sp_ntt_mulmod(w, w_step);
            }
        }
    }
    // Scale by N_inv.
    for (int i = 0; i < SP_NTT_N; ++i) {
        a[i] = sp_ntt_mulmod(a[i], SP_NTT_N_INV);
    }
}

// ----- Public API: negacyclic forward / inverse + pre/post twist ----------

void sp_ntt_forward(uint64_t a[SP_NTT_N]) {
    // Pre-twist: a[i] *= psi^i.
    for (int i = 0; i < SP_NTT_N; ++i) {
        a[i] = sp_ntt_mulmod(a[i], sp_ntt_psi_pow[i]);
    }
    ntt_cyclic_forward(a);
}

void sp_ntt_inverse(uint64_t a[SP_NTT_N]) {
    ntt_cyclic_inverse(a);
    // Post-twist: a[i] *= psi^-i.
    for (int i = 0; i < SP_NTT_N; ++i) {
        a[i] = sp_ntt_mulmod(a[i], sp_ntt_psi_inv_pow[i]);
    }
}

void sp_ntt_pointwise_mul(uint64_t c[SP_NTT_N],
                          const uint64_t a[SP_NTT_N],
                          const uint64_t b[SP_NTT_N]) {
    for (int i = 0; i < SP_NTT_N; ++i) {
        c[i] = sp_ntt_mulmod(a[i], b[i]);
    }
}

// ----- Bridge to sp_poly (signed int64 ↔ unsigned mod-q uint64) -----------

void sp_ntt_coeffs_from_int64(uint64_t* out, const int64_t* in, int len) {
    const uint64_t Q = SP_NTT_Q;
    for (int i = 0; i < len; ++i) {
        int64_t v = in[i];
        if (v >= 0) {
            // Positive: reduce mod q.
            uint64_t uv = (uint64_t)v;
            out[i] = (uv >= Q) ? (uv % Q) : uv;
        } else {
            // Negative: add q until positive (worst case in our use: v near
            // -2^62, so |v| < 8*q; a few subtractions). Use modulo for safety.
            uint64_t mag = (uint64_t)(-v);
            uint64_t r = (mag >= Q) ? (mag % Q) : mag;
            out[i] = (r == 0) ? 0 : (Q - r);
        }
    }
}

void sp_ntt_coeffs_to_int64(int64_t* out, const uint64_t* in, int len) {
    const uint64_t HALF = SP_NTT_Q >> 1;
    for (int i = 0; i < len; ++i) {
        uint64_t u = in[i];
        if (u > HALF) {
            // Map to negative: signed value = u - q.
            out[i] = -(int64_t)(SP_NTT_Q - u);
        } else {
            out[i] = (int64_t)u;
        }
    }
}

/* ----- sp_poly-compatible NTT multiply ----------------------------------- */

int sp_poly_mul_ntt(sp_poly* c, const sp_poly* a, const sp_poly* b,
                    uint64_t* A_buf, uint64_t* B_buf, uint64_t* C_buf) {
    if (a->N != SP_NTT_N || b->N != SP_NTT_N || c->N != SP_NTT_N) {
        return -1;
    }
    sp_ntt_coeffs_from_int64(A_buf, a->coeffs, SP_NTT_N);
    sp_ntt_coeffs_from_int64(B_buf, b->coeffs, SP_NTT_N);
    sp_ntt_forward(A_buf);
    sp_ntt_forward(B_buf);
    sp_ntt_pointwise_mul(C_buf, A_buf, B_buf);
    sp_ntt_inverse(C_buf);
    sp_ntt_coeffs_to_int64(c->coeffs, C_buf, SP_NTT_N);
    return 0;
}

/* ----- NTT-backed CKKS dot product --------------------------------------- */

float sp_poly_dot_product_ntt(const float* q_vec, const float* k_vec,
                              int d, double delta,
                              int64_t* int_scratch,
                              uint64_t* u64_scratch,
                              int* ok) {
    if (d > SP_NTT_N) {
        if (ok) *ok = 0;
        return 0.0f;
    }
    int64_t* Q_int = int_scratch;
    int64_t* K_int = int_scratch + SP_NTT_N;
    uint64_t* A_buf = u64_scratch;
    uint64_t* B_buf = u64_scratch + SP_NTT_N;
    uint64_t* C_buf = u64_scratch + 2 * SP_NTT_N;

    sp_poly Qp = { Q_int, SP_NTT_N };
    sp_poly Kp = { K_int, SP_NTT_N };
    sp_poly_zero(&Qp);
    sp_poly_zero(&Kp);
    sp_poly_encode_fp32(&Qp, q_vec, d, delta, /*reversed=*/false);
    sp_poly_encode_fp32(&Kp, k_vec, d, delta, /*reversed=*/true);

    sp_ntt_coeffs_from_int64(A_buf, Q_int, SP_NTT_N);
    sp_ntt_coeffs_from_int64(B_buf, K_int, SP_NTT_N);
    sp_ntt_forward(A_buf);
    sp_ntt_forward(B_buf);
    sp_ntt_pointwise_mul(C_buf, A_buf, B_buf);
    sp_ntt_inverse(C_buf);

    int64_t coeff;
    {
        const uint64_t u = C_buf[d - 1];
        const uint64_t HALF = SP_NTT_Q >> 1;
        coeff = (u > HALF) ? -(int64_t)(SP_NTT_Q - u) : (int64_t)u;
    }
    if (ok) *ok = 1;
    return (float)((double)coeff / (delta * delta));
}

/* ----- Q-hoisted helpers (Phase 5b) -------------------------------------- */

void sp_poly_encode_ntt_q(uint64_t* Q_ntt,
                          const float* q_vec, int d, double delta,
                          int64_t* int_scratch) {
    sp_poly Qp = { int_scratch, SP_NTT_N };
    sp_poly_zero(&Qp);
    sp_poly_encode_fp32(&Qp, q_vec, d, delta, /*reversed=*/false);
    sp_ntt_coeffs_from_int64(Q_ntt, int_scratch, SP_NTT_N);
    sp_ntt_forward(Q_ntt);
}

float sp_poly_dot_product_ntt_q_cached(const uint64_t* Q_ntt,
                                       const float* k_vec, int d, double delta,
                                       int64_t* k_int_scratch,
                                       uint64_t* k_ntt_scratch,
                                       uint64_t* c_ntt_scratch,
                                       int* ok) {
    if (d > SP_NTT_N) {
        if (ok) *ok = 0;
        return 0.0f;
    }
    sp_poly Kp = { k_int_scratch, SP_NTT_N };
    sp_poly_zero(&Kp);
    sp_poly_encode_fp32(&Kp, k_vec, d, delta, /*reversed=*/true);
    sp_ntt_coeffs_from_int64(k_ntt_scratch, k_int_scratch, SP_NTT_N);
    sp_ntt_forward(k_ntt_scratch);
    sp_ntt_pointwise_mul(c_ntt_scratch, Q_ntt, k_ntt_scratch);
    sp_ntt_inverse(c_ntt_scratch);
    const uint64_t u = c_ntt_scratch[d - 1];
    const uint64_t HALF = SP_NTT_Q >> 1;
    const int64_t coeff = (u > HALF) ? -(int64_t)(SP_NTT_Q - u) : (int64_t)u;
    if (ok) *ok = 1;
    return (float)((double)coeff / (delta * delta));
}

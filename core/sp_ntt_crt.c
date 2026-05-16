/* sp_ntt_crt.c — Phase 9 dual-prime CRT-sharded NTT implementation.
 * Math verified end-to-end in Python (scripts/gen_ntt_crt_consts.py +
 * test_sp_ntt_crt.cpp). All multiplications stay in uint64.
 */

#include "sp_ntt_crt.h"
#include "sp_ntt_crt_consts.h"

#include <string.h>

/* Predefined contexts for the two production primes. */
const sp_ntt_ctx SP_NTT_CRT_CTX_Q1 = {
    SP_NTT_CRT_Q1,
    SP_NTT_CRT_N_INV1,
    sp_ntt_crt_psi_pow1,
    sp_ntt_crt_psi_inv_pow1,
    SP_NTT_CRT_PSI1,
    SP_NTT_CRT_PSI_INV1
};

const sp_ntt_ctx SP_NTT_CRT_CTX_Q2 = {
    SP_NTT_CRT_Q2,
    SP_NTT_CRT_N_INV2,
    sp_ntt_crt_psi_pow2,
    sp_ntt_crt_psi_inv_pow2,
    SP_NTT_CRT_PSI2,
    SP_NTT_CRT_PSI_INV2
};

/* Plain modular multiply.  (q-1)^2 < 2^60 < 2^64, so a single uint64
 * multiply followed by % q is fully native — no __int128 anywhere. */
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

/* Modular exponentiation for the w_step at each butterfly layer. */
static uint64_t powmod(uint64_t base, uint64_t exp, uint64_t q) {
    uint64_t r = 1;
    while (exp > 0) {
        if (exp & 1) r = mulmod(r, base, q);
        base = mulmod(base, base, q);
        exp >>= 1;
    }
    return r;
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

/* Cooley-Tukey radix-2 DIT NTT, natural-order in/out (bit-reverse
 * done internally). Operates in the field F_q via the supplied ctx. */
static void ntt_cyclic_fwd(uint64_t* a, const sp_ntt_ctx* ctx) {
    bitrev_permute(a);
    const uint64_t q = ctx->q;
    const uint64_t omega = mulmod(ctx->psi, ctx->psi, q);  /* psi^2 */
    for (int length = 2; length <= SP_NTT_CRT_N; length <<= 1) {
        uint64_t w_step = powmod(omega, (uint64_t)(SP_NTT_CRT_N / length), q);
        int half = length / 2;
        for (int start = 0; start < SP_NTT_CRT_N; start += length) {
            uint64_t w = 1;
            for (int j = start; j < start + half; ++j) {
                uint64_t t = mulmod(w, a[j + half], q);
                a[j + half] = submod(a[j], t, q);
                a[j]        = addmod(a[j], t, q);
                w = mulmod(w, w_step, q);
            }
        }
    }
}

static void ntt_cyclic_inv(uint64_t* a, const sp_ntt_ctx* ctx) {
    bitrev_permute(a);
    const uint64_t q = ctx->q;
    const uint64_t omega_inv =
        mulmod(ctx->psi_inv, ctx->psi_inv, q);  /* psi^-2 */
    for (int length = 2; length <= SP_NTT_CRT_N; length <<= 1) {
        uint64_t w_step = powmod(omega_inv, (uint64_t)(SP_NTT_CRT_N / length), q);
        int half = length / 2;
        for (int start = 0; start < SP_NTT_CRT_N; start += length) {
            uint64_t w = 1;
            for (int j = start; j < start + half; ++j) {
                uint64_t t = mulmod(w, a[j + half], q);
                a[j + half] = submod(a[j], t, q);
                a[j]        = addmod(a[j], t, q);
                w = mulmod(w, w_step, q);
            }
        }
    }
    /* Scale by N^-1. */
    for (int i = 0; i < SP_NTT_CRT_N; ++i) {
        a[i] = mulmod(a[i], ctx->n_inv, q);
    }
}

/* ---- Public API ------------------------------------------------------ */

void sp_ntt_crt_forward(uint64_t* a, const sp_ntt_ctx* ctx) {
    /* Pre-twist: a[i] *= psi^i mod q. */
    for (int i = 0; i < SP_NTT_CRT_N; ++i) {
        a[i] = mulmod(a[i], ctx->psi_pow[i], ctx->q);
    }
    ntt_cyclic_fwd(a, ctx);
}

void sp_ntt_crt_inverse(uint64_t* a, const sp_ntt_ctx* ctx) {
    ntt_cyclic_inv(a, ctx);
    for (int i = 0; i < SP_NTT_CRT_N; ++i) {
        a[i] = mulmod(a[i], ctx->psi_inv_pow[i], ctx->q);
    }
}

void sp_ntt_crt_pointwise_mul(uint64_t* c,
                              const uint64_t* a,
                              const uint64_t* b,
                              const sp_ntt_ctx* ctx) {
    const uint64_t q = ctx->q;
    for (int i = 0; i < SP_NTT_CRT_N; ++i) {
        c[i] = mulmod(a[i], b[i], q);
    }
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

    /* CRT-stitch and map to signed int64.
     * M = q1 * q2 fits in uint64 (60 bits). */
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
    /* Encode Q forward (not reversed) into an int64 polynomial of
     * length SP_NTT_CRT_N. Slots [0, d) hold round(q_vec[i] * delta);
     * slots [d, N) are zero. */
    sp_poly Qp = { int_scratch, SP_NTT_CRT_N };
    sp_poly_zero(&Qp);
    sp_poly_encode_fp32(&Qp, q_vec, d, delta, /*reversed=*/false);
    /* Lift into each prime universe and forward NTT. */
    sp_ntt_crt_coeffs_from_int64(Q_ntt_q1, int_scratch, SP_NTT_CRT_N,
                                 SP_NTT_CRT_Q1);
    sp_ntt_crt_coeffs_from_int64(Q_ntt_q2, int_scratch, SP_NTT_CRT_N,
                                 SP_NTT_CRT_Q2);
    sp_ntt_crt_forward(Q_ntt_q1, &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_forward(Q_ntt_q2, &SP_NTT_CRT_CTX_Q2);
}

void sp_poly_encode_ntt_k_reversed_crt(uint64_t* K_ntt_q1,
                                       uint64_t* K_ntt_q2,
                                       const float* k_vec, int d, double delta,
                                       int64_t* int_scratch) {
    /* Reversed K-encode: slot[d-1-i] = round(k_vec[i] * delta), zero elsewhere.
     * The reversal is what makes Σ q_i*k_i land at coefficient x^(d-1)
     * of Q(x)*K_rev(x). */
    sp_poly Kp = { int_scratch, SP_NTT_CRT_N };
    sp_poly_zero(&Kp);
    sp_poly_encode_fp32(&Kp, k_vec, d, delta, /*reversed=*/true);
    sp_ntt_crt_coeffs_from_int64(K_ntt_q1, int_scratch, SP_NTT_CRT_N,
                                 SP_NTT_CRT_Q1);
    sp_ntt_crt_coeffs_from_int64(K_ntt_q2, int_scratch, SP_NTT_CRT_N,
                                 SP_NTT_CRT_Q2);
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
    /* Pointwise multiply + inverse NTT in each prime universe. */
    sp_ntt_crt_pointwise_mul(c_q1_scratch, Q_ntt_q1, K_ntt_q1,
                             &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_pointwise_mul(c_q2_scratch, Q_ntt_q2, K_ntt_q2,
                             &SP_NTT_CRT_CTX_Q2);
    sp_ntt_crt_inverse(c_q1_scratch, &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_inverse(c_q2_scratch, &SP_NTT_CRT_CTX_Q2);
    /* Extract coefficient (d-1), CRT-stitch, map to signed. */
    const uint64_t u1 = c_q1_scratch[d - 1];
    const uint64_t u2 = c_q2_scratch[d - 1];
    const uint64_t x  = sp_ntt_crt_combine(u1, u2);
    const uint64_t M    = SP_NTT_CRT_Q1 * SP_NTT_CRT_Q2;
    const uint64_t HALF = M >> 1;
    const int64_t  coeff = (x > HALF) ? -(int64_t)(M - x) : (int64_t)x;
    if (ok) *ok = 1;
    return (float)((double)coeff / (delta * delta));
}

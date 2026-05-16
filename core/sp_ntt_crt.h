/* sp_ntt_crt.h — Dual-prime CRT-sharded NTT over Z_q1[x]/(x^N+1) x Z_q2[x]/(x^N+1).
 *
 * Phase 9: replaces the single 60-bit Proth prime path with two 30-bit
 * primes q1, q2 (combined modulus M = q1*q2 ~ 2^60). Every intermediate
 * product fits in uint64 (since (q-1)^2 < 2^60 < 2^64), eliminating
 * __int128 entirely. Portable to ARM64, RISC-V, GPU shaders, any
 * 64-bit ALU.
 *
 * Pipeline (per prime, q in {q1, q2}):
 *   forward(a_signed):
 *     1. lift each int64 a_i into [0, q)        (reduce mod q)
 *     2. pre-twist  a_i *= psi^i  mod q
 *     3. bit-rev permute
 *     4. Cooley-Tukey DIT NTT in omega = psi^2
 *   pointwise_mul(A, B):  C[i] = A[i] * B[i] mod q
 *   inverse(C):
 *     1. bit-rev permute
 *     2. inverse butterflies in omega^-1
 *     3. scale every coeff by N^-1
 *     4. post-twist a_i *= psi^-i mod q
 *
 * High-level negacyclic multiply (sp_ntt_crt_poly_mul):
 *   for q in {q1, q2}:
 *     run forward, pointwise_mul, inverse  →  coefficient vector c_q
 *   for each coefficient i:
 *     x = CRT-stitch(c_q1[i], c_q2[i])     →  unsigned, x ∈ [0, M)
 *     out[i] = (x > M/2) ? x - M : x       →  signed int64
 */

#ifndef SP_NTT_CRT_H
#define SP_NTT_CRT_H

#include <stdint.h>
#include <stddef.h>

#include "sp_poly_ring.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-prime parameter bundle. Two of these are predefined (q1, q2)
 * but the kernel takes them as parameters so future primes can be
 * dropped in without recompiling. */
typedef struct sp_ntt_ctx {
    uint64_t        q;             /* the prime modulus */
    uint64_t        n_inv;         /* N^-1 mod q */
    const uint64_t* psi_pow;       /* psi^i mod q,    length N */
    const uint64_t* psi_inv_pow;   /* psi^-i mod q,   length N */
    uint64_t        psi;           /* primitive 2N-th root of unity */
    uint64_t        psi_inv;       /* psi^-1 mod q */
} sp_ntt_ctx;

/* Pre-built contexts for the two production primes. */
extern const sp_ntt_ctx SP_NTT_CRT_CTX_Q1;
extern const sp_ntt_ctx SP_NTT_CRT_CTX_Q2;

/* In-place forward negacyclic NTT under the given prime context.
 *   - input:  signed int64 polynomial coefficients (length SP_NTT_CRT_N)
 *   - output: uint64 NTT-domain values in [0, ctx->q)
 * Caller-supplied workspace of length SP_NTT_CRT_N. */
void sp_ntt_crt_forward(uint64_t* a, const sp_ntt_ctx* ctx);

/* In-place inverse negacyclic NTT (mirror of forward). */
void sp_ntt_crt_inverse(uint64_t* a, const sp_ntt_ctx* ctx);

/* Pointwise multiply mod ctx->q. c[i] = a[i] * b[i] mod q. */
void sp_ntt_crt_pointwise_mul(uint64_t* c,
                              const uint64_t* a,
                              const uint64_t* b,
                              const sp_ntt_ctx* ctx);

/* Convert signed int64 coefficient buffer to reduced uint64 in [0, q). */
void sp_ntt_crt_coeffs_from_int64(uint64_t* out,
                                  const int64_t* in,
                                  int len,
                                  uint64_t q);

/* CRT stitch one coefficient: combine (a1 mod q1, a2 mod q2) into
 * x mod q1*q2 in [0, M). All intermediates fit in uint64. */
static inline uint64_t sp_ntt_crt_combine(uint64_t a1, uint64_t a2);

/* CRT-sharded negacyclic multiply c = a * b in Z[x]/(x^N + 1) where
 * inputs are signed int64 polynomials. Output is signed (mapped from
 * [0, M) via the M/2 threshold). Returns 0 on success, -1 on shape
 * mismatch (N must equal SP_NTT_CRT_N).
 *
 * Scratch layout: caller supplies 6 uint64 buffers of length N
 *   - workspace[0..N-1]      A_q1 (forward NTT of a mod q1)
 *   - workspace[N..2N-1]     B_q1 (forward NTT of b mod q1)
 *   - workspace[2N..3N-1]    C_q1 (product mod q1)
 *   - workspace[3N..4N-1]    A_q2
 *   - workspace[4N..5N-1]    B_q2
 *   - workspace[5N..6N-1]    C_q2
 */
int sp_ntt_crt_poly_mul(int64_t* c,
                        const int64_t* a,
                        const int64_t* b,
                        int N,
                        uint64_t* workspace);


/* Phase 9b engine integration helpers — match the Phase 5b/6 60-bit
 * helpers (sp_poly_encode_ntt_q, sp_poly_encode_ntt_k_reversed,
 * sp_poly_dot_product_ntt_qk_cached) but produce dual-universe output.
 *
 * Both Q_ntt_q1/Q_ntt_q2 buffers are uint64[SP_NTT_CRT_N]. The Q-encoder
 * writes both in one pass over the same fp32 input. The K-encoder
 * mirrors the reversed encoding used by sp_poly_dot_product (so the
 * dot product Σ q_i k_i lands at coefficient x^(d-1) of Q(x)*K_rev(x)).
 *
 * int_scratch is a single int64[SP_NTT_CRT_N] workspace shared across
 * the encode steps (encoder writes once; both NTT pipelines consume it).
 */
void sp_poly_encode_ntt_q_crt(uint64_t* Q_ntt_q1,
                              uint64_t* Q_ntt_q2,
                              const float* q_vec, int d, double delta,
                              int64_t* int_scratch);

void sp_poly_encode_ntt_k_reversed_crt(uint64_t* K_ntt_q1,
                                       uint64_t* K_ntt_q2,
                                       const float* k_vec, int d, double delta,
                                       int64_t* int_scratch);

/* CRT-stitched dot product: both Q and K are pre-transformed under
 * each prime context. This call is pure pointwise + inverse + extract +
 * CRT stitch + fp32 decode — the hot inner loop of CRT attention.
 *
 * Scratch: c_q1_scratch and c_q2_scratch each uint64[SP_NTT_CRT_N].
 * Returns 0 on success (always, given valid d). Sets *ok=1.
 */
float sp_poly_dot_product_ntt_crt_qk_cached(const uint64_t* Q_ntt_q1,
                                            const uint64_t* Q_ntt_q2,
                                            const uint64_t* K_ntt_q1,
                                            const uint64_t* K_ntt_q2,
                                            int d, double delta,
                                            uint64_t* c_q1_scratch,
                                            uint64_t* c_q2_scratch,
                                            int* ok);

/* Inline implementations. */
#include "sp_ntt_crt_consts.h"

static inline uint64_t sp_ntt_crt_combine(uint64_t a1, uint64_t a2) {
    /* diff = (a2 - a1) mod q2 */
    uint64_t diff = (a2 >= a1)
                       ? (a2 - a1)
                       : (SP_NTT_CRT_Q2 - (a1 - a2));
    /* u = (diff * crt_q1_inv_q2) mod q2.  diff < q2 (30 bits), the
     * inverse constant < q2 (30 bits), product < 2^60 fits uint64. */
    uint64_t u = (diff * SP_NTT_CRT_Q1_INV_Q2) % SP_NTT_CRT_Q2;
    /* x = a1 + u * q1.  u < q2 (30 bits), q1 < 2^30, product < 2^60,
     * plus a1 (30 bits) → fits uint64. */
    return a1 + u * SP_NTT_CRT_Q1;
}

#ifdef __cplusplus
}
#endif

#endif /* SP_NTT_CRT_H */

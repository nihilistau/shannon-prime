// Shannon-Prime — Number Theoretic Transform over Z_q[x] / (x^N + 1).
// Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
//
// Phase 4: accelerate the polynomial-ring attention bridge from O(N^2)
// to O(N log N) via NTT. The constants in sp_ntt_consts.h are tuned
// for N = 256 (fits head_dim up to 256, the Gemma3 size) and a 60-bit
// Proth prime q, leaving ~30 bits of headroom for the CKKS scale Δ.
//
// Algorithm (clean textbook layout; verified bit-exact vs O(N^2) baseline
// in tests/test_sp_ntt.cpp):
//
//   Forward negacyclic NTT:
//     1. pre-twist:  a[i] *= psi^i mod q                  (psi^N ≡ -1 mod q)
//     2. bit-rev:    a    = bitrev_permute(a)
//     3. NTT:        Cooley-Tukey radix-2 DIT, w_step = omega^(N/length)
//                    where omega = psi^2 is the N-th root of unity.
//
//   Inverse negacyclic NTT:
//     1. bit-rev permute
//     2. inverse NTT (same butterflies, w_step = omega_inv^(N/length))
//     3. multiply every coefficient by N_inv
//     4. post-twist: a[i] *= psi^-i mod q
//
//   Convolution in Z_q[x]/(x^N + 1):
//     c = INTT( NTT(a) * NTT(b) ) — pointwise multiply between the
//     two forward transforms.

#ifndef SP_NTT_H
#define SP_NTT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "sp_poly_ring.h"  // for sp_poly type

#ifdef __cplusplus
extern "C" {
#endif

// 60-bit modular multiply: returns (a * b) mod q, where q = SP_NTT_Q.
// Both inputs MUST be in [0, q).
uint64_t sp_ntt_mulmod(uint64_t a, uint64_t b);

// Modular add / sub with q = SP_NTT_Q. Inputs in [0, q); outputs in [0, q).
static inline uint64_t sp_ntt_addmod(uint64_t a, uint64_t b);
static inline uint64_t sp_ntt_submod(uint64_t a, uint64_t b);

// In-place forward negacyclic NTT over Z_q[x]/(x^N + 1).
// Input/output array length = SP_NTT_N (256). All values in [0, q).
void sp_ntt_forward(uint64_t a[/* SP_NTT_N */]);

// In-place inverse negacyclic NTT. Input/output length = SP_NTT_N.
void sp_ntt_inverse(uint64_t a[/* SP_NTT_N */]);

// Pointwise multiply in NTT domain: c[i] = a[i] * b[i] mod q.
void sp_ntt_pointwise_mul(uint64_t c[/* SP_NTT_N */],
                          const uint64_t a[/* SP_NTT_N */],
                          const uint64_t b[/* SP_NTT_N */]);

// Convert signed-int64 coefficient buffer to reduced uint64 in [0, q).
// Used to bridge sp_poly (int64) → NTT domain (uint64).
void sp_ntt_coeffs_from_int64(uint64_t* out,
                              const int64_t* in,
                              int len);

// Reverse direction: convert uint64 in [0, q) to int64 with signed
// representation (values > q/2 mapped to negative). Used to read back
// the NTT result into sp_poly's signed coefficient buffer.
void sp_ntt_coeffs_to_int64(int64_t* out,
                            const uint64_t* in,
                            int len);

// NTT-accelerated negacyclic multiply with sp_poly semantics. Matches
// sp_poly_mul's output bit-for-bit (after mod-q lift) when N == SP_NTT_N.
// Returns 0 on success, -1 if N != SP_NTT_N (caller must fall back).
//
// Scratch: caller provides three uint64 buffers of length SP_NTT_N
// (A_buf, B_buf, C_buf) for the NTT domain.
int sp_poly_mul_ntt(sp_poly* c, const sp_poly* a, const sp_poly* b,
                    uint64_t* A_buf, uint64_t* B_buf, uint64_t* C_buf);

// NTT-backed CKKS dot product: same semantics as sp_poly_dot_product
// (declared in sp_poly_ring.h), but uses the NTT path. Requires the ring
// degree N to equal SP_NTT_N; returns 0.0f and sets *ok=0 otherwise.
//
// Scratch layout: caller supplies
//   - int_scratch of length 2*SP_NTT_N (Q_int, K_int polynomials)
//   - u64_scratch of length 3*SP_NTT_N (A_buf, B_buf, C_buf NTT domain)
float sp_poly_dot_product_ntt(const float* q_vec, const float* k_vec,
                              int d, double delta,
                              int64_t* int_scratch,
                              uint64_t* u64_scratch,
                              int* ok);

/* Q-hoisted variant for repeated dot products with a fixed Q (Phase 5b).
 * Caller transforms Q once via sp_poly_encode_ntt_q (writes Q_ntt buffer of
 * length SP_NTT_N in the NTT domain) and reuses it for every key:
 *   sp_poly_encode_ntt_q(Q_ntt, q_vec, d, delta, int_scratch_qi);
 *   for (t in keys)
 *     score[t] = sp_poly_dot_product_ntt_q_cached(Q_ntt, k_vec_t, ...);
 */
void sp_poly_encode_ntt_q(uint64_t* Q_ntt,
                          const float* q_vec, int d, double delta,
                          int64_t* int_scratch);

float sp_poly_dot_product_ntt_q_cached(const uint64_t* Q_ntt,
                                       const float* k_vec, int d, double delta,
                                       int64_t* k_int_scratch,
                                       uint64_t* k_ntt_scratch,
                                       uint64_t* c_ntt_scratch,
                                       int* ok);


/* Phase 6: both Q and K pre-NTT'd. Just pointwise multiply + inverse NTT
 * + extract coefficient (d-1). Caller manages a persistent K_ntt cache
 * (one buffer of length SP_NTT_N per (kv_h, t)) and the Q_ntt buffer (one
 * per (h, qi) via sp_poly_encode_ntt_q). c_ntt_scratch is per-call. */
float sp_poly_dot_product_ntt_qk_cached(const uint64_t* Q_ntt,
                                        const uint64_t* K_ntt,
                                        int d, double delta,
                                        uint64_t* c_ntt_scratch);

/* Encode reversed K + forward NTT into the supplied K_ntt buffer of length
 * SP_NTT_N. Used to populate the Phase 6 K-cache once per (kv_h, t). */
void sp_poly_encode_ntt_k_reversed(uint64_t* K_ntt,
                                   const float* k_vec, int d, double delta,
                                   int64_t* int_scratch);

// Inline implementations of add/sub for tight inner loops.
#include "sp_ntt_consts.h"

static inline uint64_t sp_ntt_addmod(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    if (s >= SP_NTT_Q || s < a) s -= SP_NTT_Q;  // also guards against (rare) wrap
    return s;
}

static inline uint64_t sp_ntt_submod(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (a + SP_NTT_Q - b);
}

#ifdef __cplusplus
}
#endif

#endif  // SP_NTT_H


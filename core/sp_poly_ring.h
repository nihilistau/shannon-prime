// Shannon-Prime — polynomial ring R_q = Z_q[x] / (x^N + 1).
// Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
//
// Phase 3 pivot: replaces the Weil-pairing attention bridge with a
// CKKS-style integer polynomial dot product. Unlike the pairing path,
// this preserves the metric topology of R^d — scale-and-round into
// integer coefficients, then exploit polynomial convolution to compute
// q·k exactly (modulo the integer scale Δ²).
//
// Math identity we lean on:
//
//   Q(x)     = q_0 + q_1 x + ... + q_{d-1} x^{d-1}
//   K_rev(x) = k_{d-1} + k_{d-2} x + ... + k_0 x^{d-1}
//
//   Q(x) * K_rev(x)  has  Σ q_i · k_i  as the coefficient of x^{d-1}.
//
// All operations stay in the integer ring Z_q[x] / (x^N + 1) with N ≥ d
// chosen as a power of 2. The negacyclic reduction (x^N = -1) keeps the
// multiply closed.
//
// Numerical setup for the prototype: q a 64-bit prime ≈ 2^60 or larger
// so that |coeff| · N stays well under q (no overflow after one multiply).
// For 64-bit math without modular arithmetic we pick q = 2^62 — i.e.
// implicit wraparound — and verify the recovered dot product matches
// the fp32 reference within Δ²-ULP.

#ifndef SP_POLY_RING_H
#define SP_POLY_RING_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Coefficient type: int64 in the prototype. A real CKKS impl would use
// a modular reduction with a chosen prime q; for the in-engine attention
// proxy we let the int64 carry the full unreduced sum.
typedef int64_t sp_poly_coeff;

// Polynomial in Z_q[x] / (x^N + 1). The caller owns the coefficient
// buffer of length N.
typedef struct sp_poly {
    sp_poly_coeff* coeffs;   // length N, lowest-degree first
    int            N;        // ring degree (power of 2, N ≥ d)
} sp_poly;

// In-place zero out.
void sp_poly_zero(sp_poly* p);

// Set p[i] = coeffs[i] for i in [0, len). Remaining slots set to 0.
// len must be ≤ p->N.
void sp_poly_set_from_int64(sp_poly* p, const sp_poly_coeff* coeffs, int len);

// CKKS-style encoding from fp32 → integer polynomial.
//
//   p[i] = round(vec[i] * delta)        for i in [0, d)
//   p[i] = 0                            for i in [d, N)
//
// `delta` is the scale factor (e.g. 1<<14). Caller picks N as power of 2.
// `reversed` swaps the coefficient order so the dot product trick works:
//   forward:  p[i] = round(vec[i]     * delta)
//   reversed: p[i] = round(vec[d-1-i] * delta)
void sp_poly_encode_fp32(sp_poly* p, const float* vec, int d,
                          double delta, bool reversed);

// Decode coefficient `i` back to fp32 by dividing by delta_recovery.
// Used to extract the dot product from coeff[d-1] after multiplication:
//   dot ≈ coeffs[d-1] / (delta * delta)
float sp_poly_decode_coeff(const sp_poly* p, int i, double delta_recovery);

// c = a * b in R = Z[x] / (x^N + 1). All three polynomials have the
// same degree N. c may NOT alias a or b.
//
// Negacyclic convolution: c[k] = Σ_{i+j ≡ k mod N} (sign · a[i] b[j])
// where sign = -1 if i + j ≥ N (because x^N = -1 in the ring), else +1.
//
// Naive O(N^2). For the prototype this is fine; an NTT/FFT impl can
// drop in later by replacing this function.
void sp_poly_mul(sp_poly* c, const sp_poly* a, const sp_poly* b);

// Inner product convenience: given fp32 vectors q, k of length d and
// scale delta, returns the recovered dot product q·k as fp32.
//
// Internally: encode Q forward, K reversed; multiply; read coeff[d-1].
// Caller-supplied scratch buffer of size 3*N int64 (Q, K_rev, product).
float sp_poly_dot_product(const float* q, const float* k, int d,
                            int N, double delta,
                            sp_poly_coeff* scratch);

#ifdef __cplusplus
}
#endif

#endif  // SP_POLY_RING_H

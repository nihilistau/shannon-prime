// Shannon-Prime — elliptic curve arithmetic + Weil pairing.
// Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
//
// Phase 3: replaces the fp32 softmax bridge in attention with a pure
// integer Weil pairing on E[n] torsion. Reference: Paper A §9.2.
//
// Algebraic setup:
//   Curve   E: y^2 = x^3 + a*x + b over F_p
//   Group   E(F_p): affine points + the point at infinity O
//   E[n]    = { P in E(F_p) : n*P = O }  (n-torsion subgroup)
//   Pairing e_n: E[n] x E[n] -> mu_n     (n-th roots of unity in F_p)
//
//   Miller's algorithm: O(log n) curve ops + line evaluations.
//   Weil:   e_n(P, Q) = (-1)^n * f_{n,P}(Q) / f_{n,Q}(P)
//
// Numerical bounds: all arithmetic is mod p. For 64-bit ops to stay
// in int64 without overflow, p must satisfy 2*p^2 < 2^63 — i.e.
// p < 2^31 (about 2.1e9). For the Weil-pairing attention prototype
// we use small primes (p < 2^16) where head_dim positions can be
// safely mapped into F_p.
//
// Conventions:
//   - x = -1 sentinel for the point at infinity (any negative x).
//   - y is meaningful only when x >= 0.

#ifndef SP_EC_WEIL_H
#define SP_EC_WEIL_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sp_ec_point {
    int64_t x;   // -1 for the point at infinity
    int64_t y;
} sp_ec_point;

typedef struct sp_ec_curve {
    int64_t a;
    int64_t b;
    int64_t p;   // prime modulus
} sp_ec_curve;

static const sp_ec_point SP_EC_INFINITY = { -1, 0 };

// Field helpers. All inputs are taken mod p.
int64_t sp_ec_mod(int64_t x, int64_t p);
int64_t sp_ec_mod_pow(int64_t base, int64_t exp, int64_t p);
int64_t sp_ec_mod_inv(int64_t x, int64_t p);          // Fermat's little theorem

// Predicates.
bool sp_ec_is_infinity(sp_ec_point P);
bool sp_ec_eq(sp_ec_point P, sp_ec_point Q);
bool sp_ec_is_on_curve(const sp_ec_curve* E, sp_ec_point P);

// Group ops.
sp_ec_point sp_ec_neg(const sp_ec_curve* E, sp_ec_point P);
sp_ec_point sp_ec_add(const sp_ec_curve* E, sp_ec_point P, sp_ec_point Q);
sp_ec_point sp_ec_mul(const sp_ec_curve* E, int64_t k, sp_ec_point P);

// Order of a point (linear search up to max_order; returns -1 if not found).
int64_t sp_ec_order(const sp_ec_curve* E, sp_ec_point P, int64_t max_order);

// Miller's algorithm: returns f_{n, P}(Q) mod p. Caller must ensure
// Q != P and Q != O (the function diverges on its own support). Returns
// -1 on degeneracy (interpreted as "Miller undefined here").
int64_t sp_ec_miller(const sp_ec_curve* E, int64_t n,
                       sp_ec_point P, sp_ec_point Q);

// Weil pairing e_n(P, Q) for P, Q in E[n]. Returns a value in F_p
// representing an n-th root of unity. Returns 1 if P or Q is infinity
// or P == Q (alternating property).
int64_t sp_ec_weil_pairing(const sp_ec_curve* E, int64_t n,
                              sp_ec_point P, sp_ec_point Q);

#ifdef __cplusplus
}
#endif

#endif  // SP_EC_WEIL_H

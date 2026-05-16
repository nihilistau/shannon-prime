// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

//
// sp_ok_arith — Integer arithmetic in O_K = Z[omega], where
//   omega = (1 + sqrt(-163)) / 2,    omega^2 = omega - 41.
//
// This is the ground-truth integer ring underlying Paper A §3 / §6 / Paper C §3.
// All operations are exact int64 arithmetic; the bit-exact contract is enforced
// by the Shannon-Prime Test Suite (Python oracle at test-suite/src/sp_algebra.py).
//
// Overflow: int64 has range ~9.2e18; a single multiplication can grow operands
// by ~41. For state magnitudes < 2^20, a chain of 16 multiplications stays
// within int64. For larger states or longer chains, the int128 path (below)
// should be used.

#ifndef SHANNON_PRIME_OK_ARITH_H
#define SHANNON_PRIME_OK_ARITH_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Element of O_K represented as a + b*omega with a, b in Z.
typedef struct {
    int64_t a;  // coefficient of 1
    int64_t b;  // coefficient of omega
} sp_ok_t;

// Constants (header-defined so they can be used as compile-time initializers).
#define SP_OK_ZERO    ((sp_ok_t){0, 0})
#define SP_OK_ONE     ((sp_ok_t){1, 0})
#define SP_OK_OMEGA   ((sp_ok_t){0, 1})

// The relation omega^2 = omega - 41 is encoded in the multiplication formula.
#define SP_OK_OMEGA_NORM  41

// --- Basic arithmetic -----------------------------------------------------

static inline sp_ok_t sp_ok_add(sp_ok_t x, sp_ok_t y) {
    sp_ok_t r = { x.a + y.a, x.b + y.b };
    return r;
}

static inline sp_ok_t sp_ok_sub(sp_ok_t x, sp_ok_t y) {
    sp_ok_t r = { x.a - y.a, x.b - y.b };
    return r;
}

static inline sp_ok_t sp_ok_neg(sp_ok_t x) {
    sp_ok_t r = { -x.a, -x.b };
    return r;
}

// (a1 + b1 w)(a2 + b2 w) = (a1 a2 - 41 b1 b2) + (a1 b2 + a2 b1 + b1 b2) w
// using w^2 = w - 41.
static inline sp_ok_t sp_ok_mul(sp_ok_t x, sp_ok_t y) {
    sp_ok_t r;
    r.a = x.a * y.a - SP_OK_OMEGA_NORM * x.b * y.b;
    r.b = x.a * y.b + y.a * x.b + x.b * y.b;
    return r;
}

static inline sp_ok_t sp_ok_scalar_mul(sp_ok_t x, int64_t s) {
    sp_ok_t r = { x.a * s, x.b * s };
    return r;
}

// Conjugate: (a + b w)_bar = (a + b) + (-b) w  (since w_bar = 1 - w).
static inline sp_ok_t sp_ok_conjugate(sp_ok_t x) {
    sp_ok_t r = { x.a + x.b, -x.b };
    return r;
}

// Norm N(a + b w) = a^2 + a b + 41 b^2 = (a + b w)(a + b w_bar).
static inline int64_t sp_ok_norm(sp_ok_t x) {
    return x.a * x.a + x.a * x.b + SP_OK_OMEGA_NORM * x.b * x.b;
}

// Trace Tr(a + b w) = 2 a + b.
static inline int64_t sp_ok_trace(sp_ok_t x) {
    return 2 * x.a + x.b;
}

static inline bool sp_ok_equal(sp_ok_t x, sp_ok_t y) {
    return x.a == y.a && x.b == y.b;
}

// --- Exponentiation -------------------------------------------------------

// x^k via square-and-multiply. k >= 0. Overflow is caller's responsibility.
sp_ok_t sp_ok_pow(sp_ok_t x, int64_t k);

// Reduce coordinates modulo p (for arithmetic in O_K / p O_K).
static inline sp_ok_t sp_ok_mod(sp_ok_t x, int64_t p) {
    int64_t a = x.a % p;
    int64_t b = x.b % p;
    if (a < 0) a += p;
    if (b < 0) b += p;
    sp_ok_t r = { a, b };
    return r;
}

#ifdef __cplusplus
}
#endif

#endif  // SHANNON_PRIME_OK_ARITH_H

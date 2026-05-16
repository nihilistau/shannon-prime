// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

//
// sp_frobenius.c — Frobenius endomorphism for K = Q(sqrt(-163)).
//
// Bit-exact reference: test-suite/src/sp_algebra.py + engine_hooks2.py.

#include "sp_frobenius.h"
#include "sp_ok_arith.h"
#include <stddef.h>

static const int64_t SP_DISCRIMINANT = -163;

// ============================================================================
// Modular exponentiation (for Legendre symbol via Euler's criterion).
// ============================================================================

static int64_t sp_powmod_i64(int64_t base, int64_t exp, int64_t mod) {
    int64_t result = 1 % mod;
    base = base % mod;
    if (base < 0) base += mod;
    while (exp > 0) {
        if (exp & 1) result = (result * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return result;
}

// ============================================================================
// Prime classification (Lemma 3.1 of Paper C).
// ============================================================================

int sp_legendre_neg163(int64_t p) {
    int64_t n = SP_DISCRIMINANT % p;
    if (n < 0) n += p;
    if (n == 0) return 0;
    int64_t r = sp_powmod_i64(n, (p - 1) / 2, p);
    return (r <= 1) ? (int)r : (int)(r - p);
}

bool sp_is_inert(int64_t p) {
    if (p == 2) return true;          // -163 mod 8 = 5
    if (p == 163) return false;        // ramified
    return sp_legendre_neg163(p) == -1;
}

bool sp_is_split(int64_t p) {
    if (p == 2 || p == 163) return false;
    return sp_legendre_neg163(p) == 1;
}

bool sp_is_ramified(int64_t p) {
    return p == 163;
}

// ============================================================================
// Find element of given norm by exhaustive search over small b.
// ============================================================================

static int64_t sp_isqrt_i64(int64_t n) {
    if (n < 2) return n;
    int64_t x = (int64_t)((double)n / 2.0 + 0.5);
    if (x <= 0) x = 1;
    for (int i = 0; i < 80; i++) {
        int64_t nx = (x + n / x) / 2;
        if (nx == x || nx == x - 1 || nx == x + 1) {
            // converged; pick correct floor
            while (nx * nx > n) nx--;
            while ((nx + 1) * (nx + 1) <= n) nx++;
            return nx;
        }
        x = nx;
    }
    while (x * x > n) x--;
    while ((x + 1) * (x + 1) <= n) x++;
    return x;
}

bool sp_find_element_of_norm(int64_t n, sp_ok_t *out) {
    if (n < 0 || out == NULL) return false;
    // N(a + b w) = a^2 + a b + 41 b^2 = n
    // bound on |b|: 4n - 163 b^2 >= 0 -> b^2 <= 4 n / 163
    int64_t b_bound = sp_isqrt_i64((4 * n) / 163) + 1;
    for (int64_t b = -b_bound; b <= b_bound; b++) {
        int64_t disc = 4 * n - 163 * b * b;
        if (disc < 0) continue;
        int64_t s = sp_isqrt_i64(disc);
        if (s * s != disc) continue;
        // Iterate sign +1 first then -1 to match the Python oracle's
        // canonical representative (test-suite/src/sp_algebra.py).
        // Bit-exact contract requires the same choice of pi vs pi_bar.
        for (int sign_step = 0; sign_step < 2; sign_step++) {
            int sign = (sign_step == 0) ? +1 : -1;
            int64_t num = -b + sign * s;
            if (num & 1) continue;       // odd numerator -> a not integer
            int64_t a = num / 2;
            sp_ok_t el = { a, b };
            if (sp_ok_norm(el) == n) {
                *out = el;
                return true;
            }
        }
    }
    return false;
}

// ============================================================================
// Frobenius application (Theorem 4 of Paper A, Theorem 2 of Paper C).
// ============================================================================

sp_ok_t sp_frobenius_pi_41(void) {
    sp_ok_t pi;
    // 41 is split; find_element_of_norm returns a representative.
    if (sp_find_element_of_norm(41, &pi)) return pi;
    // Fallback (should never happen): omega itself has norm 41.
    sp_ok_t fallback = { 0, 1 };
    return fallback;
}

sp_ok_t sp_apply_frobenius(sp_ok_t state, int64_t p, int64_t k) {
    if (k == 0) return state;
    if (sp_is_ramified(p)) return SP_OK_ZERO;

    if (sp_is_inert(p)) {
        if (k & 1) return SP_OK_ZERO;  // odd power of phi_p not in O_K
        // phi_p^(2m) = (-p)^m as a scalar acting on state
        int64_t m = k / 2;
        int64_t scalar = 1;
        int64_t base = -p;
        while (m > 0) {
            if (m & 1) scalar *= base;
            base *= base;
            m >>= 1;
        }
        return sp_ok_scalar_mul(state, scalar);
    }
    // split case: phi_p = pi where N(pi) = p
    sp_ok_t pi;
    if (!sp_find_element_of_norm(p, &pi)) return SP_OK_ZERO;
    sp_ok_t pi_pow = sp_ok_pow(pi, k);
    return sp_ok_mul(state, pi_pow);
}

// ============================================================================
// Tensor-level operations.
// ============================================================================

void sp_frobenius_quant_tensor(sp_ok_t *state, size_t n_elements,
                               int64_t p, int64_t k) {
    if (state == NULL || n_elements == 0) return;
    // For split p, precompute pi^k once.
    if (sp_is_split(p)) {
        sp_ok_t pi;
        if (!sp_find_element_of_norm(p, &pi)) return;
        sp_ok_t pi_pow = sp_ok_pow(pi, k);
        for (size_t i = 0; i < n_elements; i++) {
            state[i] = sp_ok_mul(state[i], pi_pow);
        }
        return;
    }
    // Inert case: scalar action.
    if (sp_is_inert(p)) {
        if (k & 1) return;
        int64_t m = k / 2;
        int64_t scalar = 1;
        int64_t base = -p;
        while (m > 0) {
            if (m & 1) scalar *= base;
            base *= base;
            m >>= 1;
        }
        for (size_t i = 0; i < n_elements; i++) {
            state[i] = sp_ok_scalar_mul(state[i], scalar);
        }
    }
    // ramified: no-op
}

void sp_sato_tate_mix_tensor(sp_ok_t *state, size_t n_elements,
                              int64_t p1, int64_t k1,
                              int64_t p2, int64_t k2) {
    // Apply inert channel first by convention (zero-drift, scalar).
    sp_frobenius_quant_tensor(state, n_elements, p1, k1);
    sp_frobenius_quant_tensor(state, n_elements, p2, k2);
}

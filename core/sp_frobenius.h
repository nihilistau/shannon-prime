// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

//
// sp_frobenius — Frobenius endomorphism, prime classification in K = Q(sqrt(-163)).
//
// This module implements Theorem 4 of Paper A and Theorem 2 of Paper C.
// For inert primes p (Deuring): a_p = 0, phi_p^2 = -p in O_K.
// For split primes p:           a_p = pi + pi_bar where N(pi) = p, pi in O_K.
//
// The split-prime constants for p = 41 (smallest split prime) are precomputed.

#ifndef SHANNON_PRIME_FROBENIUS_H
#define SHANNON_PRIME_FROBENIUS_H

#include "sp_ok_arith.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Production prime parameters (Paper D v0.3, Config B and Config E).
#define SP_P1_INERT     2     // phi_2^2 = -2; zero-drift skeleton
#define SP_P2_SPLIT     41    // smallest split prime; a_41 = +/- 1

// --- Prime classification -------------------------------------------------

// Compute the Legendre symbol (-163 / p) for odd prime p. Returns -1, 0, +1.
int sp_legendre_neg163(int64_t p);

// True iff p is inert in K = Q(sqrt(-163)).
//   p = 2:   true   (-163 mod 8 = 5)
//   p = 163: false  (ramified)
//   else:    true iff (-163 / p) = -1
bool sp_is_inert(int64_t p);

// True iff p splits in K.
bool sp_is_split(int64_t p);

// True iff p is ramified in K (only p = 163).
bool sp_is_ramified(int64_t p);

// --- Element of given norm ------------------------------------------------

// Find alpha in O_K with N(alpha) = n. Sets *out and returns true on success;
// returns false (out unchanged) if no such element exists.
// For split prime p, this returns a valid Frobenius element pi.
bool sp_find_element_of_norm(int64_t n, sp_ok_t *out);

// --- Frobenius application ------------------------------------------------

// Apply phi_p^k to a state element (Theorem 4 of Paper A).
//
// For inert p: k must be EVEN; result is state * (-p)^(k/2) (scalar action,
// zero drift in the omega-direction). Asserts (and returns SP_OK_ZERO on
// violation in release builds).
//
// For split p: phi_p = pi where N(pi) = p; result is state * pi^k.
//
// For ramified p (=163): not supported; returns SP_OK_ZERO.
sp_ok_t sp_apply_frobenius(sp_ok_t state, int64_t p, int64_t k);

// Cached Frobenius element for p = 41 (split, the production split prime).
// Either omega = (0, 1) or omega_bar = (1, -1); both have norm 41. The
// implementation picks the canonical representative returned by
// sp_find_element_of_norm(41); callers should NOT depend on the choice
// — the bit-exact contract is via norm invariant (see test_sp_frobenius).
sp_ok_t sp_frobenius_pi_41(void);

// --- Tensor-level operations (Paper D §4.1 implementation) ----------------

// --frobenius-quant: apply phi_p^k to every element of a tensor.
// state[i] = phi_p^k(state[i]) for i in [0, n_elements).
//
// For Config B in Paper D: p = SP_P2_SPLIT = 41, k = 8.
void sp_frobenius_quant_tensor(sp_ok_t *state, size_t n_elements,
                               int64_t p, int64_t k);

// --sato-tate-mix: apply phi_p1^k1 ∘ phi_p2^k2 to every element.
// By commutativity of O_K, order is irrelevant; we apply inert first
// (zero-drift channel) by convention.
//
// For Config E in Paper D: p1 = 2 inert, k1 = 2; p2 = 41 split, k2 = 8.
void sp_sato_tate_mix_tensor(sp_ok_t *state, size_t n_elements,
                              int64_t p1, int64_t k1,
                              int64_t p2, int64_t k2);

#ifdef __cplusplus
}
#endif

#endif  // SHANNON_PRIME_FROBENIUS_H

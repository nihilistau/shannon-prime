/* sp_kste_ramanujan.c — Phase 9 Ramanujan-Fourier KSTE modulation.
 *
 * Injects position-aware coprimality structure into the K-vector before
 * the KSTE encoder runs, by adding a small Ramanujan-sum bias to each
 * component.  Two K-vectors at the same continuous coordinates but
 * different integer positions then produce different packed trees,
 * because their VHT2-reordered spectra differ by the position-dependent
 * Ramanujan term.
 *
 * MATH (Paper IV §10+).  Ramanujan's sum is
 *     c_q(p) = Σ_{a ∈ (Z/qZ)*} exp(2πi · a · p / q)
 * Real-valued, bounded by Euler's totient: |c_q(p)| ≤ φ(q).  Kluyver's
 * theorem (1906) gives a direct combinatorial formula:
 *     c_q(p) = Σ_{d | gcd(p,q)} μ(q/d) · d
 * where μ is the Möbius function.  We use the Kluyver form because it
 * is exact, integer-valued, and computable in O(σ_0(gcd(p,q))) ops.
 *
 * The Ramanujan-Fourier expansion of σ(n)/n (sum-of-divisors / n) is
 *     σ(p)/p = ζ(2) · Σ_{q=1..∞} c_q(p) / q²
 * with ζ(2) = π²/6.  The 1/q² weighting gives absolute convergence and
 * means the highest-frequency components are naturally attenuated.
 * We use a small q-bank {2, 3, 5, 6, 10} weighted by 1/q² and apply
 * the result as an additive perturbation to K:
 *     K'[i] = K[i] + λ · c_{q_i}(p) / q_i²
 * where q_i = BANK[i mod |BANK|] cycles through the bank across dims.
 *
 * Properties:
 *   - λ == 0 makes this a no-op (preserves Phase 8 baseline exactly).
 *   - Bounded perturbation: |perturbation_i| ≤ λ · φ(q_max) / q_min² ≤
 *     λ · 4 / 4 = λ for our bank.
 *   - Position-distinguishing: c_q values change between p and p+1 in
 *     a pattern that depends on the coprimality structure (e.g.,
 *     c_2(244) - c_2(245) = 2 because 244 is even, 245 is odd).
 *   - Pure C, no __int128, no overflow possible (all arithmetic on
 *     small ints + a single fp32 multiply per dim).
 *
 * Cost: ~O(N_Q · σ_0(gcd)) ≈ O(20) ops to compute the bank values
 * once, then O(head_dim) fp32 adds.  Negligible vs the VHT2 cost.
 */

#include "sp_kste.h"

#include <stdint.h>
#include <stddef.h>

/* ---------- Möbius function for q ≤ 2^31 ------------------------------ */

static int sp_kste_mobius(int q)
{
    if (q <= 0)  return 0;
    if (q == 1)  return 1;
    int result = 1;
    /* Trial-divide by primes up to sqrt(q).  For our bank q ≤ 10 the
     * cost is bounded by ~3 iterations.  Even at q=100 it's < 10. */
    for (int p = 2; (int64_t)p * p <= (int64_t)q; ++p) {
        if (q % p == 0) {
            q /= p;
            if (q % p == 0) return 0;   /* p² divides q ⇒ μ(q) = 0 */
            result = -result;
        }
    }
    if (q > 1) result = -result;        /* the leftover prime factor */
    return result;
}

/* ---------- gcd ------------------------------------------------------- */

static int sp_kste_gcd(int a, int b)
{
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/* ---------- c_q(p) via Kluyver's theorem ------------------------------ */

static int sp_kste_cq(int q, int p)
{
    if (q <= 0) return 0;
    /* gcd(0, q) = q, so c_q(0) = Σ_{d|q} μ(q/d)·d = φ(q) by a classical
     * identity; we handle p=0 explicitly to avoid the edge case below. */
    int g = sp_kste_gcd(p < 0 ? -p : p, q);
    int sum = 0;
    /* Walk divisors of g.  σ_0(g) ≤ σ_0(q) which is tiny for q ≤ 10. */
    for (int d = 1; (int64_t)d * d <= (int64_t)g; ++d) {
        if (g % d == 0) {
            sum += sp_kste_mobius(q / d) * d;
            int dd = g / d;
            if (dd != d) sum += sp_kste_mobius(q / dd) * dd;
        }
    }
    return sum;
}

/* ---------- Q-bank ---------------------------------------------------- *
 * Chosen for fast convergence (small q, large 1/q² weight) and for
 * structural complementarity:
 *   q=2  : even/odd parity                (period 2)
 *   q=3  : divisibility-by-3 indicator    (period 3)
 *   q=5  : divisibility-by-5 indicator    (period 5)
 *   q=6  : composite divisibility (2&3)   (period 6, captures cross-term)
 *   q=10 : composite divisibility (2&5)   (period 10)
 *
 * gcd(p, q) for q in this bank decomposes p's prime structure into
 * its small-prime fingerprint, which is exactly the discriminator the
 * KSTE encoder needs for position-aware uniqueness on natural-language
 * token streams (which rarely have a periodicity > ~10).
 *
 * Bank size and contents are fixed at compile time; the only runtime
 * parameter is λ.
 */
#define SP_KSTE_RAMANUJAN_BANK_SIZE 5
static const int sp_kste_ramanujan_bank[SP_KSTE_RAMANUJAN_BANK_SIZE] = {
    2, 3, 5, 6, 10
};

/* Pre-computed inverse squares used for the 1/q² weighting.  fp32 is
 * sufficient: the largest is 1/4 = 0.25, the smallest is 1/100 = 0.01. */
static const float sp_kste_ramanujan_inv_q2[SP_KSTE_RAMANUJAN_BANK_SIZE] = {
    1.0f / 4.0f,    /*  q=2  */
    1.0f / 9.0f,    /*  q=3  */
    1.0f / 25.0f,   /*  q=5  */
    1.0f / 36.0f,   /*  q=6  */
    1.0f / 100.0f   /*  q=10 */
};

/* ---------- Public API ------------------------------------------------ */

void sp_kste_ramanujan_modulate(float *K, int head_dim,
                                int position, float lambda)
{
    if (!K || head_dim <= 0) return;
    if (lambda == 0.0f) return;          /* preserve no-op semantics */

    /* Pre-compute c_q(position) / q² for the whole bank. */
    float cq_weighted[SP_KSTE_RAMANUJAN_BANK_SIZE];
    for (int j = 0; j < SP_KSTE_RAMANUJAN_BANK_SIZE; ++j) {
        int q = sp_kste_ramanujan_bank[j];
        int cq = sp_kste_cq(q, position);
        cq_weighted[j] = (float)cq * sp_kste_ramanujan_inv_q2[j];
    }

    /* Modulate each dim with the bank entry chosen by (i mod |BANK|). */
    for (int i = 0; i < head_dim; ++i) {
        K[i] += lambda * cq_weighted[i % SP_KSTE_RAMANUJAN_BANK_SIZE];
    }
}

/* Test-only access (used by test_sp_kste_ramanujan if added). */
int sp_kste_ramanujan_cq_for_test(int q, int p)
{
    return sp_kste_cq(q, p);
}

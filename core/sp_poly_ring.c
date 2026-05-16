// Shannon-Prime — polynomial ring R_q = Z_q[x] / (x^N + 1) (impl).
// Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.

#include "sp_poly_ring.h"

#include <math.h>
#include <string.h>

void sp_poly_zero(sp_poly* p) {
    if (!p || !p->coeffs) return;
    for (int i = 0; i < p->N; ++i) p->coeffs[i] = 0;
}

void sp_poly_set_from_int64(sp_poly* p, const sp_poly_coeff* coeffs, int len) {
    if (!p || !p->coeffs) return;
    if (len > p->N) len = p->N;
    for (int i = 0; i < len; ++i) p->coeffs[i] = coeffs[i];
    for (int i = len; i < p->N; ++i) p->coeffs[i] = 0;
}

static int64_t llrint_d(double v) {
    // round-half-to-even, returned as int64
    if (v >= 0.0) return (int64_t)(v + 0.5);
    return -(int64_t)(-v + 0.5);
}

void sp_poly_encode_fp32(sp_poly* p, const float* vec, int d,
                          double delta, bool reversed) {
    sp_poly_zero(p);
    if (!p || !p->coeffs || !vec || d <= 0) return;
    if (d > p->N) d = p->N;
    if (reversed) {
        for (int i = 0; i < d; ++i) {
            double v = (double)vec[d - 1 - i] * delta;
            p->coeffs[i] = (sp_poly_coeff)llrint_d(v);
        }
    } else {
        for (int i = 0; i < d; ++i) {
            double v = (double)vec[i] * delta;
            p->coeffs[i] = (sp_poly_coeff)llrint_d(v);
        }
    }
}

float sp_poly_decode_coeff(const sp_poly* p, int i, double delta_recovery) {
    if (!p || !p->coeffs || i < 0 || i >= p->N || delta_recovery == 0.0) return 0.0f;
    double v = (double)p->coeffs[i] / delta_recovery;
    return (float)v;
}

void sp_poly_mul(sp_poly* c, const sp_poly* a, const sp_poly* b) {
    if (!a || !b || !c || a->N != b->N || a->N != c->N) return;
    const int N = a->N;
    // Negacyclic convolution: c[k] = Σ_{i+j ≡ k (mod N)} sign(i+j) a[i] b[j]
    // where sign = -1 if i + j ≥ N.
    for (int k = 0; k < N; ++k) c->coeffs[k] = 0;
    for (int i = 0; i < N; ++i) {
        sp_poly_coeff ai = a->coeffs[i];
        if (ai == 0) continue;
        for (int j = 0; j < N; ++j) {
            sp_poly_coeff bj = b->coeffs[j];
            if (bj == 0) continue;
            int idx = i + j;
            sp_poly_coeff term = ai * bj;
            if (idx >= N) {
                idx -= N;
                term = -term;
            }
            c->coeffs[idx] += term;
        }
    }
}

float sp_poly_dot_product(const float* q, const float* k, int d,
                            int N, double delta,
                            sp_poly_coeff* scratch) {
    if (!q || !k || !scratch || d <= 0 || N < d) return 0.0f;
    sp_poly Q = { scratch + 0 * N, N };
    sp_poly K = { scratch + 1 * N, N };
    sp_poly C = { scratch + 2 * N, N };
    sp_poly_encode_fp32(&Q, q, d, delta, false);
    sp_poly_encode_fp32(&K, k, d, delta, true);
    sp_poly_mul(&C, &Q, &K);
    // Σ q_i k_i lands on coeff x^{d-1}.
    return sp_poly_decode_coeff(&C, d - 1, delta * delta);
}

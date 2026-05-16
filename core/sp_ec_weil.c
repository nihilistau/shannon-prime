// Shannon-Prime — elliptic curve arithmetic + Weil pairing (impl).
// Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.

#include "sp_ec_weil.h"

#include <stddef.h>

// =========================================================================
// Field arithmetic over F_p
// =========================================================================

int64_t sp_ec_mod(int64_t x, int64_t p) {
    int64_t r = x % p;
    if (r < 0) r += p;
    return r;
}

int64_t sp_ec_mod_pow(int64_t base, int64_t exp, int64_t p) {
    if (p == 1) return 0;
    int64_t result = 1;
    int64_t b = sp_ec_mod(base, p);
    while (exp > 0) {
        if (exp & 1) result = (result * b) % p;
        b = (b * b) % p;
        exp >>= 1;
    }
    return result;
}

int64_t sp_ec_mod_inv(int64_t x, int64_t p) {
    // Fermat: x^(p-2) mod p for prime p.
    return sp_ec_mod_pow(x, p - 2, p);
}

// Modular division: a / b mod p.
static int64_t f_div(int64_t a, int64_t b, int64_t p) {
    return (sp_ec_mod(a, p) * sp_ec_mod_inv(b, p)) % p;
}

// =========================================================================
// Predicates
// =========================================================================

bool sp_ec_is_infinity(sp_ec_point P) {
    return P.x < 0;
}

bool sp_ec_eq(sp_ec_point P, sp_ec_point Q) {
    if (sp_ec_is_infinity(P) && sp_ec_is_infinity(Q)) return true;
    if (sp_ec_is_infinity(P) || sp_ec_is_infinity(Q)) return false;
    return P.x == Q.x && P.y == Q.y;
}

bool sp_ec_is_on_curve(const sp_ec_curve* E, sp_ec_point P) {
    if (sp_ec_is_infinity(P)) return true;
    int64_t x = sp_ec_mod(P.x, E->p);
    int64_t y = sp_ec_mod(P.y, E->p);
    int64_t lhs = (y * y) % E->p;
    int64_t rhs = (((x * x) % E->p) * x % E->p + E->a * x % E->p + E->b) % E->p;
    return sp_ec_mod(lhs - rhs, E->p) == 0;
}

// =========================================================================
// Group operations
// =========================================================================

sp_ec_point sp_ec_neg(const sp_ec_curve* E, sp_ec_point P) {
    if (sp_ec_is_infinity(P)) return SP_EC_INFINITY;
    sp_ec_point R;
    R.x = sp_ec_mod(P.x, E->p);
    R.y = sp_ec_mod(-P.y, E->p);
    return R;
}

sp_ec_point sp_ec_add(const sp_ec_curve* E, sp_ec_point P, sp_ec_point Q) {
    if (sp_ec_is_infinity(P)) return Q;
    if (sp_ec_is_infinity(Q)) return P;
    // P + (-P) = O
    if (P.x == Q.x && sp_ec_mod(P.y + Q.y, E->p) == 0) return SP_EC_INFINITY;
    int64_t m;
    if (P.x == Q.x && P.y == Q.y) {
        // doubling: m = (3 x^2 + a) / (2 y)
        int64_t num = sp_ec_mod(3 * P.x % E->p * P.x % E->p + E->a, E->p);
        int64_t den = sp_ec_mod(2 * P.y, E->p);
        m = f_div(num, den, E->p);
    } else {
        // chord: m = (Qy - Py) / (Qx - Px)
        int64_t num = sp_ec_mod(Q.y - P.y, E->p);
        int64_t den = sp_ec_mod(Q.x - P.x, E->p);
        m = f_div(num, den, E->p);
    }
    int64_t x3 = sp_ec_mod(m * m - P.x - Q.x, E->p);
    int64_t y3 = sp_ec_mod(m * (P.x - x3) - P.y, E->p);
    sp_ec_point R = { x3, y3 };
    return R;
}

sp_ec_point sp_ec_mul(const sp_ec_curve* E, int64_t k, sp_ec_point P) {
    if (k == 0 || sp_ec_is_infinity(P)) return SP_EC_INFINITY;
    if (k < 0) return sp_ec_mul(E, -k, sp_ec_neg(E, P));
    sp_ec_point result = SP_EC_INFINITY;
    sp_ec_point Q = P;
    while (k > 0) {
        if (k & 1) result = sp_ec_add(E, result, Q);
        Q = sp_ec_add(E, Q, Q);
        k >>= 1;
    }
    return result;
}

int64_t sp_ec_order(const sp_ec_curve* E, sp_ec_point P, int64_t max_order) {
    if (sp_ec_is_infinity(P)) return 1;
    sp_ec_point Q = P;
    for (int64_t k = 1; k <= max_order; ++k) {
        if (sp_ec_is_infinity(Q)) return k;
        Q = sp_ec_add(E, Q, P);
    }
    return -1;
}

// =========================================================================
// Miller's algorithm
//
// Evaluates the rational function f_{n,P} (divisor n[P] - n[O]) at the
// point Q via double-and-add, where each step multiplies in the line
// equation through the current accumulator point and divides by the
// vertical line through the new doubled point.
// =========================================================================

// Line value at R for the line through P and Q (or tangent if P == Q).
// Returns the F_p value, or 0 if the line is vertical at R.x = P.x
// (we encode "we hit the vertical and need a different treatment" by
// returning 0 — handled by the Miller-loop branching).
static int64_t ec_line_eval(const sp_ec_curve* E,
                              sp_ec_point P, sp_ec_point Q, sp_ec_point R) {
    int64_t p = E->p;
    if (sp_ec_is_infinity(P) || sp_ec_is_infinity(Q)) return 1;
    if (P.x == Q.x && sp_ec_mod(P.y + Q.y, p) == 0) {
        // vertical line x = P.x evaluated at R: value = R.x - P.x
        return sp_ec_mod(R.x - P.x, p);
    }
    int64_t m;
    if (P.x == Q.x && P.y == Q.y) {
        int64_t num = sp_ec_mod(3 * P.x % p * P.x % p + E->a, p);
        int64_t den = sp_ec_mod(2 * P.y, p);
        m = f_div(num, den, p);
    } else {
        int64_t num = sp_ec_mod(Q.y - P.y, p);
        int64_t den = sp_ec_mod(Q.x - P.x, p);
        m = f_div(num, den, p);
    }
    // Line: y = m(x - P.x) + P.y. Evaluate (y_R - m*x_R - P.y + m*P.x).
    int64_t val = sp_ec_mod(R.y - m * R.x % p - P.y + m * P.x % p, p);
    return val;
}

int64_t sp_ec_miller(const sp_ec_curve* E, int64_t n,
                       sp_ec_point P, sp_ec_point Q) {
    if (sp_ec_is_infinity(P)) return 1;
    if (sp_ec_is_infinity(Q) || sp_ec_eq(P, Q)) {
        // Miller undefined here — caller should use a translation trick or
        // accept that the pairing is 1 in this position.
        return -1;
    }
    int64_t p = E->p;
    sp_ec_point T = P;
    int64_t f = 1;

    // Iterate over bits of n from MSB-1 down to bit 0 (skip leading 1).
    // Find the position of the highest set bit.
    int top = 63;
    while (top > 0 && ((n >> top) & 1) == 0) --top;
    for (int bit = top - 1; bit >= 0; --bit) {
        // Doubling step.
        int64_t ell = ec_line_eval(E, T, T, Q);
        sp_ec_point T2 = sp_ec_add(E, T, T);
        int64_t vert;
        if (sp_ec_is_infinity(T2)) {
            vert = 1;
        } else {
            vert = sp_ec_mod(Q.x - T2.x, p);
        }
        f = (f * f) % p;
        f = (f * sp_ec_mod(ell, p)) % p;
        if (vert != 0) f = f_div(f, vert, p);
        T = T2;
        // Add P if the current bit is 1.
        if ((n >> bit) & 1) {
            ell = ec_line_eval(E, T, P, Q);
            sp_ec_point Tn = sp_ec_add(E, T, P);
            if (sp_ec_is_infinity(Tn)) {
                vert = 1;
            } else {
                vert = sp_ec_mod(Q.x - Tn.x, p);
            }
            f = (f * sp_ec_mod(ell, p)) % p;
            if (vert != 0) f = f_div(f, vert, p);
            T = Tn;
        }
    }
    return sp_ec_mod(f, p);
}

int64_t sp_ec_weil_pairing(const sp_ec_curve* E, int64_t n,
                              sp_ec_point P, sp_ec_point Q) {
    if (sp_ec_is_infinity(P) || sp_ec_is_infinity(Q)) return 1;
    if (sp_ec_eq(P, Q)) return 1;
    int64_t fP_Q = sp_ec_miller(E, n, P, Q);
    int64_t fQ_P = sp_ec_miller(E, n, Q, P);
    if (fP_Q < 0 || fQ_P < 0 || fQ_P == 0) return 1;
    int64_t raw = f_div(fP_Q, fQ_P, E->p);
    int64_t sign = (n & 1) ? -1 : 1;  // (-1)^n
    return sp_ec_mod(sign * raw, E->p);
}

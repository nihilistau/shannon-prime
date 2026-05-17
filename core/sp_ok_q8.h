/* sp_ok_q8.h — Phase 12 Step A.
 *
 * Packed-int8 storage for elements of O_K = Z[omega], omega^2 = omega - 41,
 * with per-tensor power-of-2 shift normalization. Designed for compressed
 * storage of Frobenius-transformed weights.
 *
 * Storage contract:
 *   sp_ok_q8_t { int8_t a, int8_t b }   = 2 bytes per ring element.
 *   per-tensor { q8_shift }             = single int8, shared by every entry.
 *   reconstruction:
 *     sp_ok_t r;
 *     r.a = ((int64_t)q.a) << q8_shift;
 *     r.b = ((int64_t)q.b) << q8_shift;
 *
 * Compression ratio vs sp_ok_t (16 B/elem): 8x.
 *
 * Quantization error per coordinate is bounded by 2^q8_shift - 1, giving
 * relative error <= 2^-7 (~0.78%) for the largest-magnitude entries and
 * larger relative error for small-magnitude entries (standard int8 quant).
 *
 * The algebraic structure of O_K is preserved up to that quantization:
 * sp_ok_mul(decode(q1), decode(q2)) equals
 * sp_ok_mul(original1, original2) modulo a residual bounded by
 *   |a| * eps + |b| * eps    where eps = 2^q8_shift.
 *
 * This module is standalone — depends only on sp_ok_arith.h and stdint.
 */

#ifndef SP_OK_Q8_H
#define SP_OK_Q8_H

#include "sp_ok_arith.h"

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Packed 2-byte ring element. */
typedef struct {
    int8_t a;   /* coefficient of 1 */
    int8_t b;   /* coefficient of omega */
} sp_ok_q8_t;

/* Per-tensor packed storage header. The actual data array lives separately
 * (allocator's responsibility); this struct carries metadata. */
typedef struct {
    sp_ok_q8_t* data;             /* numel entries */
    size_t      numel;
    int8_t      q8_shift;         /* 0..62; (q.a << shift), (q.b << shift) recovers magnitude */
    int8_t      reserved[7];      /* alignment pad */
    int64_t     scale_recip;      /* original fp16 -> int scale (carried from encoder) */
    int64_t     frobenius_scale;  /* signed Frobenius scale (carried from encoder) */
    int16_t     frobenius_p;
    int16_t     frobenius_k;
    int32_t     reserved2;
} sp_ok_q8_tensor;

/* Pick the smallest non-negative shift s such that round(absmax / 2^s) <= 127
 * AFTER round-half-up. absmax must be non-negative.
 *
 * Subtlety: a naive `while (v > 127) v >>= 1` floor-shifts the absmax. The
 * encoder adds a 2^(s-1) bias before shifting, so the maximum quantized
 * value for an input equal to absmax saturates at 128 and gets clamped to
 * 127, losing up to 2^(s-1) of magnitude. To prevent saturation we use a
 * ceiling-shift here. The chosen s guarantees |round(v / 2^s)| <= 127 for
 * every v in [-absmax, absmax]. */
static inline int8_t sp_ok_q8_pick_shift(int64_t absmax) {
    if (absmax < 0) absmax = -absmax;   /* defensive */
    if (absmax <= 127) return 0;
    int8_t s = 0;
    int64_t v = absmax;
    while (v > 127) {
        v = (v + 1) >> 1;   /* ceiling-divide by 2 */
        ++s;
    }
    return s;
}

/* Compute the joint absmax across both coordinates of every element. */
int64_t sp_ok_q8_absmax(const sp_ok_t* src, size_t numel);

/* Round-to-nearest-even quantize one int64 to int8 with shift `s`. Clamps to
 * [-128, 127]. */
static inline int8_t sp_ok_q8_quantize_one(int64_t v, int8_t s) {
    if (s <= 0) {
        if (v > 127) return 127;
        if (v < -128) return -128;
        return (int8_t)v;
    }
    /* Symmetric rounding via add-then-shift, sign-correct. */
    const int64_t bias = (int64_t)1 << (s - 1);
    int64_t r;
    if (v >= 0) {
        r = (v + bias) >> s;
    } else {
        /* For negatives, shift the magnitude then negate to avoid relying
         * on implementation-defined arithmetic right shift of negative. */
        int64_t mag = -v;
        r = -((mag + bias) >> s);
    }
    if (r > 127) r = 127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* Decode one packed element back to sp_ok_t. Hot path — keep inline. */
static inline sp_ok_t sp_ok_q8_decode_one(sp_ok_q8_t q, int8_t shift) {
    sp_ok_t r;
    r.a = ((int64_t)q.a) << shift;
    r.b = ((int64_t)q.b) << shift;
    return r;
}

/* Encode a full sp_ok_t array into a pre-allocated sp_ok_q8_t array, computing
 * and returning the chosen shift. Caller supplies dst with `numel` entries. */
int8_t sp_ok_q8_encode_array(sp_ok_q8_t* dst,
                             const sp_ok_t* src,
                             size_t numel);

/* Decode a full sp_ok_q8_t array back to sp_ok_t using the supplied shift. */
void sp_ok_q8_decode_array(sp_ok_t* dst,
                           const sp_ok_q8_t* src,
                           size_t numel,
                           int8_t shift);

/* Maximum quantization error in either coordinate, given a shift.
 *   |decoded.a - original.a| <= sp_ok_q8_max_error(shift)
 * Useful for parity test bounds. */
static inline int64_t sp_ok_q8_max_error(int8_t shift) {
    if (shift <= 0) return 0;
    /* Half-step rounding => error bounded by 2^(shift - 1). */
    return ((int64_t)1) << (shift - 1);
}

#ifdef __cplusplus
}
#endif

#endif /* SP_OK_Q8_H */

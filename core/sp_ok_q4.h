/* sp_ok_q4.h — Phase 14 disk-shrink: 4-bit packed O_K storage.
 *
 * Compresses each ring element (a, b) of O_K = Z[omega], omega^2 = omega - 41
 * to a single byte by packing two signed 4-bit nybbles. The encoder and
 * decoder mirror sp_ok_q8 exactly — same ceiling-shift, same round-half-up,
 * same per-tensor shared shift — but with the storage container halved.
 *
 * Compression ratio vs sp_ok_t (16 B/elem): 16x.
 * Compression ratio vs sp_ok_q8_t (2 B/elem): 2x.
 *
 * Storage contract:
 *   sp_ok_q4_t { uint8_t packed }    = 1 byte per ring element
 *     packed & 0x0F (signed 4-bit) = a coordinate
 *     packed >> 4   (signed 4-bit) = b coordinate
 *   per-tensor { q4_shift }          = single int8, shared by every entry
 *   reconstruction:
 *     int8_t a4 = (int8_t)(packed << 4) >> 4;     -- sign-extend low nybble
 *     int8_t b4 = (int8_t)packed >> 4;             -- sign-extend high nybble
 *     sp_ok_t r;
 *     r.a = ((int64_t)a4) << q4_shift;
 *     r.b = ((int64_t)b4) << q4_shift;
 *
 * The arithmetic-shift sign-extension trick costs two cycles on any CPU
 * with signed shift (every modern x86/ARM/RISC-V), zero cycles of float
 * math, and folds cleanly into AVX-512 / NEON / SVE vector lanes.
 *
 * Quantization error per coordinate is bounded by 2^(q4_shift - 1), the
 * same half-step bound as sp_ok_q8 just at a different scale. The 4-bit
 * codebook has ~16 levels vs ~256 for int8, so for the same numerical
 * absmax the q4_shift will be larger by ~4 bits, giving 16x larger
 * absolute quantization noise per coordinate. Whether that's tolerable
 * depends on the downstream invariant — Theorem 2's projective cancellation
 * absorbs uniform scale changes, but the quantization noise itself is the
 * standard rate-distortion cost of dropping from 8 bits to 4.
 *
 * This module is standalone — depends only on sp_ok_arith.h and stdint.
 */

#ifndef SP_OK_Q4_H
#define SP_OK_Q4_H

#include "sp_ok_arith.h"

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Packed 1-byte ring element. Low 4 bits = a coordinate (signed),
 * high 4 bits = b coordinate (signed). Both interpreted as int4_t
 * via sign extension at decode time. */
typedef struct {
    uint8_t packed;
} sp_ok_q4_t;

/* Per-tensor packed storage header.  Field layout mirrors sp_ok_q8_tensor
 * exactly so loader code can swap one for the other with a type pun on
 * the data pointer. */
typedef struct {
    sp_ok_q4_t* data;             /* numel entries, 1 byte each */
    size_t      numel;
    int8_t      q4_shift;         /* 0..62; shared shift, see encode/decode */
    int8_t      reserved[7];      /* alignment pad */
    int64_t     scale_recip;      /* original fp16 -> int scale (carried from encoder) */
    int64_t     frobenius_scale;  /* signed Frobenius scale (carried from encoder) */
    int16_t     frobenius_p;
    int16_t     frobenius_k;
    int32_t     reserved2;
} sp_ok_q4_tensor;

/* Signed-4-bit limits. */
#define SP_OK_Q4_MAX  ((int8_t) 7)
#define SP_OK_Q4_MIN  ((int8_t)-8)

/* Pick the smallest non-negative shift s such that the absolute value
 * round(absmax / 2^s) fits in 4 signed bits (<= 7) AFTER round-half-up.
 * Uses the same ceiling-shift logic as sp_ok_q8_pick_shift so the max
 * input value doesn't saturate the codebook on round-up. */
static inline int8_t sp_ok_q4_pick_shift(int64_t absmax) {
    if (absmax < 0) absmax = -absmax;
    if (absmax <= SP_OK_Q4_MAX) return 0;
    int8_t s = 0;
    int64_t v = absmax;
    while (v > SP_OK_Q4_MAX) {
        v = (v + 1) >> 1;       /* ceiling-divide by 2 */
        ++s;
    }
    return s;
}

/* Joint absmax across both coordinates of every element. */
int64_t sp_ok_q4_absmax(const sp_ok_t* src, size_t numel);

/* Round-to-nearest quantize one int64 to a signed 4-bit code in [-8, 7]
 * via shift `s`. Returns a value in that signed range, cast to int8 for
 * caller convenience. Clamps to [-8, 7]. */
static inline int8_t sp_ok_q4_quantize_one(int64_t v, int8_t s) {
    int64_t r;
    if (s <= 0) {
        r = v;
    } else {
        const int64_t bias = (int64_t)1 << (s - 1);
        if (v >= 0) {
            r = (v + bias) >> s;
        } else {
            int64_t mag = -v;
            r = -((mag + bias) >> s);
        }
    }
    if (r > SP_OK_Q4_MAX) r = SP_OK_Q4_MAX;
    if (r < SP_OK_Q4_MIN) r = SP_OK_Q4_MIN;
    return (int8_t)r;
}

/* Pack two signed 4-bit codes (each must already be in [-8, 7]) into the
 * single-byte storage. Caller is expected to have run sp_ok_q4_quantize_one
 * on both inputs first. */
static inline uint8_t sp_ok_q4_pack_pair(int8_t a4, int8_t b4) {
    /* Mask off the low nybble of each input. The sign bit (bit 3 of the
     * 4-bit value) is preserved in the bottom 4 bits of the int8 after
     * the explicit mask. */
    uint8_t a_nyb = (uint8_t)((uint8_t)a4 & 0x0Fu);
    uint8_t b_nyb = (uint8_t)((uint8_t)b4 & 0x0Fu);
    return (uint8_t)(a_nyb | (b_nyb << 4));
}

/* Decode one packed byte back to sp_ok_t. The arithmetic-shift idiom
 * sign-extends each 4-bit field into a 32-bit register without any
 * branches or mask tables. Hot path — keep inline. */
static inline sp_ok_t sp_ok_q4_decode_one(sp_ok_q4_t q, int8_t shift) {
    /* Shift low nybble (bits [3:0]) to bits [31:28], then arithmetic shift
     * right by 28 to broadcast the sign bit and recover a signed int. */
    int32_t a4 = ((int32_t)((uint32_t)q.packed << 28)) >> 28;
    /* Shift high nybble (bits [7:4]) to bits [31:28] via << 24, then ASR by 28. */
    int32_t b4 = ((int32_t)((uint32_t)q.packed << 24)) >> 28;
    sp_ok_t r;
    r.a = ((int64_t)a4) << shift;
    r.b = ((int64_t)b4) << shift;
    return r;
}

/* Encode a full sp_ok_t array into a pre-allocated sp_ok_q4_t array,
 * computing and returning the chosen shift. Caller supplies dst with
 * `numel` entries (numel bytes). */
int8_t sp_ok_q4_encode_array(sp_ok_q4_t* dst,
                             const sp_ok_t* src,
                             size_t numel);

/* Decode a full sp_ok_q4_t array back to sp_ok_t using the supplied shift. */
void sp_ok_q4_decode_array(sp_ok_t* dst,
                           const sp_ok_q4_t* src,
                           size_t numel,
                           int8_t shift);

/* Maximum quantization error in either coordinate, given a shift.
 *   |decoded.a - original.a| <= sp_ok_q4_max_error(shift)
 * Useful for parity test bounds. */
static inline int64_t sp_ok_q4_max_error(int8_t shift) {
    if (shift <= 0) return 0;
    return ((int64_t)1) << (shift - 1);
}

/* ============================================================================
 * Phase 14 Step 2: Lattice-norm pruning
 * ============================================================================
 *
 * For every coordinate pair (a, b), compute the algebraic norm in the
 * Heegner field Q(sqrt(-163)) — but using the omega^2 = omega - 41 basis,
 * the relevant norm is a^2 - a*b + (some factor)*b^2 for the Eisenstein-like
 * extension we use. Per the existing Frobenius shim's invariants, the
 * appropriate norm for filtering is the squared L2 magnitude of the
 * coordinate pair, weighted to match O_K's geometry:
 *
 *   N(a + b*omega) = a^2 + a*b + 41*b^2     (standard, omega has min poly
 *                                            x^2 - x + 41)
 *
 * Elements with N(alpha) below a caller-supplied threshold are zeroed.
 * Produces long runs of (0, 0) bytes for downstream entropy coding.
 *
 * This is OPT-IN at encode time. The Q4 encoder ignores pruning unless
 * sp_ok_q4_encode_array_pruned is used explicitly. */

/* Compute N(alpha) for one ring element. Always positive (omega^2 = omega - 41
 * gives discriminant 1 - 4*41 = -163; the norm form a^2 + ab + 41 b^2 is
 * positive-definite by classical theory of imaginary quadratic fields). */
static inline uint64_t sp_ok_q4_norm(int64_t a, int64_t b) {
    /* All terms fit in int64 for any a, b in the post-Frobenius range
     * (typically |a|, |b| < 2^28). a*b can be up to 2^56, 41*b^2 up to
     * 2^61, well within uint64. */
    int64_t ab = a * b;
    int64_t a2 = a * a;
    int64_t b2 = b * b;
    int64_t n  = a2 + ab + 41 * b2;
    if (n < 0) n = -n;     /* defensive; shouldn't happen for valid inputs */
    return (uint64_t)n;
}

/* Encode with optional lattice-norm pruning. Elements with norm < threshold
 * are pre-zeroed in-place (modifying src) before quantization, producing
 * runs of 0x00 packed bytes in dst.
 *
 *   threshold == 0  → no pruning (equivalent to sp_ok_q4_encode_array)
 *   threshold >  0  → zero (a, b) when a^2 + ab + 41*b^2 < threshold
 *
 * Returns the chosen shift like the unpruned encoder. */
int8_t sp_ok_q4_encode_array_pruned(sp_ok_q4_t* dst,
                                    sp_ok_t* src,         /* mutated */
                                    size_t numel,
                                    uint64_t norm_threshold);

/* Count of zeroed coordinate pairs after the most recent encode_array_pruned
 * call. Diagnostic — caller can read this to characterize sparsity. */
extern size_t sp_ok_q4_last_pruned_count;

#ifdef __cplusplus
}
#endif

#endif /* SP_OK_Q4_H */

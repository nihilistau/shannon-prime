/* sp_ok_q8.c — Phase 12 Step A: packed-int8 O_K storage.
 *
 * Implementation note on quantization: we use round-half-away-from-zero
 * symmetric rounding. This keeps the encoder's behavior strictly symmetric
 * across +/- coordinates, which matters because Theorem 2 (CM Sato-Tate)
 * predicts a balanced distribution of post-Frobenius coordinates around 0.
 *
 * All operations are exact integer arithmetic. No floating point appears in
 * either the encoder or the decoder hot path.
 */

#include "sp_ok_q8.h"

int64_t sp_ok_q8_absmax(const sp_ok_t* src, size_t numel) {
    int64_t m = 0;
    for (size_t i = 0; i < numel; ++i) {
        int64_t a = src[i].a;
        int64_t b = src[i].b;
        if (a < 0) a = -a;
        if (b < 0) b = -b;
        if (a > m) m = a;
        if (b > m) m = b;
    }
    return m;
}

int8_t sp_ok_q8_encode_array(sp_ok_q8_t* dst,
                             const sp_ok_t* src,
                             size_t numel) {
    int64_t am = sp_ok_q8_absmax(src, numel);
    int8_t  s  = sp_ok_q8_pick_shift(am);
    for (size_t i = 0; i < numel; ++i) {
        dst[i].a = sp_ok_q8_quantize_one(src[i].a, s);
        dst[i].b = sp_ok_q8_quantize_one(src[i].b, s);
    }
    return s;
}

/* AVX-512 vectorized decoder: sign-extends 16 int8 pairs at a time, then
 * left-shifts each int64 lane by `shift`. The packed input is 16-byte
 * (8 sp_ok_q8_t = 16 int8s); the unpacked output is 256 bytes (8 sp_ok_t
 * pairs = 16 int64s). When `shift` is 0, we degenerate to a sign-extend
 * only -- still 8x scalar throughput. */
#if defined(__AVX512F__)
#include <immintrin.h>
#define SP_OK_Q8_DECODE_AVX512 1
#endif

void sp_ok_q8_decode_array(sp_ok_t* dst,
                           const sp_ok_q8_t* src,
                           size_t numel,
                           int8_t shift) {
    size_t i = 0;
#if SP_OK_Q8_DECODE_AVX512
    /* Process 8 sp_ok_q8_t (= 8 packed pairs = 16 int8s = 16 bytes) per
     * iteration, producing 8 sp_ok_t (= 16 int64s = 128 bytes) of output. */
    const int s = (int)shift;
    for (; i + 8 <= numel; i += 8) {
        /* Load 16 int8s. The 16 bytes interleave a0,b0,a1,b1,...,a7,b7. */
        __m128i v8 = _mm_loadu_si128((const __m128i*)(src + i));
        /* Sign-extend 16 x int8 -> 16 x int64 across two 512-bit regs. */
        __m512i lo = _mm512_cvtepi8_epi64(v8);
        __m512i hi = _mm512_cvtepi8_epi64(_mm_srli_si128(v8, 8));
        /* Apply shared left shift (zero-shift is a no-op). */
        if (s > 0) {
            lo = _mm512_slli_epi64(lo, s);
            hi = _mm512_slli_epi64(hi, s);
        }
        /* sp_ok_t is { int64 a; int64 b } in memory; the cvtepi8_epi64
         * spread already produced the right interleaving: a0,b0,a1,b1...
         * Stream the 1024 bits straight out. */
        _mm512_storeu_si512((__m512i*)(dst + i),     lo);
        _mm512_storeu_si512((__m512i*)(dst + i + 4), hi);
    }
#endif
    /* Scalar tail (also the entire loop on non-AVX-512 builds). */
    for (; i < numel; ++i) {
        dst[i] = sp_ok_q8_decode_one(src[i], shift);
    }
}

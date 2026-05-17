/* sp_ok_q4.c — Phase 14 disk-shrink: 4-bit packed O_K storage (impl).
 *
 * Mirrors sp_ok_q8.c exactly but with a 4-bit codebook. Same ceiling-shift
 * picker, same round-half-up quantizer, same scalar/SIMD decode pattern.
 *
 * Optional lattice-norm pruning lives in sp_ok_q4_encode_array_pruned:
 * elements whose N(a + b*omega) = a^2 + ab + 41*b^2 falls below the
 * caller-supplied threshold are zeroed before quantization, producing
 * runs of 0x00 in the packed output. Downstream entropy coding (zstd,
 * Huffman) compresses those runs aggressively, hitting the disk-size
 * goal of the Phase 14 spec.
 */

#include "sp_ok_q4.h"

#include <string.h>

size_t sp_ok_q4_last_pruned_count = 0;

int64_t sp_ok_q4_absmax(const sp_ok_t* src, size_t numel) {
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

int8_t sp_ok_q4_encode_array(sp_ok_q4_t* dst,
                             const sp_ok_t* src,
                             size_t numel) {
    int64_t am = sp_ok_q4_absmax(src, numel);
    int8_t  s  = sp_ok_q4_pick_shift(am);
    for (size_t i = 0; i < numel; ++i) {
        int8_t a4 = sp_ok_q4_quantize_one(src[i].a, s);
        int8_t b4 = sp_ok_q4_quantize_one(src[i].b, s);
        dst[i].packed = sp_ok_q4_pack_pair(a4, b4);
    }
    return s;
}

int8_t sp_ok_q4_encode_array_pruned(sp_ok_q4_t* dst,
                                    sp_ok_t* src,
                                    size_t numel,
                                    uint64_t norm_threshold) {
    /* Pass 1: prune below-threshold elements in place. Track pruned count
     * for the public diagnostic counter. */
    size_t pruned = 0;
    if (norm_threshold > 0) {
        for (size_t i = 0; i < numel; ++i) {
            uint64_t n = sp_ok_q4_norm(src[i].a, src[i].b);
            if (n < norm_threshold) {
                src[i].a = 0;
                src[i].b = 0;
                ++pruned;
            }
        }
    }
    sp_ok_q4_last_pruned_count = pruned;

    /* Pass 2: standard encode. The pruned zero entries collapse to 0x00
     * packed bytes, and the absmax/shift picker now sees a tighter range
     * (often leading to a smaller shift, which improves precision on the
     * surviving entries). */
    return sp_ok_q4_encode_array(dst, src, numel);
}

/* AVX-512 vectorized decoder. Each iteration consumes 16 packed bytes
 * (= 16 ring elements), produces 32 int64 outputs across two 512-bit
 * registers. The 4-bit sign-extension is done in 32-bit lanes via the
 * arithmetic-shift trick (shift-left-then-shift-right), then promoted
 * to int64. */
#if defined(__AVX512F__)
#  include <immintrin.h>
#  define SP_OK_Q4_DECODE_AVX512 1
#endif

void sp_ok_q4_decode_array(sp_ok_t* dst,
                           const sp_ok_q4_t* src,
                           size_t numel,
                           int8_t shift) {
    size_t i = 0;

#if SP_OK_Q4_DECODE_AVX512
    /* AVX-512 path: process 8 packed bytes per iteration. We use the
     * 32-bit shift trick because AVX-512 doesn't have a native int4
     * extract, but does have efficient sign-aware shifts of int32 lanes
     * and a clean cvtepi32_epi64 promotion to int64.
     *
     * Per 8 input bytes we produce 16 int64s (16 sp_ok_t coords, i.e.
     * 8 sp_ok_t structs at 2 int64s each), so this matches one cache
     * line of output per loop iteration. */
    const int s = (int)shift;
    for (; i + 8 <= numel; i += 8) {
        /* Load 8 packed bytes -> low 64 bits of an xmm reg. */
        __m128i packed = _mm_loadl_epi64((const __m128i*)(src + i));
        /* Zero-extend the 8 uint8s into 8 int32 lanes. */
        __m256i p32   = _mm256_cvtepu8_epi32(packed);
        /* Low nybble: shift left by 28, then arithmetic shift right by 28. */
        __m256i a32   = _mm256_srai_epi32(_mm256_slli_epi32(p32, 28), 28);
        /* High nybble: shift left by 24, then arithmetic shift right by 28. */
        __m256i b32   = _mm256_srai_epi32(_mm256_slli_epi32(p32, 24), 28);

        /* Promote each 32-bit signed lane to 64-bit. */
        __m512i a64   = _mm512_cvtepi32_epi64(a32);
        __m512i b64   = _mm512_cvtepi32_epi64(b32);

        /* Apply the shared shift (zero is no-op). */
        if (s > 0) {
            a64 = _mm512_slli_epi64(a64, s);
            b64 = _mm512_slli_epi64(b64, s);
        }

        /* Interleave a64 and b64 so output memory layout is
         * (a0, b0, a1, b1, ...) matching sp_ok_t { int64 a; int64 b }.
         * AVX-512's unpacklo/hi_epi64 do this across 128-bit halves. */
        __m512i lo = _mm512_unpacklo_epi64(a64, b64);
        __m512i hi = _mm512_unpackhi_epi64(a64, b64);

        /* Store 8 sp_ok_t = 1024 bits in two halves. The unpack lanes
         * give us: lo = [a0,b0, a2,b2, a4,b4, a6,b6]
         *          hi = [a1,b1, a3,b3, a5,b5, a7,b7]
         * The interleaved output needs (a0,b0,a1,b1,...) — we cross
         * the halves via permute2x128 / blend at the 128-bit lane level.
         */
        __m512i out01 = _mm512_shuffle_i64x2(lo, hi, 0x44); /* lanes 0,1 from lo+hi */
        __m512i out23 = _mm512_shuffle_i64x2(lo, hi, 0xEE); /* lanes 2,3 from lo+hi */
        _mm512_storeu_si512((__m512i*)(dst + i),     out01);
        _mm512_storeu_si512((__m512i*)(dst + i + 4), out23);
    }
#endif

    /* Scalar tail (also the entire loop on non-AVX-512 builds). */
    for (; i < numel; ++i) {
        dst[i] = sp_ok_q4_decode_one(src[i], shift);
    }
}

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

void sp_ok_q8_decode_array(sp_ok_t* dst,
                           const sp_ok_q8_t* src,
                           size_t numel,
                           int8_t shift) {
    for (size_t i = 0; i < numel; ++i) {
        dst[i] = sp_ok_q8_decode_one(src[i], shift);
    }
}

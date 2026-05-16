// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

//
// sp_ok_arith.c — out-of-line implementations.
// The hot-path arithmetic is in the header (static inline). This file
// holds the only non-trivial routine: square-and-multiply exponentiation.

#include "sp_ok_arith.h"

sp_ok_t sp_ok_pow(sp_ok_t x, int64_t k) {
    sp_ok_t result = SP_OK_ONE;
    sp_ok_t base = x;
    while (k > 0) {
        if (k & 1) {
            result = sp_ok_mul(result, base);
        }
        base = sp_ok_mul(base, base);
        k >>= 1;
    }
    return result;
}

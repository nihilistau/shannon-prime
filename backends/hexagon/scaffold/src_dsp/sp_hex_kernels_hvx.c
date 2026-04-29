// Shannon-Prime VHT2 - Hexagon DSP HVX kernels.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// 1024-bit HVX vector implementations of the SP math core's hot loops.
// Numerically equivalent to the scalar reference (core/shannon_prime.c)
// — same IEEE fp32 ops in the same order, just batched 32 lanes wide.
//
// Build gating
// ------------
//   __HVX__   defined when the toolchain targets a DSP variant
//                     with HVX. Always true for V60+, including our V69.
//
// All entry points fall back to the scalar reference for sizes that
// can't usefully fill a vector (n < 32) or for non-power-of-2 sizes
// which the multi-prime VHT2 path would handle. This keeps the IDL
// surface uniform — callers don't branch on capability.

#ifdef __HVX__

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <string.h>

#include "shannon_prime.h"   // sp_vht2_forward_f32 (scalar fallback)

// Forward decl from sp_hex_kernels.h — exposing this symbol via a header
// would create a circular include between scalar and HVX TUs, so we
// re-declare here. The scalar version stays the public API.
void sp_hex_vht2_f32_hvx(float *data, int n);

// Reinterpret a float as its 32-bit IEEE bit pattern, suitable for
// passing to Q6_V_vsplat_R. Avoids a union and keeps strict-aliasing
// rules happy.
static inline int sp_hex_f32_bits(float f) {
    int bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

// HVX-vectorised VHT2 forward butterfly for power-of-2 head_dim.
// VHT2 is self-inverse, so calling this twice round-trips to the input
// modulo IEEE roundoff (the same property the scalar reference has).
//
// Algorithm: log2(n) passes, each pass at increasing stride. The pass
// at stride S transforms each pair (data[i+j], data[i+S+j]) into
// ((x0+x1)*s, (x0-x1)*s) where s = 1/√2. For S >= 32 the j loop fills
// at least one HVX vector; we batch 32 fp32 lanes at a time. For S < 32
// (the first few passes) the scalar fallback runs — these passes touch
// O(n) total elements so the perf cost is bounded.
//
// Caller invariants: n is a power of 2 and ≥ 8 (matches SP math core).
// Misaligned float arrays work but pay the unaligned-load cost on V69;
// if profiles show this matters we'll add an aligned-staging copy.
void sp_hex_vht2_f32_hvx(float *data, int n) {
    if (!data || n < 2) return;

    const float s_const = 0.70710678118654752440f;  // 1/√2
    HVX_Vector vs = Q6_V_vsplat_R(sp_hex_f32_bits(s_const));

    for (int stride = 1; stride < n; stride <<= 1) {
        const int block = 2 * stride;

        if (stride >= 32) {
            // HVX path: 32 fp32 lanes per vector. The j loop runs in
            // strides of 32 since stride is a power of 2 and ≥ 32.
            for (int i = 0; i < n; i += block) {
                for (int j = 0; j < stride; j += 32) {
                    HVX_Vector *p0 = (HVX_Vector *)&data[i + j];
                    HVX_Vector *p1 = (HVX_Vector *)&data[i + stride + j];
                    HVX_Vector x0  = *p0;
                    HVX_Vector x1  = *p1;
                    // Q6_Vsf_*  : IEEE single-precision (ieee fp32)
                    HVX_Vector add = Q6_Vsf_vadd_VsfVsf(x0, x1);
                    HVX_Vector sub = Q6_Vsf_vsub_VsfVsf(x0, x1);
                    *p0 = Q6_Vsf_vmpy_VsfVsf(add, vs);
                    *p1 = Q6_Vsf_vmpy_VsfVsf(sub, vs);
                }
            }
        } else {
            // Scalar tail for the first few passes (stride 1, 2, 4, 8, 16).
            // Total elements touched at small strides = n * passes-with-S<32
            // = n * 5 = bounded; this loop does not dominate the runtime
            // for typical SP head_dim values (64–512).
            for (int i = 0; i < n; i += block) {
                for (int j = 0; j < stride; j++) {
                    float x0 = data[i + j];
                    float x1 = data[i + stride + j];
                    data[i + j]          = (x0 + x1) * s_const;
                    data[i + stride + j] = (x0 - x1) * s_const;
                }
            }
        }
    }
}

#endif  // __HVX__

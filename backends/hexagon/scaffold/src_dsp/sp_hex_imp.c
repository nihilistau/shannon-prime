// Shannon-Prime VHT2 - Hexagon DSP scaffold (FastRPC interface impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Implements the IDL methods declared in inc/sp_hex.idl. The qaic-generated
// sp_hex_skel.c calls these functions on the DSP side after unmarshalling
// FastRPC arguments. Forked from the Hexagon SDK 5.5.6.0 S22U sample
// (Qualcomm copyright on the lifecycle pattern; SP-specific math is AGPLv3).

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "HAP_farf.h"
#include "sp_hex.h"            // qaic-generated header from sp_hex.idl
#include "sp_hex_kernels.h"    // scalar reference math

// ============================================================================
// Lifecycle - inherited from remote_handle64.
// ============================================================================

int sp_hex_open(const char *uri, remote_handle64 *handle) {
    (void)uri;
    void *tptr = malloc(1);  // FastRPC requires a non-null handle
    *handle = (remote_handle64)tptr;
    assert(*handle);
    FARF(RUNTIME_HIGH, "[sp_hex] open() -> handle=0x%llx", *handle);
    return 0;
}

int sp_hex_close(remote_handle64 handle) {
    if (handle) free((void *)handle);
    FARF(RUNTIME_HIGH, "[sp_hex] close()");
    return 0;
}

// ============================================================================
// IDL methods.
// ============================================================================

// VHT2(VHT2(x)) ≈ x to fp32 epsilon. The scaffold's smoke test verifies this
// round-trip. Once HVX kernels land, we'll widen the test to include the
// quantize+dequantize sandwich.
int sp_hex_round_trip_f32(remote_handle64 h,
                           const float *in_vec, int in_len,
                           int head_dim,
                           float *out_vec, int out_len) {
    (void)h;
    if (in_len != head_dim || out_len != head_dim) {
        FARF(ERROR, "[sp_hex] round_trip: length mismatch in=%d out=%d hd=%d",
             in_len, out_len, head_dim);
        return -1;
    }
    if (head_dim < 8 || (head_dim & (head_dim - 1)) != 0) {
        FARF(ERROR, "[sp_hex] round_trip: head_dim=%d must be pow2 and >=8",
             head_dim);
        return -1;
    }

    memcpy(out_vec, in_vec, sizeof(float) * head_dim);
    sp_hex_vht2_f32(out_vec, head_dim);   // forward
    sp_hex_vht2_f32(out_vec, head_dim);   // inverse (self-inverse property)

    FARF(RUNTIME_HIGH, "[sp_hex] round_trip done: hd=%d in[0]=%f out[0]=%f",
         head_dim, in_vec[0], out_vec[0]);
    return 0;
}

int sp_hex_vht2_forward(remote_handle64 h, float *data, int data_len, int n) {
    (void)h;
    if (data_len != n) {
        FARF(ERROR, "[sp_hex] vht2_forward: data_len %d != n %d", data_len, n);
        return -1;
    }
    sp_hex_vht2_f32(data, n);
    return 0;
}

int sp_hex_band_quantize(remote_handle64 h,
                          const float *coeffs, int coeffs_len,
                          int head_dim,
                          unsigned char *out, int out_capacity) {
    (void)h;
    if (coeffs_len != head_dim) return -1;
    int written = 0;
    return sp_hex_band_quantize_scalar(coeffs, head_dim, out, out_capacity,
                                       &written);
}

int sp_hex_band_dequantize(remote_handle64 h,
                            const unsigned char *in, int in_len,
                            int head_dim, int max_bands,
                            float *out_coeffs, int out_len) {
    (void)h;
    if (out_len != head_dim) return -1;
    return sp_hex_band_dequantize_scalar(in, in_len, head_dim, max_bands,
                                          out_coeffs);
}

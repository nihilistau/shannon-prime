// Shannon-Prime Mode D Stage 1 probe — host driver.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3.
//
// This is the host (ARM Android) side of the SP-packed-bytes UBWCDMA
// probe. It:
//
//   1. Generates a known coefficient sequence per band.
//   2. Quantizes those coefficients to {3, 4, 5}-bit codes per the
//      SP K config (band_bits = {5, 5, 4, 3}).
//   3. Packs the codes contiguously per band (no fp16 scale header —
//      the scale travels as a separate IDL parameter in this probe).
//   4. Calls sp_dma_raw_run() over FastRPC.
//   5. Receives the fp32 dequantized output from the DSP.
//   6. Compares to a scalar reference computed locally; reports
//      max-error / RMS / pass-fail, and the timing the DSP measured.
//
// Patterned after dma_raw_blur_rw_async/host/main.cpp's verify() pattern.
//
// usage: ./test-sp-dma-raw [iterations]

#include "buffer.h"
#include "ion_allocation.h"
#include "rpcmem.h"
#include "sp_dma_raw.h"        // FastRPC stub (generated from .idl)
#include "test_report.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// SP K config we're probing. {5,5,4,3}-bit bands at hd=128 split into
// 4 equal bands of 32 coefficients each.
static constexpr int    N_BANDS              = 4;
static constexpr int    COEFFS_PER_BAND[]    = {32, 32, 32, 32};
static constexpr int    BITS_PER_BAND[]      = { 5,  5,  4,  3};
static constexpr float  SCALE_PER_BAND[]     = {0.04f, 0.02f, 0.015f, 0.01f};
// max_val[b] = (1 << (bits-1)) - 1
static constexpr int    MAX_VAL_PER_BAND[]   = {15, 15, 7, 3};

// Ground-truth coefficient generator. Uses a deterministic pattern
// (sine + small phase offset per band) so the values are spread across
// the dynamic range of each quantizer.
static float gen_coeff(int band, int i) {
    // amplitude near MAX_VAL[b] * SCALE[b] so we exercise saturation
    // without clipping it
    float amp = (MAX_VAL_PER_BAND[band] - 1) * SCALE_PER_BAND[band];
    float phase = 0.3f * band + 0.05f * i;
    return amp * std::sin(phase);
}

// Quantize a single coefficient to a code in [-max_val, +max_val].
static int quantize(float x, float scale, int max_val) {
    int q = (int)std::lround(x / scale);
    if (q >  max_val) q =  max_val;
    if (q < -max_val) q = -max_val;
    return q;
}

// Pack the codes for one band into bytes. Returns ceil(n*bits/8).
// Bit order: code i occupies bits [i*bits, (i+1)*bits) of the byte stream
// in little-endian-within-byte order, which matches what the Halide
// generator's `(window >> bit_offset) & mask` recovers.
static int pack_band(const int *codes, int n, int bits, int max_val,
                     uint8_t *out, int out_capacity) {
    const int n_bytes = (n * bits + 7) / 8;
    if (n_bytes > out_capacity) {
        std::fprintf(stderr, "pack_band: out_capacity %d < %d\n", out_capacity, n_bytes);
        return -1;
    }
    std::memset(out, 0, n_bytes);
    for (int i = 0; i < n; ++i) {
        int signed_q = codes[i];
        // Shift to unsigned [0, 2*max_val] for packing — matches the
        // Halide kernel's `q = (int)u - max_val` decode step.
        int u = signed_q + max_val;
        int start_bit = i * bits;
        int byte_pos  = start_bit / 8;
        int bit_off   = start_bit % 8;
        // 16-bit window write so we can straddle byte boundaries
        uint16_t window = (uint16_t)out[byte_pos] | ((uint16_t)out[byte_pos + 1] << 8);
        window |= (uint16_t)(u << bit_off);
        out[byte_pos]     = (uint8_t)(window & 0xFF);
        out[byte_pos + 1] = (uint8_t)((window >> 8) & 0xFF);
    }
    return n_bytes;
}

// Scalar SP dequantize reference — what the DSP output should match.
static float dequantize(int unsigned_code, int max_val, float scale) {
    int q = unsigned_code - max_val;
    return (float)q * scale;
}

int main(int argc, char *argv[]) {
    int iterations = (argc > 1) ? std::atoi(argv[1]) : 100;

    alloc_init();

    // Power on HVX + turbo
    if (sp_dma_raw_power_on_hvx() != 0) {
        std::fprintf(stderr, "power_on_hvx failed\n");
        return 1;
    }
    sp_dma_raw_set_hvx_perf_mode_turbo();

    // Pick a packed_band_stride that's a multiple of 128 (vector size)
    // and large enough for the widest band: 32 coeffs * 5 bits = 20 bytes.
    // Pad to 128 to give the schedule room.
    const int packed_band_stride = 128;
    const int coeffs_stride      = 32;  // == max coeffs per band

    // Allocate ION buffers (rpcmem-backed, contiguous, DMA-friendly).
    buffer_2d<uint8_t> packed_in(packed_band_stride, N_BANDS);
    buffer_2d<float>   coeffs_out(coeffs_stride, N_BANDS);

    // Ground-truth + reference output we expect.
    std::vector<float> ref_coeffs(N_BANDS * coeffs_stride, 0.0f);

    // Pack per band.
    uint8_t *packed_buf = packed_in.get_buffer();
    std::memset(packed_buf, 0, packed_band_stride * N_BANDS);
    for (int b = 0; b < N_BANDS; ++b) {
        int codes[64] = {0};
        for (int i = 0; i < COEFFS_PER_BAND[b]; ++i) {
            float x = gen_coeff(b, i);
            codes[i] = quantize(x, SCALE_PER_BAND[b], MAX_VAL_PER_BAND[b]);
            // What we EXPECT the DSP to produce after dequantize.
            ref_coeffs[b * coeffs_stride + i] =
                dequantize(codes[i] + MAX_VAL_PER_BAND[b],
                           MAX_VAL_PER_BAND[b], SCALE_PER_BAND[b]);
        }
        int n_bytes = pack_band(codes, COEFFS_PER_BAND[b],
                                BITS_PER_BAND[b], MAX_VAL_PER_BAND[b],
                                packed_buf + b * packed_band_stride,
                                packed_band_stride);
        if (n_bytes < 0) return 1;
        std::printf("band %d: %d coeffs * %d bits -> %d bytes (capacity %d)\n",
                    b, COEFFS_PER_BAND[b], BITS_PER_BAND[b],
                    n_bytes, packed_band_stride);
    }

    // Send to DSP.
    uint64_t avg_time = 0;
    int rc = sp_dma_raw_run(
        packed_in.get_buffer(), packed_band_stride * N_BANDS,
        packed_band_stride,
        N_BANDS,
        /*is_input_ubwc=*/0,
        SCALE_PER_BAND,    N_BANDS,
        BITS_PER_BAND,     N_BANDS,
        MAX_VAL_PER_BAND,  N_BANDS,
        COEFFS_PER_BAND,   N_BANDS,
        coeffs_stride,
        coeffs_out.get_buffer(), coeffs_stride * N_BANDS,
        iterations,
        &avg_time);

    sp_dma_raw_power_off_hvx();

    if (rc != 0) {
        std::fprintf(stderr, "PROBE FAIL: sp_dma_raw_run returned %d\n", rc);
        std::fprintf(stderr,
            "Most likely culprits in order of probability:\n"
            "  1. prepare_for_copy_to_device rejects fp32 RAW (output type)\n"
            "  2. prepare_for_copy_to_host rejects packed-bytes RAW (input layout)\n"
            "  3. UBWCDMA descriptor issue (frame size / stride alignment)\n"
            "  4. ION allocation alignment (need 128-byte aligned for HVX)\n"
            "Check FARF logs from DSP for the failing call site.\n");
        TestReport tr("sp_dma_raw_probe", 0, "microseconds",
                      Mode::Device_Standalone, Result::Fail);
        tr.print();
        alloc_finalize();
        return rc;
    }

    // Verify: max error per band, overall RMS, # mismatches > 1e-3.
    int total = N_BANDS * coeffs_stride;
    float max_err = 0.0f;
    double sse = 0.0;
    int n_mismatch = 0;
    const float TOL = 1e-5f;  // pure dequantize, no rounding loss expected
    for (int b = 0; b < N_BANDS; ++b) {
        float band_max = 0.0f;
        for (int i = 0; i < COEFFS_PER_BAND[b]; ++i) {
            int idx = b * coeffs_stride + i;
            float got = coeffs_out.get_buffer()[idx];
            float ref = ref_coeffs[idx];
            float err = std::fabs(got - ref);
            if (err > band_max) band_max = err;
            if (err > max_err)  max_err  = err;
            sse += (double)err * err;
            if (err > TOL) ++n_mismatch;
        }
        std::printf("band %d max_err = %.3e\n", b, band_max);
    }
    double rms = std::sqrt(sse / total);
    std::printf("overall: max_err=%.3e rms=%.3e mismatches=%d/%d\n",
                max_err, rms, n_mismatch, total);
    std::printf("DSP avg_time = %llu us over %d iterations\n",
                (unsigned long long)avg_time, iterations);

    Result r = (n_mismatch == 0) ? Result::Pass : Result::Fail;
    TestReport tr("sp_dma_raw_probe", avg_time, "microseconds",
                  Mode::Device_Standalone, r);
    tr.print();

    packed_in.free_buff();
    coeffs_out.free_buff();
    alloc_finalize();
    return (r == Result::Pass) ? 0 : 1;
}

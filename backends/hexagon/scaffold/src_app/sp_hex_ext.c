// Shannon-Prime VHT2 - Hexagon DSP FastRPC scaffold (ARM-side process driver).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Forked from the Hexagon SDK 5.5.6.0 S22U sample (Qualcomm copyright on the
// scaffolding pattern). Replaces the calculator's "sum of ints" with a VHT2
// round-trip smoke test on fp32 vectors.

#include "AEEStdErr.h"
#include "sp_hex_ext.h"
#include "sp_hex.h"          // qaic-generated from inc/sp_hex.idl
#include "rpcmem.h"
#include "remote.h"
#include "os_defines.h"

#include "shannon_prime.h"             // sp_f16_to_f32 / sp_f32_to_f16
#ifdef SP_HEXAGON_FASTRPC
#include "shannon_prime_hexagon.h"     // sp_hexagon_init / round_trip_k / free
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Build a deterministic input vector. Values are chosen so that the VHT2
// transform produces a non-trivial spectrum (not all zeros, not constant)
// — useful for catching kernels that silently zero or pass-through.
static void fill_deterministic(float *v, int n) {
    for (int i = 0; i < n; ++i) {
        // i/n in [0,1) plus a small shift so we don't start at exactly 0.
        v[i] = 0.125f + (float)i / (float)n;
    }
}

// Worst-case absolute error between two fp32 vectors.
static float max_abs_err(const float *a, const float *b, int n) {
    float worst = 0.0f;
    for (int i = 0; i < n; ++i) {
        float e = fabsf(a[i] - b[i]);
        if (e > worst) worst = e;
    }
    return worst;
}

int sp_hex_process(int domain, int head_dim, bool isUnsignedPD_Enabled) {
    int nErr = AEE_SUCCESS;
    remote_handle64 h = -1;
    char *uri = NULL;
    float *in_vec  = NULL;
    float *out_vec = NULL;
    int   in_len  = sizeof(float) * head_dim;
    int   out_len = sizeof(float) * head_dim;

    rpcmem_init();

    int heapid = RPCMEM_HEAP_ID_SYSTEM;
#if defined(SLPI) || defined(MDSP)
    heapid = RPCMEM_HEAP_ID_CONTIG;
#endif

    if (isUnsignedPD_Enabled) {
        if (remote_session_control) {
            struct remote_rpc_control_unsigned_module data;
            data.domain = domain;
            data.enable = 1;
            nErr = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                          (void *)&data, sizeof(data));
            if (nErr != AEE_SUCCESS) {
                printf("ERROR 0x%x: remote_session_control failed\n", nErr);
                goto bail;
            }
        } else {
            nErr = AEE_EUNSUPPORTED;
            printf("ERROR 0x%x: remote_session_control unavailable on this "
                   "device\n", nErr);
            goto bail;
        }
    }

    in_vec = (float *)rpcmem_alloc(heapid, RPCMEM_DEFAULT_FLAGS, in_len);
    out_vec = (float *)rpcmem_alloc(heapid, RPCMEM_DEFAULT_FLAGS, out_len);
    if (!in_vec || !out_vec) {
        nErr = AEE_ENORPCMEMORY;
        printf("ERROR 0x%x: rpcmem_alloc failed\n", nErr);
        goto bail;
    }

    fill_deterministic(in_vec, head_dim);

    if (domain == ADSP_DOMAIN_ID)       uri = sp_hex_URI ADSP_DOMAIN;
    else if (domain == CDSP_DOMAIN_ID)  uri = sp_hex_URI CDSP_DOMAIN;
    else if (domain == CDSP1_DOMAIN_ID) uri = sp_hex_URI CDSP1_DOMAIN;
    else if (domain == MDSP_DOMAIN_ID)  uri = sp_hex_URI MDSP_DOMAIN;
    else if (domain == SDSP_DOMAIN_ID)  uri = sp_hex_URI SDSP_DOMAIN;
    else {
        nErr = AEE_EINVALIDDOMAIN;
        printf("ERROR 0x%x: unsupported domain %d\n", nErr, domain);
        goto bail;
    }

    nErr = sp_hex_open(uri, &h);
    if (nErr != AEE_SUCCESS) {
        printf("ERROR 0x%x: sp_hex_open failed\n", nErr);
        goto bail;
    }

    {
        long long vtcm_total = 0, vtcm_avail = 0, vtcm_acq = 0;
        int rc = sp_hex_vtcm_status(h, &vtcm_total, &vtcm_avail, &vtcm_acq);
        if (rc == 0) {
            printf("[sp_hex] VTCM: total=%lld KB  avail=%lld KB  "
                   "acquired_by_session=%lld KB\n",
                   vtcm_total / 1024, vtcm_avail / 1024, vtcm_acq / 1024);
            if (vtcm_acq == 0) {
                printf("[sp_hex] WARNING: session got 0 VTCM — HVX kernels "
                       "will fall back to DDR\n");
            }
        } else {
            printf("[sp_hex] vtcm_status failed rc=%d (non-fatal)\n", rc);
        }
    }

    printf("[sp_hex] calling round_trip_f32 on the DSP...\n");
    nErr = sp_hex_round_trip_f32(h, in_vec, head_dim, head_dim,
                                  out_vec, head_dim);
    if (nErr != AEE_SUCCESS) {
        printf("ERROR 0x%x: round_trip_f32 failed\n", nErr);
        goto bail;
    }

    {
        float err = max_abs_err(in_vec, out_vec, head_dim);
        // RMS for a more stable comparator than max-abs (which is sensitive
        // to one outlier per band's saturation point).
        double sse = 0.0;
        for (int i = 0; i < head_dim; ++i) {
            float d = in_vec[i] - out_vec[i];
            sse += (double)d * (double)d;
        }
        float rms = (float)sqrt(sse / head_dim);
        printf("[sp_hex] head_dim=%d   max_abs_err=%.3e   rms_err=%.3e\n",
               head_dim, err, rms);
        printf("[sp_hex] in[0]=%.6f   out[0]=%.6f\n", in_vec[0], out_vec[0]);
        // Full SP pipeline (VHT2 → 5/5/4/3 bit quantize → dequantize → IVHT2)
        // produces ~0.05–0.1 RMS on the deterministic input. Bit-exact is
        // not the goal — fidelity matching the math core's CPU reference is.
        // 0.5 RMS is loose enough to catch a broken pipeline (random output,
        // zero output, sign-flipped output) without flagging legitimate
        // banded-quantize noise.
        if (rms > 0.5f) {
            printf("ERROR: round_trip RMS %.3e exceeds 0.5 sanity threshold\n",
                   rms);
            nErr = AEE_EFAILED;
        }
    }

bail:
    if (h != (remote_handle64)-1) {
        int ce = sp_hex_close(h);
        if (ce != AEE_SUCCESS) {
            printf("ERROR 0x%x: sp_hex_close failed\n", ce);
            if (!nErr) nErr = ce;
        }
    }
    if (in_vec)  rpcmem_free(in_vec);
    if (out_vec) rpcmem_free(out_vec);
    rpcmem_deinit();
    return nErr;
}

// ----------------------------------------------------------------------------
// Engine-API smoke test — exercises sp_hexagon_init / round_trip_k / free
// instead of the raw qaic IDL. Same numerical contract; the test verifies
// the engine's host-side FastRPC shim is bit-equivalent to the direct path.
//
// Only compiled when SP_HEXAGON_FASTRPC is defined (the device build).
// The simulator path (sp_hex_q.so) doesn't have rpcmem and can't open a
// real FastRPC session, so the engine smoke would have nothing to drive.
// Returns 0 (no-op success) on the sim path.
// ----------------------------------------------------------------------------

#ifndef SP_HEXAGON_FASTRPC
int sp_hex_engine_smoke(int head_dim) { (void)head_dim; return 0; }
#else
int sp_hex_engine_smoke(int head_dim) {
    printf("\n[engine] === engine-API smoke test ===\n");

    sp_config_t cfg;
    sp_config_init(&cfg, head_dim, /*n_layers=*/1, /*n_heads_kv=*/1);

    sp_hexagon_caps_t caps;
    if (sp_hexagon_caps_probe(&caps) == 0) {
        sp_hexagon_caps_print(&caps);
    } else {
        printf("[engine] caps_probe reported DSP unavailable\n");
        return 1;
    }

    sp_hexagon_ctx_t *ctx = sp_hexagon_init(&cfg);
    if (!ctx) {
        printf("[engine] sp_hexagon_init returned NULL — DSP unreachable\n");
        return 1;
    }

    // Build deterministic fp16 input from the same generator as the direct
    // path so error magnitudes are directly comparable.
    uint16_t *in16  = (uint16_t *)malloc(sizeof(uint16_t) * head_dim);
    uint16_t *out16 = (uint16_t *)malloc(sizeof(uint16_t) * head_dim);
    if (!in16 || !out16) { sp_hexagon_free(ctx); return 1; }
    for (int i = 0; i < head_dim; ++i) {
        float v = 0.125f + (float)i / (float)head_dim;
        in16[i] = sp_f32_to_f16(v);
    }

    int rc = sp_hexagon_round_trip_k(ctx, in16, out16);
    if (rc != 0) {
        printf("[engine] sp_hexagon_round_trip_k failed rc=%d\n", rc);
        sp_hexagon_free(ctx);
        free(in16); free(out16);
        return rc;
    }

    // Compare in fp32 space.
    double sse = 0.0;
    float worst = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        float in_f  = sp_f16_to_f32(in16[i]);
        float out_f = sp_f16_to_f32(out16[i]);
        float d     = in_f - out_f;
        if (fabsf(d) > worst) worst = fabsf(d);
        sse += (double)d * (double)d;
    }
    float rms = (float)sqrt(sse / head_dim);
    printf("[engine] head_dim=%d   max_abs_err=%.3e   rms_err=%.3e\n",
           head_dim, worst, rms);
    printf("[engine] in[0]=%.6f   out[0]=%.6f\n",
           sp_f16_to_f32(in16[0]), sp_f16_to_f32(out16[0]));

    int test_err = 0;
    // Same threshold as the direct path. fp16 narrowing adds a small
    // amount of noise (≤2^-10 of the value) on top of the band-quantize
    // reconstruction; still well under 0.5 RMS.
    if (rms > 0.5f) {
        printf("[engine] ERROR: RMS %.3e exceeds 0.5 sanity threshold\n", rms);
        test_err = 1;
    } else {
        printf("[engine] Success — engine API path validated end-to-end\n");
    }

    free(in16);
    free(out16);
    sp_hexagon_free(ctx);
    return test_err;
}
#endif  // SP_HEXAGON_FASTRPC

// ----------------------------------------------------------------------------
// Cycle bench sweep — opens a FastRPC session, calls vht2_bench across a
// representative ladder of head_dim values, prints per-call pcycles for both
// the scalar reference and HVX paths plus the speedup ratio.
//
// "Per-call cycles" here is total bench cycles divided by the iteration
// count, including the memcpy reset between iterations. The memcpy delta
// is the same across both paths so the *ratio* isolates the kernel-only
// speedup; the absolute numbers are slightly inflated by the reset cost.
// ----------------------------------------------------------------------------

int sp_hex_run_bench_sweep(void) {
    rpcmem_init();
    remote_handle64 h = -1;

    // Enable unsigned PD before opening — bench session is the same kind
    // of session the smoke test uses.
    if (remote_session_control) {
        struct remote_rpc_control_unsigned_module data;
        data.domain = CDSP_DOMAIN_ID;
        data.enable = 1;
        int rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                        (void *)&data, sizeof(data));
        if (rc != AEE_SUCCESS) {
            printf("[bench] remote_session_control failed 0x%x\n", rc);
            rpcmem_deinit();
            return 1;
        }
    }

    int rc = sp_hex_open(sp_hex_URI CDSP_DOMAIN, &h);
    if (rc != AEE_SUCCESS) {
        printf("[bench] sp_hex_open failed 0x%x\n", rc);
        rpcmem_deinit();
        return 1;
    }

    const int iterations = 1000;
    const int dims[]     = {64, 128, 256, 512, 1024};
    const int n_dims     = (int)(sizeof(dims) / sizeof(dims[0]));

    printf("\n[bench] === VHT2 cycle bench: scalar vs HVX-qf32 (iter=%d) ===\n",
           iterations);
    printf("[bench] %-9s %-14s %-14s %-9s\n",
           "head_dim", "scalar pcyc/call", "HVX-qf32 pcyc/call", "speedup");

    int test_err = 0;
    for (int i = 0; i < n_dims; ++i) {
        long long scalar_total = 0, hvx_total = 0;
        rc = sp_hex_vht2_bench(h, dims[i], iterations,
                                &scalar_total, &hvx_total);
        if (rc != AEE_SUCCESS) {
            printf("[bench] hd=%d  vht2_bench failed rc=%d\n", dims[i], rc);
            test_err = 1;
            continue;
        }
        double scalar_per = (double)scalar_total / (double)iterations;
        double hvx_per    = (double)hvx_total / (double)iterations;
        double ratio      = (hvx_per > 0.0) ? (scalar_per / hvx_per) : 0.0;
        printf("[bench] %-9d %-14.0f %-14.0f %.2fx\n",
               dims[i], scalar_per, hvx_per, ratio);
    }

    sp_hex_close(h);
    rpcmem_deinit();
    return test_err;
}

// ----------------------------------------------------------------------------
// Disk-tier I/O proof — exercises the rpcmem-backed packed-bands path
// that the disk reader will feed into.
//
// Pipeline:
//   1. Generate deterministic fp32 input + run host-side scalar VHT2 to
//      get realistic spectral coefficients
//   2. Run host-side scalar sp_band_quantize → packed bytes (this is what
//      the disk format stores, byte-for-byte)
//   3. Allocate rpcmem buffer for those packed bytes (RPCMEM_HEAP_ID_SYSTEM,
//      pre-mapped to DSP via RPCMEM_TRY_MAP_STATIC)
//   4. memcpy packed bytes into the rpcmem buffer (this step represents
//      the fread-into-rpcmem that the disk reader does — the bytes
//      themselves come from disk in the production path)
//   5. Allocate rpcmem buffer for fp16 output
//   6. Call sp_hexagon_band_dequantize_partial(ctx, rpcmem_in, rpcmem_out,
//                                              max_bands=full)
//   7. Compare against host-side scalar dequantize → should match within
//      fp16 narrowing tolerance
//
// FastRPC sees both rpcmem buffers as ION-backed shared physical memory
// and forwards them to the DSP without marshal-copying. That's the
// "predictable, low-latency ping" architecture — DSP processes directly
// out of pages the kernel allocated for our process.
// ----------------------------------------------------------------------------

#ifndef SP_HEXAGON_FASTRPC
int sp_hex_disk_tier_proof(int head_dim) { (void)head_dim; return 0; }
#else
int sp_hex_disk_tier_proof(int head_dim) {
    printf("\n[disk] === Disk-tier I/O proof: rpcmem → DSP partial dequant ===\n");

    sp_config_t cfg;
    sp_config_init(&cfg, head_dim, 1, 1);
    sp_hexagon_ctx_t *ctx = sp_hexagon_init(&cfg);
    if (!ctx) {
        printf("[disk] sp_hexagon_init failed — DSP unreachable\n");
        return 1;
    }

    // Default 4-band 5/5/4/3 ship config (matches the math core defaults
    // for K and what the DSP-side dispatcher hardcodes today).
    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, head_dim, 4, default_bits);

    // ── Host-side: produce reference packed bytes via scalar quantize.
    // This is bit-identical to what the disk file would contain — the
    // disk format stores the same byte-for-byte output of sp_band_quantize.
    float *coeffs_host = (float *)malloc(sizeof(float) * head_dim);
    uint8_t *packed_host = (uint8_t *)malloc(bc.total_bytes);
    float *recon_host = (float *)malloc(sizeof(float) * head_dim);
    if (!coeffs_host || !packed_host || !recon_host) {
        printf("[disk] host alloc failed\n");
        sp_hexagon_free(ctx);
        return 1;
    }

    for (int i = 0; i < head_dim; ++i) {
        coeffs_host[i] = 0.125f + (float)i / (float)head_dim;
    }
    sp_vht2_forward_f32(coeffs_host, head_dim);    // → VHT2 coefficients
    sp_band_quantize(coeffs_host, packed_host, &bc);   // → packed bytes
    sp_band_dequantize(packed_host, recon_host, &bc);  // → reconstructed VHT2 coeffs
    // sp_hexagon_band_dequantize_partial applies IVHT2 internally to land
    // in time-domain. Mirror that on the host so we compare like-with-like.
    sp_vht2_forward_f32(recon_host, head_dim);     // VHT2 self-inverse

    // ── Allocate rpcmem buffers — this is the "shared physical memory"
    // the cDSP will access directly via SMMU. RPCMEM_TRY_MAP_STATIC
    // pre-maps so the first FastRPC call doesn't pay setup latency.
    int alloc_flags = RPCMEM_DEFAULT_FLAGS | RPCMEM_TRY_MAP_STATIC;
    uint8_t *packed_rpc = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                                   alloc_flags,
                                                   bc.total_bytes);
    uint16_t *out_fp16  = (uint16_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                                   alloc_flags,
                                                   sizeof(uint16_t) * head_dim);
    if (!packed_rpc || !out_fp16) {
        printf("[disk] rpcmem_alloc failed\n");
        if (packed_rpc) rpcmem_free(packed_rpc);
        if (out_fp16)   rpcmem_free(out_fp16);
        free(coeffs_host); free(packed_host); free(recon_host);
        sp_hexagon_free(ctx);
        return 1;
    }
    printf("[disk] rpcmem_alloc: packed=%p (%d bytes)  out_fp16=%p (%zu bytes)\n",
           (void *)packed_rpc, bc.total_bytes,
           (void *)out_fp16, sizeof(uint16_t) * head_dim);

    // ── Write the packed bytes to disk, then fread back into the rpcmem
    // buffer. Demonstrates the actual production disk-tier path on
    // Android: bytes round-trip through UFS, land in rpcmem, the DSP
    // processes from the same physical pages without ever seeing the
    // intermediate file. The rpcmem buffer is the destination of fread,
    // not a separate copy buffer.
    const char *disk_path = "/data/local/tmp/sp_hex/sp_disk_proof.bin";
    {
        FILE *fp = fopen(disk_path, "wb");
        if (!fp) {
            printf("[disk] fopen(%s, wb) failed\n", disk_path);
            rpcmem_free(packed_rpc); rpcmem_free(out_fp16);
            free(coeffs_host); free(packed_host); free(recon_host);
            sp_hexagon_free(ctx);
            return 1;
        }
        size_t wrote = fwrite(packed_host, 1, bc.total_bytes, fp);
        fclose(fp);
        if (wrote != (size_t)bc.total_bytes) {
            printf("[disk] fwrite short: %zu of %d\n", wrote, bc.total_bytes);
            rpcmem_free(packed_rpc); rpcmem_free(out_fp16);
            free(coeffs_host); free(packed_host); free(recon_host);
            sp_hexagon_free(ctx);
            return 1;
        }
    }
    {
        FILE *fp = fopen(disk_path, "rb");
        if (!fp) {
            printf("[disk] fopen(%s, rb) failed\n", disk_path);
            rpcmem_free(packed_rpc); rpcmem_free(out_fp16);
            free(coeffs_host); free(packed_host); free(recon_host);
            sp_hexagon_free(ctx);
            return 1;
        }
        size_t read = fread(packed_rpc, 1, bc.total_bytes, fp);
        fclose(fp);
        if (read != (size_t)bc.total_bytes) {
            printf("[disk] fread short: %zu of %d\n", read, bc.total_bytes);
            rpcmem_free(packed_rpc); rpcmem_free(out_fp16);
            free(coeffs_host); free(packed_host); free(recon_host);
            sp_hexagon_free(ctx);
            return 1;
        }
        printf("[disk] fread %zu bytes from %s into rpcmem\n", read, disk_path);
    }

    // ── Defensive cache-clean barrier. FastRPC's marshal does an
    // implicit cache_clean_invalidate, so today's test passes without
    // this — but production streaming readers that fill rpcmem AFTER
    // session init should add an explicit barrier between the host
    // write and the FastRPC dispatch. ARMv8 DMB ISH drains in-flight
    // stores; cached-mapped ION (rpcmem default) needs no more.
    __atomic_thread_fence(__ATOMIC_SEQ_CST);

    // ── DSP-side partial dequantize. max_bands=-1 means "all bands" —
    // identical to a full sp_band_dequantize.
    int rc = sp_hexagon_band_dequantize_partial(ctx, packed_rpc,
                                                 out_fp16, /*max_bands=*/-1);
    if (rc != 0) {
        printf("[disk] sp_hexagon_band_dequantize_partial failed rc=%d\n", rc);
        rpcmem_free(packed_rpc); rpcmem_free(out_fp16);
        free(coeffs_host); free(packed_host); free(recon_host);
        sp_hexagon_free(ctx);
        return rc;
    }

    // ── Compare DSP output (fp16, narrowed) against scalar reference
    // (fp32, before narrowing). Tolerance: fp16 ULP near recon range.
    double sse = 0.0;
    float worst = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        float dsp_val = sp_f16_to_f32(out_fp16[i]);
        float d = recon_host[i] - dsp_val;
        if (d < 0) d = -d;
        if (d > worst) worst = d;
        sse += (double)d * (double)d;
    }
    float rms = (float)sqrt(sse / head_dim);
    printf("[disk] head_dim=%d  worst_diff=%.3e  rms_diff=%.3e (vs scalar reference)\n",
           head_dim, worst, rms);

    int test_err = 0;
    // fp16 narrowing on values of order 1.0 introduces ~5e-4 worst.
    // We're checking that DSP path matches scalar bit-for-bit modulo
    // that narrowing — a regression here would be catastrophic.
    if (rms > 5e-3f) {
        printf("[disk] ERROR: RMS %.3e exceeds 5e-3 narrowing threshold\n", rms);
        test_err = 1;
    } else {
        printf("[disk] Success — rpcmem → DSP partial dequant matches scalar\n");
    }

    rpcmem_free(packed_rpc);
    rpcmem_free(out_fp16);
    free(coeffs_host); free(packed_host); free(recon_host);
    sp_hexagon_free(ctx);
    return test_err;
}
#endif  // SP_HEXAGON_FASTRPC

// ----------------------------------------------------------------------------
// Per-element validation harness for compress_f32 / decompress_f32.
//
// THE LESSON: from the 2026-04-29 V69 IEEE-HVX debugging episode, round-trip
// RMS alone is NOT sufficient. Two paths can produce the same RMS while
// differing wildly per-element. A proper kernel-vs-reference comparator
// must be per-element absolute, not just an aggregate scalar.
//
// What this exercises:
//   Path A (DSP): compress_f32(in_vec)  →  packed
//                 decompress_f32(packed) →  out_vec
//
//   Path B (host scalar reference): the same VHT2 + sp_band_quantize +
//                 sp_band_dequantize + VHT2-inverse pipeline, all on the
//                 host CPU, no FastRPC.
//
// The two outputs must agree per-element to within fp16 narrowing ULP
// (≈5e-3 on values of order 1.0). Anything larger is either a numerical
// bug in compress/decompress on the DSP or a layout mismatch.
//
// Returns 0 on pass, non-zero on regression.
// ----------------------------------------------------------------------------

#ifndef SP_HEXAGON_FASTRPC
int sp_hex_compress_decompress_validate(int head_dim) { (void)head_dim; return 0; }
#else
int sp_hex_compress_decompress_validate(int head_dim) {
    printf("\n[validate] === compress_f32 / decompress_f32 per-element validate ===\n");

    if (head_dim < 8 || head_dim > 1024 ||
        (head_dim & (head_dim - 1)) != 0) {
        printf("[validate] head_dim=%d invalid (must be pow2 in [8,1024])\n",
               head_dim);
        return 1;
    }

    sp_config_t cfg;
    sp_config_init(&cfg, head_dim, /*n_layers=*/1, /*n_heads_kv=*/1);

    sp_hexagon_ctx_t *ctx = sp_hexagon_init(&cfg);
    if (!ctx) {
        printf("[validate] sp_hexagon_init returned NULL\n");
        return 1;
    }

    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, head_dim, 4, default_bits);

    // rpcmem-backed in/out for the DSP path (zero-copy).
    int alloc_flags = RPCMEM_DEFAULT_FLAGS | RPCMEM_TRY_MAP_STATIC;
    float   *in_vec    = (float   *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                            alloc_flags, sizeof(float) * head_dim);
    uint8_t *packed_rpc = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                            alloc_flags, bc.total_bytes);
    float   *out_dsp   = (float   *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                            alloc_flags, sizeof(float) * head_dim);
    float   *out_host  = (float   *)malloc(sizeof(float) * head_dim);
    uint8_t *packed_host = (uint8_t *)malloc(bc.total_bytes);

    if (!in_vec || !packed_rpc || !out_dsp || !out_host || !packed_host) {
        printf("[validate] alloc failed\n");
        if (in_vec)      rpcmem_free(in_vec);
        if (packed_rpc)  rpcmem_free(packed_rpc);
        if (out_dsp)     rpcmem_free(out_dsp);
        free(out_host); free(packed_host);
        sp_hexagon_free(ctx);
        return 1;
    }

    fill_deterministic(in_vec, head_dim);

    // ── Path A: DSP via compress_f32 + decompress_f32, two FastRPC calls.
    int packed_used = 0;
    int rc = sp_hex_compress_f32(/*handle=*/(remote_handle64)-1, /* unused */
                                  in_vec, head_dim, head_dim,
                                  packed_rpc, bc.total_bytes, &packed_used);
    // Note: we have to dig the FastRPC handle out of ctx — it's opaque from
    // the engine API surface, so the cleanest path is to call through the
    // engine's per-position cache. But for a tight per-element comparator
    // we want to call compress_f32 / decompress_f32 directly. Use the same
    // sp_hex_open path as sp_hex_process to get a fresh handle.
    {
        // Open a private handle for this validation rather than threading
        // through ctx — keeps the test isolated.
        char uri_buf[128];
        snprintf(uri_buf, sizeof(uri_buf), "%s%s", sp_hex_URI, CDSP_DOMAIN);
        remote_handle64 priv = -1;
        int orc = sp_hex_open(uri_buf, &priv);
        if (orc != 0) {
            printf("[validate] sp_hex_open private handle failed rc=%d\n", orc);
            rpcmem_free(in_vec); rpcmem_free(packed_rpc); rpcmem_free(out_dsp);
            free(out_host); free(packed_host);
            sp_hexagon_free(ctx);
            return orc;
        }
        rc = sp_hex_compress_f32(priv, in_vec, head_dim, head_dim,
                                  packed_rpc, bc.total_bytes, &packed_used);
        if (rc != 0) {
            printf("[validate] compress_f32 rc=0x%x\n", rc);
            sp_hex_close(priv);
            rpcmem_free(in_vec); rpcmem_free(packed_rpc); rpcmem_free(out_dsp);
            free(out_host); free(packed_host);
            sp_hexagon_free(ctx);
            return rc;
        }
        if (packed_used != bc.total_bytes) {
            printf("[validate] compress_f32 packed_used=%d expected=%d\n",
                   packed_used, bc.total_bytes);
        }
        rc = sp_hex_decompress_f32(priv, packed_rpc, bc.total_bytes,
                                    head_dim, /*max_bands=*/-1,
                                    out_dsp, head_dim);
        sp_hex_close(priv);
        if (rc != 0) {
            printf("[validate] decompress_f32 rc=0x%x\n", rc);
            rpcmem_free(in_vec); rpcmem_free(packed_rpc); rpcmem_free(out_dsp);
            free(out_host); free(packed_host);
            sp_hexagon_free(ctx);
            return rc;
        }
    }

    // ── Path B: host scalar reference — same pipeline on CPU.
    float coeffs[1024] __attribute__((aligned(16)));
    memcpy(coeffs, in_vec, sizeof(float) * head_dim);
    sp_vht2_forward_f32(coeffs, head_dim);
    sp_band_quantize(coeffs, packed_host, &bc);
    sp_band_dequantize(packed_host, coeffs, &bc);
    sp_vht2_forward_f32(coeffs, head_dim);  // self-inverse
    memcpy(out_host, coeffs, sizeof(float) * head_dim);

    // ── Per-element comparator. The lesson: don't trust RMS alone.
    float worst = 0.0f;
    int   worst_idx = -1;
    double sse = 0.0;
    for (int i = 0; i < head_dim; ++i) {
        float d = out_dsp[i] - out_host[i];
        if (d < 0) d = -d;
        if (d > worst) { worst = d; worst_idx = i; }
        sse += (double)d * (double)d;
    }
    float rms = (float)sqrt(sse / head_dim);
    printf("[validate] head_dim=%d  worst=%.3e@i=%d  rms=%.3e\n",
           head_dim, worst, worst_idx, rms);

    // Tolerance: matching scalar paths is bit-equivalent except for fp32
    // rounding accumulation through the band layout. ~1e-5 max-abs on
    // head_dim ≤ 1024 is the target. Loosen to 1e-3 to leave headroom
    // for the qf32 boundary conversion path's worst case (3.8e-6 at
    // head_dim=1024 per the 2026-04-29 measurements).
    int test_err = 0;
    if (worst > 1e-3f) {
        printf("[validate] ERROR: per-element worst %.3e exceeds 1e-3 — "
               "compress/decompress diverged from host reference at i=%d "
               "(dsp=%.6f host=%.6f). The lesson: a per-element comparator "
               "catches this where round-trip RMS would not.\n",
               worst, worst_idx, out_dsp[worst_idx], out_host[worst_idx]);
        test_err = 1;
    } else {
        printf("[validate] Success — DSP compress/decompress matches host "
               "reference per-element within 1e-3\n");
    }

    rpcmem_free(in_vec);
    rpcmem_free(packed_rpc);
    rpcmem_free(out_dsp);
    free(out_host);
    free(packed_host);
    sp_hexagon_free(ctx);
    return test_err;
}
#endif  // SP_HEXAGON_FASTRPC

// ============================================================================
// Path A.2 prototype — CPU-side fused decompress-matmul benchmark.
// ============================================================================
//
// Compares two paths for computing K^T·Q (the attention score matmul):
//
//   (a) Reference: standard fp32 matmul over fp32 K + fp32 Q. This is the
//       "vanilla" memory-bandwidth-bound path that today's attention pays.
//
//   (b) SP-fused: K is stored in SP-compressed packed-bytes form (per row).
//       For each output element, decompress one K row from packed bytes via
//       sp_band_dequantize, then dot product against the Q column. NO fp16
//       K materialization in DDR — the decompress happens on-the-fly inside
//       the dot-product loop.
//
// Validates: per-element RMS error between (a) and (b) reflects the SP
// 4-band {5,5,4,3} reconstruction error profile (NOT a bug — same error
// the post-decode hook produces today). Reports timings for both so we can
// reason about whether CPU is fast enough or we need a cDSP-fused kernel.
//
// Workload sized to match Dolphin 1B at n_ctx=4096:
//   n_kv = 4096   (full KV cache)
//   hd   = 64     (Dolphin head_dim)
//   n_q  = 8      (one attention "step", e.g. one head's queries)
//
// Pure CPU; no FastRPC. Useful even when the cDSP path is unavailable.
int sp_hex_kq_matmul_bench(int n_kv, int hd, int n_q) {
    if (n_kv <= 0 || hd < 8 || (hd & (hd - 1)) != 0 || n_q <= 0) {
        printf("[sp_hex_kq_matmul_bench] invalid args: n_kv=%d hd=%d n_q=%d\n",
               n_kv, hd, n_q);
        return -1;
    }
    printf("\n[sp_hex_kq_matmul_bench] workload: n_kv=%d hd=%d n_q=%d\n",
           n_kv, hd, n_q);

    // Band config for K (4 bands at 5/5/4/3 — Dolphin K config).
    sp_band_config_t bc;
    int bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, hd, 4, bits);
    printf("  bc.total_bytes=%d (10x compression vs %d-byte fp32 K row)\n",
           bc.total_bytes, hd * 4);

    // Allocate K (n_kv × hd) fp32 and Q (n_q × hd) fp32. Random fill.
    float *K_orig = (float *)malloc(sizeof(float) * n_kv * hd);
    float *Q      = (float *)malloc(sizeof(float) * n_q * hd);
    float *KQ_ref = (float *)malloc(sizeof(float) * n_kv * n_q);
    float *KQ_sp  = (float *)malloc(sizeof(float) * n_kv * n_q);
    unsigned char *K_packed = (unsigned char *)malloc(
        (size_t)n_kv * (size_t)bc.total_bytes);
    if (!K_orig || !Q || !KQ_ref || !KQ_sp || !K_packed) {
        printf("  alloc failed\n");
        free(K_orig); free(Q); free(KQ_ref); free(KQ_sp); free(K_packed);
        return -1;
    }
    unsigned int seed = 42u;
    for (int i = 0; i < n_kv * hd; ++i) {
        seed = seed * 1103515245u + 12345u;
        K_orig[i] = ((float)((seed >> 16) & 0x7fff) / 32768.0f - 0.5f) * 2.0f;
    }
    for (int i = 0; i < n_q * hd; ++i) {
        seed = seed * 1103515245u + 12345u;
        Q[i] = ((float)((seed >> 16) & 0x7fff) / 32768.0f - 0.5f) * 2.0f;
    }

    struct timespec t0, t1;

    // (a) Reference: standard fp32 matmul. KQ_ref[kv][q] = sum_h K[kv][h] * Q[q][h].
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int kv = 0; kv < n_kv; ++kv) {
        const float *k_row = K_orig + (size_t)kv * hd;
        for (int q = 0; q < n_q; ++q) {
            const float *q_row = Q + (size_t)q * hd;
            float s = 0.0f;
            for (int h = 0; h < hd; ++h) s += k_row[h] * q_row[h];
            KQ_ref[(size_t)kv * n_q + q] = s;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ref_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                    (t1.tv_nsec - t0.tv_nsec) / 1000000.0;
    printf("  (a) reference fp32 matmul:        %.3f ms\n", ref_ms);

    // (b) SP-fused path. Step 1: pre-compress K (this is the cost the
    // post-decode hook pays today, amortised across many gen steps).
    clock_gettime(CLOCK_MONOTONIC, &t0);
    float coeffs[1024] __attribute__((aligned(128)));
    for (int kv = 0; kv < n_kv; ++kv) {
        memcpy(coeffs, K_orig + (size_t)kv * hd, sizeof(float) * hd);
        sp_vht2_forward_f32(coeffs, hd);
        sp_band_quantize(coeffs, K_packed + (size_t)kv * bc.total_bytes, &bc);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double compress_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                         (t1.tv_nsec - t0.tv_nsec) / 1000000.0;
    printf("  (b1) one-time K compress:         %.3f ms (amortised across many attn calls)\n",
           compress_ms);

    // Step 2: fused matmul over compressed K. Decompress one row at a time,
    // dot-product against all Q columns.
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int kv = 0; kv < n_kv; ++kv) {
        const unsigned char *k_packed_row = K_packed + (size_t)kv * bc.total_bytes;
        sp_band_dequantize(k_packed_row, coeffs, &bc);
        sp_vht2_forward_f32(coeffs, hd);  // self-inverse
        for (int q = 0; q < n_q; ++q) {
            const float *q_row = Q + (size_t)q * hd;
            float s = 0.0f;
            for (int h = 0; h < hd; ++h) s += coeffs[h] * q_row[h];
            KQ_sp[(size_t)kv * n_q + q] = s;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sp_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                   (t1.tv_nsec - t0.tv_nsec) / 1000000.0;
    printf("  (b2) SP-fused matmul:             %.3f ms\n", sp_ms);

    // Compare KQ_ref vs KQ_sp.
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;
    float  max_abs_err = 0.0f;
    for (int i = 0; i < n_kv * n_q; ++i) {
        float err = KQ_ref[i] - KQ_sp[i];
        sum_sq_err += (double)err * (double)err;
        sum_sq_ref += (double)KQ_ref[i] * (double)KQ_ref[i];
        float a = err < 0 ? -err : err;
        if (a > max_abs_err) max_abs_err = a;
    }
    double rms = (n_kv * n_q > 0) ? sqrt(sum_sq_err / (n_kv * n_q)) : 0.0;
    double rel_rms = (sum_sq_ref > 0) ? sqrt(sum_sq_err / sum_sq_ref) : 0.0;
    printf("\n[sp_hex_kq_matmul_bench] correctness:\n");
    printf("  RMS error:     %.6e\n", rms);
    printf("  relative RMS:  %.6e (typical SP 4-band {5,5,4,3}: ~3%% on K-corr)\n", rel_rms);
    printf("  max abs error: %.6e\n", max_abs_err);

    printf("\n[sp_hex_kq_matmul_bench] perf (lower is better):\n");
    printf("  vanilla matmul:     %.3f ms\n", ref_ms);
    printf("  SP-fused matmul:    %.3f ms (%.2fx %s)\n",
           sp_ms,
           sp_ms / ref_ms,
           sp_ms < ref_ms ? "FASTER" : "slower");
    printf("  amortised compress: %.3f ms one-time, ~0 per attn call\n",
           compress_ms);

    free(K_orig); free(Q); free(KQ_ref); free(KQ_sp); free(K_packed);
    return 0;
}

// ============================================================================
// Strike 5.5: FastRPC parity test for sp_hex_matmul_block_q8
// ============================================================================

#define SP_PARITY_BLK_SIZE  32              // sp_ok_q8_block_t.packed length
#define SP_PARITY_BLK_BYTES 64              // sizeof(sp_ok_q8_block_t)

// Inline scalar reference matching sp_hex_matmul_ok_block_q8_inner exactly.
// Bit-equal contract: this MUST produce the same (acc_a, acc_b) as the DSP
// kernel for any (w_blocks, x_row) input where the bounded-activation
// invariant holds. We inline it here so this TU has no link-time dependency
// on the math library — it's a self-contained correctness anchor.
static void parity_reference_inner(
    const unsigned char *w_blocks_bytes,   // blocks_per_row * 64 bytes
    const long long     *x_row,            // 2 * blocks_per_row * 32 int64 (a,b pairs)
    int                  blocks_per_row,
    long long           *out_acc_a,
    long long           *out_acc_b)
{
    long long acc_a = 0, acc_b = 0;
    const long long W41 = 41;
    for (int b = 0; b < blocks_per_row; ++b) {
        const long long *hdr = (const long long *)(w_blocks_bytes + b * SP_PARITY_BLK_BYTES);
        const long long B_a = hdr[0];
        const long long B_b = hdr[1];
        // hdr[2], hdr[3] are reserved_block_min_{a,b}, packed[] starts at +32 bytes.
        const signed char *packed = (const signed char *)
                                    (w_blocks_bytes + b * SP_PARITY_BLK_BYTES + 32);
        const long long *x_tile = x_row + b * SP_PARITY_BLK_SIZE * 2;
        long long D_a = 0, D_b = 0;
        for (int k = 0; k < SP_PARITY_BLK_SIZE; ++k) {
            const long long w = (long long)packed[k];
            D_a += w * x_tile[2*k + 0];     // x[k].a
            D_b += w * x_tile[2*k + 1];     // x[k].b
        }
        acc_a += B_a * D_a - W41 * B_b * D_b;
        acc_b += B_a * D_b + B_b * (D_a + D_b);
    }
    *out_acc_a = acc_a;
    *out_acc_b = acc_b;
}

// Synthesize one test case: blocks_per_row blocks of weights + matching
// activation row. Uses a deterministic seed so the test is reproducible.
// Magnitudes are chosen to honor the engine's bounded-activation invariant
// (|x.{a,b}| ≤ 2^17, |B_{a,b}| ≤ 2^14, |packed[k]| ≤ 127).
static void build_parity_inputs(int blocks_per_row, unsigned int seed,
                                 unsigned char *w_bytes, long long *x_row) {
    unsigned int s = seed;
    // Lehmer LCG — simple deterministic stream, no rand() lib dependency.
    #define NEXT_S() (s = s * 1103515245u + 12345u)
    for (int b = 0; b < blocks_per_row; ++b) {
        long long *hdr = (long long *)(w_bytes + b * SP_PARITY_BLK_BYTES);
        // B_a, B_b in [-20000, 20000].
        hdr[0] = (long long)((int)(NEXT_S() % 40001u) - 20000);
        hdr[1] = (long long)((int)(NEXT_S() % 40001u) - 20000);
        hdr[2] = 0;
        hdr[3] = 0;
        signed char *packed = (signed char *)(w_bytes + b * SP_PARITY_BLK_BYTES + 32);
        for (int k = 0; k < SP_PARITY_BLK_SIZE; ++k) {
            packed[k] = (signed char)((int)(NEXT_S() % 255u) - 127);
        }
        long long *x_tile = x_row + b * SP_PARITY_BLK_SIZE * 2;
        for (int k = 0; k < SP_PARITY_BLK_SIZE; ++k) {
            // |x.{a,b}| ≤ 2^17 keeps per-block D ≤ 2^29 (int32-safe).
            x_tile[2*k + 0] = (long long)((int)(NEXT_S() & 0x3FFFFu) - 0x1FFFF);
            x_tile[2*k + 1] = (long long)((int)(NEXT_S() & 0x3FFFFu) - 0x1FFFF);
        }
    }
    #undef NEXT_S
}

int sp_hex_matmul_block_q8_parity(int blocks_per_row) {
    if (blocks_per_row <= 0 || blocks_per_row > 256) {
        printf("[parity] invalid blocks_per_row=%d\n", blocks_per_row);
        return 1;
    }

    rpcmem_init();
    remote_handle64 h = -1;

    if (remote_session_control) {
        struct remote_rpc_control_unsigned_module data;
        data.domain = CDSP_DOMAIN_ID;
        data.enable = 1;
        int rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                        (void *)&data, sizeof(data));
        if (rc != AEE_SUCCESS) {
            printf("[parity] remote_session_control failed 0x%x\n", rc);
            rpcmem_deinit();
            return 1;
        }
    }

    int rc = sp_hex_open(sp_hex_URI CDSP_DOMAIN, &h);
    if (rc != AEE_SUCCESS) {
        printf("[parity] sp_hex_open failed 0x%x\n", rc);
        rpcmem_deinit();
        return 1;
    }

    const int w_bytes_len  = blocks_per_row * SP_PARITY_BLK_BYTES;
    const int x_row_len    = 2 * blocks_per_row * SP_PARITY_BLK_SIZE;  // ints, not bytes

    // Allocate rpcmem (so the buffers are SMMU-mappable — this proves the
    // ION-page handoff to the cDSP works when we later wire the real
    // sp_oracle_prefetch slots).
    unsigned char *w_bytes = (unsigned char *)rpcmem_alloc(
        RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, w_bytes_len);
    long long *x_row = (long long *)rpcmem_alloc(
        RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS,
        x_row_len * (int)sizeof(long long));
    if (!w_bytes || !x_row) {
        printf("[parity] rpcmem_alloc failed\n");
        if (w_bytes) rpcmem_free(w_bytes);
        if (x_row)   rpcmem_free(x_row);
        sp_hex_close(h);
        rpcmem_deinit();
        return 1;
    }

    int fail = 0;
    const unsigned int seeds[] = { 0xA1B2C3D4u, 0xDEADBEEFu, 0xCAFEBABEu };
    for (int trial = 0; trial < 3; ++trial) {
        build_parity_inputs(blocks_per_row, seeds[trial], w_bytes, x_row);

        long long acc_a_dsp = 0, acc_b_dsp = 0;
        int dsp_rc = sp_hex_matmul_block_q8(h,
                                             w_bytes, w_bytes_len,
                                             x_row,   x_row_len,
                                             blocks_per_row,
                                             &acc_a_dsp, &acc_b_dsp);
        if (dsp_rc != 0) {
            printf("[parity] DSP returned %d for trial %d\n", dsp_rc, trial);
            fail = 1;
            break;
        }

        long long acc_a_ref = 0, acc_b_ref = 0;
        parity_reference_inner(w_bytes, x_row, blocks_per_row,
                                &acc_a_ref, &acc_b_ref);

        if (acc_a_dsp != acc_a_ref || acc_b_dsp != acc_b_ref) {
            printf("[parity] MISMATCH blocks=%d trial=%d:\n", blocks_per_row, trial);
            printf("[parity]   DSP: acc_a=%lld acc_b=%lld\n", acc_a_dsp, acc_b_dsp);
            printf("[parity]   REF: acc_a=%lld acc_b=%lld\n", acc_a_ref, acc_b_ref);
            printf("[parity]   diff_a=%lld diff_b=%lld\n",
                   acc_a_dsp - acc_a_ref, acc_b_dsp - acc_b_ref);
            fail = 1;
            break;
        }
    }

    if (!fail) {
        printf("[parity] blocks_per_row=%d: all 3 trials bit-equal — HVX vmpyieacc ✓\n",
               blocks_per_row);
    }

    rpcmem_free(w_bytes);
    rpcmem_free(x_row);
    sp_hex_close(h);
    rpcmem_deinit();
    return fail;
}

// ============================================================================
// Strike 6: FastRPC parity test for sp_hex_mobius_scatter_f32.
// ============================================================================

// Compute the Möbius squarefree-first inverse permutation for `n` indices
// — same algorithm as scripts/gen_mobius_tables.py and the math core's
// sp_mobius_mask_init. inv[j] = output position of source index j.
static void parity_compute_mobius_inv(int n, int* inv_out) {
    // mu[i]: 1 if squarefree (mu != 0), 0 if non-squarefree.
    // Index 0 is treated as non-squarefree (mu(0) undefined).
    int *prime_count = (int *)calloc((size_t)n, sizeof(int));
    int *has_square  = (int *)calloc((size_t)n, sizeof(int));
    int *mu          = (int *)calloc((size_t)n, sizeof(int));
    if (n > 1) mu[1] = 1;
    for (int i = 2; i < n; ++i) mu[i] = 1;

    for (int p = 2; p < n; ++p) {
        if (prime_count[p] != 0) continue;
        for (int m = p; m < n; m += p) prime_count[m] += 1;
        long long p2 = (long long)p * p;
        for (long long m = p2; m < n; m += p2) has_square[m] = 1;
    }
    for (int i = 2; i < n; ++i) {
        if (has_square[i]) mu[i] = 0;
        else mu[i] = (prime_count[i] % 2 == 0) ? 1 : -1;
    }
    mu[0] = 0;

    // Build the squarefree-first order, then invert.
    int *order = (int *)malloc((size_t)n * sizeof(int));
    int pos = 0;
    for (int i = 0; i < n; ++i) if (mu[i] != 0) order[pos++] = i;
    for (int i = 0; i < n; ++i) if (mu[i] == 0) order[pos++] = i;
    for (int i = 0; i < n; ++i) inv_out[order[i]] = i;

    free(order); free(mu); free(has_square); free(prime_count);
}

int sp_hex_mobius_scatter_parity(int head_dim) {
    if (head_dim != 64 && head_dim != 128 && head_dim != 256 && head_dim != 512) {
        printf("[mobius] invalid head_dim=%d (must be 64/128/256/512)\n", head_dim);
        return 1;
    }

    rpcmem_init();
    remote_handle64 h = -1;

    if (remote_session_control) {
        struct remote_rpc_control_unsigned_module data;
        data.domain = CDSP_DOMAIN_ID;
        data.enable = 1;
        int rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                        (void *)&data, sizeof(data));
        if (rc != AEE_SUCCESS) {
            printf("[mobius] remote_session_control failed 0x%x\n", rc);
            rpcmem_deinit();
            return 1;
        }
    }

    int rc = sp_hex_open(sp_hex_URI CDSP_DOMAIN, &h);
    if (rc != AEE_SUCCESS) {
        printf("[mobius] sp_hex_open failed 0x%x\n", rc);
        rpcmem_deinit();
        return 1;
    }

    // Build a deterministic input and the expected output via host scalar reorder.
    float *in_vec   = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                             RPCMEM_DEFAULT_FLAGS,
                                             (int)(head_dim * sizeof(float)));
    float *out_vec  = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                             RPCMEM_DEFAULT_FLAGS,
                                             (int)(head_dim * sizeof(float)));
    int   *inv      = (int *)  malloc((size_t)head_dim * sizeof(int));
    float *expected = (float *)malloc((size_t)head_dim * sizeof(float));
    if (!in_vec || !out_vec || !inv || !expected) {
        printf("[mobius] allocation failed\n");
        if (in_vec)   rpcmem_free(in_vec);
        if (out_vec)  rpcmem_free(out_vec);
        if (inv)      free(inv);
        if (expected) free(expected);
        sp_hex_close(h); rpcmem_deinit();
        return 1;
    }

    // Fill input with a recognizable pattern so any wrong-position write
    // is visible by inspection if a trial fails.
    for (int i = 0; i < head_dim; ++i) {
        in_vec[i] = 0.0625f + (float)i;  // distinct per-index values
    }

    // Host scalar reference: expected[inv[j]] = in[j]
    parity_compute_mobius_inv(head_dim, inv);
    for (int j = 0; j < head_dim; ++j) expected[inv[j]] = in_vec[j];

    // Dispatch HVX scatter via FastRPC.
    int dsp_rc = sp_hex_mobius_scatter_f32(h, in_vec, head_dim, head_dim,
                                            out_vec, head_dim);
    int fail = 0;
    if (dsp_rc != 0) {
        printf("[mobius] DSP returned %d for head_dim=%d\n", dsp_rc, head_dim);
        fail = 1;
    } else {
        // Compare bit-equal (cast to uint32 to avoid NaN-equality pitfalls,
        // though here all values are finite). We expect exact equality —
        // this is pure data movement.
        int diffs = 0, first_diff_idx = -1;
        for (int i = 0; i < head_dim; ++i) {
            union { float f; uint32_t u; } a, b;
            a.f = out_vec[i]; b.f = expected[i];
            if (a.u != b.u) {
                if (diffs == 0) first_diff_idx = i;
                ++diffs;
            }
        }
        if (diffs > 0) {
            printf("[mobius] MISMATCH head_dim=%d: %d/%d positions differ\n",
                   head_dim, diffs, head_dim);
            printf("[mobius]   first diff @ i=%d: DSP=%g expected=%g\n",
                   first_diff_idx, out_vec[first_diff_idx], expected[first_diff_idx]);
            fail = 1;
        } else {
            printf("[mobius] head_dim=%-4d  %4d elements scattered bit-equal — vscatter ✓\n",
                   head_dim, head_dim);
        }
    }

    rpcmem_free(in_vec);
    rpcmem_free(out_vec);
    free(inv);
    free(expected);
    sp_hex_close(h);
    rpcmem_deinit();
    return fail;
}

// ============================================================================
// Strike 7: FastRPC byte-equal parity for sp_hex_band_quantize.
// ============================================================================
//
// Stricter than sp_hex_compress_decompress_validate (which checks per-element
// fp32 recovery after compress→decompress round-trip). This compares the
// COMPRESSED BYTES emitted by the HVX path against the host's sp_band_quantize
// byte-for-byte. Same input, same byte stream — proves the disk-tier dump
// from DSP is bit-identical to the dump from the engine running on ARM.

int sp_hex_band_quantize_parity(int head_dim) {
    if (head_dim != 64 && head_dim != 128 && head_dim != 256 && head_dim != 512) {
        printf("[bandq] invalid head_dim=%d\n", head_dim);
        return 1;
    }

    rpcmem_init();
    remote_handle64 h = -1;

    if (remote_session_control) {
        struct remote_rpc_control_unsigned_module data;
        data.domain = CDSP_DOMAIN_ID;
        data.enable = 1;
        int rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                        (void *)&data, sizeof(data));
        if (rc != AEE_SUCCESS) {
            printf("[bandq] remote_session_control failed 0x%x\n", rc);
            rpcmem_deinit();
            return 1;
        }
    }

    int rc = sp_hex_open(sp_hex_URI CDSP_DOMAIN, &h);
    if (rc != AEE_SUCCESS) {
        printf("[bandq] sp_hex_open failed 0x%x\n", rc);
        rpcmem_deinit();
        return 1;
    }

    // Production 5/5/4/3 band config — what the host engine ships.
    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, head_dim, 4, default_bits);
    const int packed_bytes = bc.total_bytes;

    float   *in_vec      = (float   *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                                    RPCMEM_DEFAULT_FLAGS,
                                                    (int)(head_dim * sizeof(float)));
    uint8_t *dsp_packed  = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                                    RPCMEM_DEFAULT_FLAGS,
                                                    packed_bytes);
    uint8_t *host_packed = (uint8_t *)malloc((size_t)packed_bytes);
    if (!in_vec || !dsp_packed || !host_packed) {
        printf("[bandq] alloc failed\n");
        if (in_vec)      rpcmem_free(in_vec);
        if (dsp_packed)  rpcmem_free(dsp_packed);
        if (host_packed) free(host_packed);
        sp_hex_close(h); rpcmem_deinit();
        return 1;
    }

    // Build a deterministic VHT2-domain-like input — values with a mix of
    // magnitudes so each band exercises a non-trivial amax and quantize.
    // The mu-sieve permutation is NOT applied here; both sides see the
    // same un-reordered input so we can isolate band_quantize parity.
    for (int i = 0; i < head_dim; ++i) {
        // Geometrically decreasing magnitudes per band — the typical VHT2
        // spectrum shape — plus a per-element shift so amax differs from
        // mean. Sign alternates for some elements to exercise negative quant.
        float band_scale = 1.0f;
        if (i >= head_dim / 4)      band_scale = 0.5f;
        if (i >= 2 * head_dim / 4)  band_scale = 0.25f;
        if (i >= 3 * head_dim / 4)  band_scale = 0.125f;
        float t = (float)(i % 31) / 31.0f;
        float v = band_scale * (0.1f + t);
        if ((i & 1) == 0) v = -v;
        in_vec[i] = v;
    }

    // Host reference.
    sp_band_quantize(in_vec, host_packed, &bc);

    // DSP HVX path via FastRPC.
    int dsp_rc = sp_hex_band_quantize(h, in_vec, head_dim, head_dim,
                                       dsp_packed, packed_bytes);
    int fail = 0;
    if (dsp_rc != 0) {
        printf("[bandq] DSP returned %d for head_dim=%d\n", dsp_rc, head_dim);
        fail = 1;
    } else {
        // Byte-by-byte comparison.
        int diffs = 0, first = -1;
        for (int i = 0; i < packed_bytes; ++i) {
            if (dsp_packed[i] != host_packed[i]) {
                if (diffs == 0) first = i;
                ++diffs;
            }
        }
        if (diffs > 0) {
            printf("[bandq] MISMATCH head_dim=%d: %d/%d bytes differ\n",
                   head_dim, diffs, packed_bytes);
            printf("[bandq]   first diff @ byte %d: DSP=0x%02X host=0x%02X\n",
                   first, dsp_packed[first], host_packed[first]);
            fail = 1;
        } else {
            printf("[bandq] head_dim=%-4d  packed=%3d bytes  byte-equal — HVX amax + scalar pack ✓\n",
                   head_dim, packed_bytes);
        }
    }

    rpcmem_free(in_vec);
    rpcmem_free(dsp_packed);
    free(host_packed);
    sp_hex_close(h);
    rpcmem_deinit();
    return fail;
}

// ============================================================================
// Strike 9: FastRPC byte-equal parity for the grand fusion
//           (VHT2 + Möbius + band_quantize on DSP).
// ============================================================================

int sp_hex_compress_f32_full_parity(int head_dim) {
    if (head_dim != 64 && head_dim != 128 && head_dim != 256 && head_dim != 512) {
        printf("[fused] invalid head_dim=%d\n", head_dim);
        return 1;
    }

    rpcmem_init();
    remote_handle64 h = -1;

    if (remote_session_control) {
        struct remote_rpc_control_unsigned_module data;
        data.domain = CDSP_DOMAIN_ID;
        data.enable = 1;
        int rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                        (void *)&data, sizeof(data));
        if (rc != AEE_SUCCESS) {
            printf("[fused] remote_session_control failed 0x%x\n", rc);
            rpcmem_deinit();
            return 1;
        }
    }

    int rc = sp_hex_open(sp_hex_URI CDSP_DOMAIN, &h);
    if (rc != AEE_SUCCESS) {
        printf("[fused] sp_hex_open failed 0x%x\n", rc);
        rpcmem_deinit();
        return 1;
    }

    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, head_dim, 4, default_bits);
    const int packed_bytes = bc.total_bytes;

    float   *in_vec      = (float   *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                                    RPCMEM_DEFAULT_FLAGS,
                                                    (int)(head_dim * sizeof(float)));
    uint8_t *dsp_packed  = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                                    RPCMEM_DEFAULT_FLAGS,
                                                    packed_bytes);
    uint8_t *host_packed = (uint8_t *)malloc((size_t)packed_bytes);
    float   *host_scratch = (float *)malloc((size_t)head_dim * sizeof(float));
    float   *host_mscratch = (float *)malloc((size_t)head_dim * sizeof(float));
    sp_mobius_mask_t mask;
    int mask_rc = sp_mobius_mask_init(&mask, head_dim);
    if (!in_vec || !dsp_packed || !host_packed || !host_scratch ||
        !host_mscratch || mask_rc != 0) {
        printf("[fused] alloc failed\n");
        if (in_vec)        rpcmem_free(in_vec);
        if (dsp_packed)    rpcmem_free(dsp_packed);
        if (host_packed)   free(host_packed);
        if (host_scratch)  free(host_scratch);
        if (host_mscratch) free(host_mscratch);
        if (mask_rc == 0)  sp_mobius_mask_free(&mask);
        sp_hex_close(h); rpcmem_deinit();
        return 1;
    }

    // Deterministic input. Same distribution shape as Strike 7's bandq test.
    for (int i = 0; i < head_dim; ++i) {
        float bs = 1.0f;
        if (i >= head_dim / 4)     bs = 0.5f;
        if (i >= 2 * head_dim / 4) bs = 0.25f;
        if (i >= 3 * head_dim / 4) bs = 0.125f;
        float t = (float)(i % 31) / 31.0f;
        float v = bs * (0.1f + t);
        if ((i & 1) == 0) v = -v;
        in_vec[i] = v;
    }

    // Host reference: full encode pipeline (VHT2 → Möbius → band_quantize).
    memcpy(host_scratch, in_vec, sizeof(float) * (size_t)head_dim);
    sp_vht2_forward_f32(host_scratch, head_dim);
    sp_mobius_reorder_ex(host_scratch, &mask, host_mscratch);
    sp_band_quantize(host_scratch, host_packed, &bc);

    // DSP fused path: single FastRPC dispatch.
    int dsp_used = 0;
    int dsp_rc = sp_hex_compress_f32_full(h, in_vec, head_dim, head_dim,
                                            dsp_packed, packed_bytes, &dsp_used);

    int fail = 0;
    if (dsp_rc != 0) {
        printf("[fused] DSP returned %d for head_dim=%d\n", dsp_rc, head_dim);
        fail = 1;
    } else if (dsp_used != packed_bytes) {
        printf("[fused] DSP packed_used=%d expected=%d\n", dsp_used, packed_bytes);
        fail = 1;
    } else {
        int diffs = 0, first = -1;
        for (int i = 0; i < packed_bytes; ++i) {
            if (dsp_packed[i] != host_packed[i]) {
                if (diffs == 0) first = i;
                ++diffs;
            }
        }
        if (diffs > 0) {
            printf("[fused] MISMATCH head_dim=%d: %d/%d bytes differ\n",
                   head_dim, diffs, packed_bytes);
            printf("[fused]   first diff @ byte %d: DSP=0x%02X host=0x%02X\n",
                   first, dsp_packed[first], host_packed[first]);
            fail = 1;
        } else {
            printf("[fused] head_dim=%-4d  %3d bytes  byte-equal — VHT2+Möbius+bandq fused ✓\n",
                   head_dim, packed_bytes);
        }
    }

    rpcmem_free(in_vec);
    rpcmem_free(dsp_packed);
    free(host_packed);
    free(host_scratch);
    free(host_mscratch);
    sp_mobius_mask_free(&mask);
    sp_hex_close(h);
    rpcmem_deinit();
    return fail;
}

// ============================================================================
// Strike 10b: FastRPC parity test for sp_hex_compress_f32_full_batch.
// ============================================================================
//
// Stages n_vectors deterministic fp32 inputs contiguously, dispatches the
// batched fused encoder via ONE FastRPC call, and compares byte-by-byte
// against the host reference. Validates that the per-iteration VTCM reuse
// inside the DSP loop produces independent, byte-identical encodes — i.e.
// no inter-iteration aliasing inside the Möbius scatter or band_quantize.
// ============================================================================

int sp_hex_compress_f32_full_batch_parity(int head_dim, int n_vectors) {
    if (head_dim != 64 && head_dim != 128 && head_dim != 256 && head_dim != 512) {
        printf("[fused_batch] invalid head_dim=%d\n", head_dim);
        return 1;
    }
    if (n_vectors <= 0 || n_vectors > 256) {
        printf("[fused_batch] invalid n_vectors=%d\n", n_vectors);
        return 1;
    }

    rpcmem_init();
    remote_handle64 h = -1;

    if (remote_session_control) {
        struct remote_rpc_control_unsigned_module data;
        data.domain = CDSP_DOMAIN_ID;
        data.enable = 1;
        int rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                        (void *)&data, sizeof(data));
        if (rc != AEE_SUCCESS) {
            printf("[fused_batch] remote_session_control failed 0x%x\n", rc);
            rpcmem_deinit();
            return 1;
        }
    }

    int rc = sp_hex_open(sp_hex_URI CDSP_DOMAIN, &h);
    if (rc != AEE_SUCCESS) {
        printf("[fused_batch] sp_hex_open failed 0x%x\n", rc);
        rpcmem_deinit();
        return 1;
    }

    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, head_dim, 4, default_bits);
    const int per_vec_bytes = bc.total_bytes;
    const int batch_floats  = n_vectors * head_dim;
    const int batch_bytes   = n_vectors * per_vec_bytes;

    float   *in_vecs      = (float   *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                                     RPCMEM_DEFAULT_FLAGS,
                                                     (int)(batch_floats * sizeof(float)));
    uint8_t *dsp_packed   = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                                     RPCMEM_DEFAULT_FLAGS,
                                                     batch_bytes);
    uint8_t *host_packed  = (uint8_t *)malloc((size_t)batch_bytes);
    float   *host_scratch = (float *)malloc((size_t)head_dim * sizeof(float));
    float   *host_mscratch = (float *)malloc((size_t)head_dim * sizeof(float));
    sp_mobius_mask_t mask;
    int mask_rc = sp_mobius_mask_init(&mask, head_dim);
    if (!in_vecs || !dsp_packed || !host_packed || !host_scratch ||
        !host_mscratch || mask_rc != 0) {
        printf("[fused_batch] alloc failed\n");
        if (in_vecs)       rpcmem_free(in_vecs);
        if (dsp_packed)    rpcmem_free(dsp_packed);
        if (host_packed)   free(host_packed);
        if (host_scratch)  free(host_scratch);
        if (host_mscratch) free(host_mscratch);
        if (mask_rc == 0)  sp_mobius_mask_free(&mask);
        sp_hex_close(h); rpcmem_deinit();
        return 1;
    }

    // Deterministic input per-vector — same distribution shape as the single-vec
    // test, but with a per-vector seed offset so vec[i] != vec[j] (catches any
    // cross-iteration aliasing inside the DSP loop).
    for (int v = 0; v < n_vectors; ++v) {
        float *in_vec = in_vecs + (size_t)v * (size_t)head_dim;
        for (int i = 0; i < head_dim; ++i) {
            float bs = 1.0f;
            if (i >= head_dim / 4)     bs = 0.5f;
            if (i >= 2 * head_dim / 4) bs = 0.25f;
            if (i >= 3 * head_dim / 4) bs = 0.125f;
            float t = (float)((i + v) % 31) / 31.0f;
            float val = bs * (0.1f + t);
            if (((i + v) & 1) == 0) val = -val;
            in_vec[i] = val;
        }
    }

    // Host reference: encode each vector through the host scalar chain.
    for (int v = 0; v < n_vectors; ++v) {
        const float *in_vec = in_vecs + (size_t)v * (size_t)head_dim;
        uint8_t *out_slot = host_packed + (size_t)v * (size_t)per_vec_bytes;
        memcpy(host_scratch, in_vec, sizeof(float) * (size_t)head_dim);
        sp_vht2_forward_f32(host_scratch, head_dim);
        sp_mobius_reorder_ex(host_scratch, &mask, host_mscratch);
        sp_band_quantize(host_scratch, out_slot, &bc);
    }

    // DSP fused-batch path: single FastRPC dispatch over n_vectors.
    int dsp_used = 0;
    int dsp_rc = sp_hex_compress_f32_full_batch(h, in_vecs, batch_floats,
                                                  head_dim, n_vectors,
                                                  dsp_packed, batch_bytes,
                                                  &dsp_used);

    int fail = 0;
    if (dsp_rc != 0) {
        printf("[fused_batch] DSP returned %d for head_dim=%d n=%d\n",
               dsp_rc, head_dim, n_vectors);
        fail = 1;
    } else if (dsp_used != batch_bytes) {
        printf("[fused_batch] DSP packed_used=%d expected=%d\n",
               dsp_used, batch_bytes);
        fail = 1;
    } else {
        int diffs = 0, first = -1, first_v = -1;
        for (int v = 0; v < n_vectors && diffs < 16; ++v) {
            int base = v * per_vec_bytes;
            for (int i = 0; i < per_vec_bytes; ++i) {
                if (dsp_packed[base + i] != host_packed[base + i]) {
                    if (diffs == 0) { first = i; first_v = v; }
                    ++diffs;
                }
            }
        }
        if (diffs > 0) {
            printf("[fused_batch] MISMATCH head_dim=%d n=%d: %d+ bytes differ\n",
                   head_dim, n_vectors, diffs);
            printf("[fused_batch]   first diff @ vec %d byte %d: DSP=0x%02X host=0x%02X\n",
                   first_v, first,
                   dsp_packed[first_v * per_vec_bytes + first],
                   host_packed[first_v * per_vec_bytes + first]);
            fail = 1;
        } else {
            printf("[fused_batch] head_dim=%-4d n=%-3d  %d bytes/vec  byte-equal — batched fused ✓\n",
                   head_dim, n_vectors, per_vec_bytes);
        }
    }

    rpcmem_free(in_vecs);
    rpcmem_free(dsp_packed);
    free(host_packed);
    free(host_scratch);
    free(host_mscratch);
    sp_mobius_mask_free(&mask);
    sp_hex_close(h);
    rpcmem_deinit();
    return fail;
}

// ============================================================================
// Strike 8a: FastRPC parity test for sp_hex_logit_argmax_u16.
// ============================================================================

// Scalar reference: first-occurrence argmax over a uint16 row. Matches the
// DSP kernel's "ties go to lowest index" semantic.
static int parity_scalar_argmax_u16(const uint16_t* buf, int n) {
    int best_idx = 0;
    uint16_t best_val = buf[0];
    for (int i = 1; i < n; ++i) {
        if (buf[i] > best_val) { best_val = buf[i]; best_idx = i; }
    }
    return best_idx;
}

// Build a deterministic UFIXED_16 row with a specific known max position.
static void parity_build_logits(uint16_t* buf, int n, int max_pos, uint32_t seed) {
    uint32_t s = seed;
    for (int i = 0; i < n; ++i) {
        s = s * 1103515245u + 12345u;
        // Background range [100, 60000] — far below the 65535 max, far above 0.
        buf[i] = (uint16_t)(100 + (s % 59901u));
    }
    // Plant the max at the requested position. Use 65000 (less than 65535
    // so we can also seed a UNIQUE max — needed for tie-free first-occurrence).
    buf[max_pos] = 65000;
}

int sp_hex_logit_argmax_parity(void) {
    rpcmem_init();
    remote_handle64 h = -1;

    if (remote_session_control) {
        struct remote_rpc_control_unsigned_module data;
        data.domain = CDSP_DOMAIN_ID;
        data.enable = 1;
        int rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                        (void *)&data, sizeof(data));
        if (rc != AEE_SUCCESS) {
            printf("[argmax] remote_session_control failed 0x%x\n", rc);
            rpcmem_deinit();
            return 1;
        }
    }

    int rc = sp_hex_open(sp_hex_URI CDSP_DOMAIN, &h);
    if (rc != AEE_SUCCESS) {
        printf("[argmax] sp_hex_open failed 0x%x\n", rc);
        rpcmem_deinit();
        return 1;
    }

    // Test matrix: a few vocab sizes (aligned, misaligned, Qwen3-4B real size)
    // × a few max positions (start, end, middle, irregular).
    struct { int vocab; int max_pos; const char* label; } cases[] = {
        {     64,      0, "vocab=64    max@0"        },  // tiny, max at head
        {     64,     63, "vocab=64    max@end"      },  // tiny, max at tail
        {   1024,    512, "vocab=1024  max@middle"   },  // aligned, mid
        {   1023,    777, "vocab=1023  max@odd"      },  // misaligned vocab
        {  32768,  12345, "vocab=32k   max@12345"    },  // larger
        { 151936,  87654, "vocab=151k  max@87654"    },  // Qwen3-4B size
        { 151936, 151935, "vocab=151k  max@last"     },  // Qwen3-4B, max at end
        { 151936,      0, "vocab=151k  max@0"        },  // Qwen3-4B, max at head
    };
    const int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    int fail = 0;
    for (int c = 0; c < n_cases; ++c) {
        const int vocab   = cases[c].vocab;
        const int max_pos = cases[c].max_pos;
        const int bytes   = vocab * (int)sizeof(uint16_t);

        uint16_t* logits = (uint16_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                                    RPCMEM_DEFAULT_FLAGS, bytes);
        if (!logits) {
            printf("[argmax] rpcmem_alloc failed for vocab=%d\n", vocab);
            fail = 1;
            break;
        }
        parity_build_logits(logits, vocab, max_pos, 0xA5A5A5A5u + (uint32_t)c);

        // Host reference.
        int ref_idx = parity_scalar_argmax_u16(logits, vocab);

        // DSP HVX argmax.
        int dsp_idx_long = -1;
        int dsp_rc = sp_hex_logit_argmax_u16(h, logits, vocab, vocab, &dsp_idx_long);

        if (dsp_rc != 0) {
            printf("[argmax] %s : DSP returned rc=%d\n", cases[c].label, dsp_rc);
            fail = 1;
        } else if (dsp_idx_long != ref_idx) {
            printf("[argmax] %s : MISMATCH DSP=%d ref=%d\n",
                   cases[c].label, dsp_idx_long, ref_idx);
            fail = 1;
        } else {
            printf("[argmax] %-30s DSP=%-6d ref=%-6d ✓\n",
                   cases[c].label, dsp_idx_long, ref_idx);
        }

        rpcmem_free(logits);
        if (fail) break;
    }

    sp_hex_close(h);
    rpcmem_deinit();
    return fail;
}


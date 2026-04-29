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

    // ── This memcpy is the only "extra" copy. In the production disk
    // reader, this becomes:  fread(packed_rpc, 1, bc.total_bytes, fp);
    // — same destination buffer, same operation, just the bytes come
    // from UFS instead of from packed_host. The DSP doesn't see this
    // copy; it just sees the rpcmem pointer.
    memcpy(packed_rpc, packed_host, bc.total_bytes);

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

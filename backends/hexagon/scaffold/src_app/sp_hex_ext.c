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

    printf("[sp_hex] calling round_trip_f32 on the DSP...\n");
    nErr = sp_hex_round_trip_f32(h, in_vec, head_dim, head_dim,
                                  out_vec, head_dim);
    if (nErr != AEE_SUCCESS) {
        printf("ERROR 0x%x: round_trip_f32 failed\n", nErr);
        goto bail;
    }

    {
        float err = max_abs_err(in_vec, out_vec, head_dim);
        printf("[sp_hex] head_dim=%d   max_abs_err=%.3e   in[0]=%.6f   "
               "out[0]=%.6f\n", head_dim, err, in_vec[0], out_vec[0]);
        // VHT2 self-inverse round trip — expect bit-exact (fp epsilon)
        // with the scaffold's no-quantize path.
        if (err > 1e-4f) {
            printf("ERROR: round_trip error %.3e exceeds 1e-4 threshold\n",
                   err);
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

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
#include "HAP_vtcm_mgr.h"      // HAP_request_VTCM / HAP_release_VTCM / queries
#include "HAP_perf.h"          // HAP_perf_get_pcycles for the bench IDL
#include "qurt_hvx.h"          // qurt_hvx_lock / unlock — required before HVX
#include "sp_hex.h"            // qaic-generated header from sp_hex.idl
#include "sp_hex_kernels.h"    // forwards to SP math core
#include "shannon_prime.h"     // sp_band_config_t / sp_band_quantize / etc.

// Forward decl — defined in sp_hex_kernels_hvx.c when HVX is available.
// Used directly by the bench IDL to time the HVX path in isolation,
// vs sp_hex_vht2_f32 which dispatches and would obscure the comparison.
#ifdef __HVX__
void sp_hex_vht2_f32_hvx(float *data, int n);
#endif

// Per-session DSP context. Held behind the FastRPC handle that sp_hex_open
// returns; sp_hex_close releases everything. Sized to fit a few small
// pieces of state — VTCM pointer, region size, anything else the kernels
// will need at the session level.
//
// VTCM is V69's tightly-coupled SRAM (~8 MB total on this DSP variant).
// Acquiring a small region per session ensures HVX kernels have a fast
// scratch path; HAP_request_VTCM returning NULL means the kernels will
// fall back to DDR, with predictably worse perf.
typedef struct {
    void   *vtcm_ptr;       // NULL if acquisition failed
    int     vtcm_bytes;     // 0 if no region; else bytes acquired
    int     hvx_locked;     // 1 if qurt_hvx_lock succeeded — required
                            //   before any HVX instruction executes on
                            //   this thread; without it, vector ops
                            //   fault and FastRPC returns a transport
                            //   error code (the 78 / -2147482611 / 39
                            //   chaos we saw before locking).
} sp_hex_session_t;

// Default per-session VTCM ask. Banded quantize working set on a typical
// head_dim is well under 16 KB; 64 KB gives headroom for staging packed
// bytes + scale headers + a few HVX register spills if needed. Tunable
// once we have actual HVX kernels showing pressure.
#define SP_HEX_VTCM_BYTES   (64 * 1024)

// ============================================================================
// Lifecycle - inherited from remote_handle64.
// ============================================================================

int sp_hex_open(const char *uri, remote_handle64 *handle) {
    (void)uri;
    sp_hex_session_t *sess = (sp_hex_session_t *)calloc(1, sizeof(*sess));
    if (!sess) return -1;

    // Lock the HVX context for this thread. Required before any HVX
    // instruction executes — without this lock, the first vector op
    // throws a fault and FastRPC returns a transport error to the host.
    // V69 is 128-byte (1024-bit) HVX so we ask for QURT_HVX_MODE_128B.
    // 0 on success; negative if HVX is unavailable for any reason.
    int hvx_rc = qurt_hvx_lock(QURT_HVX_MODE_128B);
    sess->hvx_locked = (hvx_rc == 0);

    // Try to acquire a VTCM region. single_page_flag=1 asks for one
    // contiguous physical page — fine for 64 KB which is well under
    // the V69 page granularity. NULL return means "couldn't get one";
    // the session is still usable, kernels just run against DDR.
    sess->vtcm_ptr = HAP_request_VTCM(SP_HEX_VTCM_BYTES, /*single_page=*/1);
    sess->vtcm_bytes = (sess->vtcm_ptr != NULL) ? SP_HEX_VTCM_BYTES : 0;

    *handle = (remote_handle64)sess;
    FARF(RUNTIME_HIGH,
         "[sp_hex] open() -> handle=0x%llx hvx_locked=%d vtcm=%p (%d bytes)",
         *handle, sess->hvx_locked, sess->vtcm_ptr, sess->vtcm_bytes);
    return 0;
}

int sp_hex_close(remote_handle64 handle) {
    sp_hex_session_t *sess = (sp_hex_session_t *)handle;
    if (sess) {
        if (sess->vtcm_ptr) {
            HAP_release_VTCM(sess->vtcm_ptr);
        }
        if (sess->hvx_locked) {
            qurt_hvx_unlock();
        }
        free(sess);
    }
    FARF(RUNTIME_HIGH, "[sp_hex] close()");
    return 0;
}

int sp_hex_vtcm_status(remote_handle64 h,
                        long long *bytes_total,
                        long long *bytes_avail,
                        long long *bytes_acquired) {
    sp_hex_session_t *sess = (sp_hex_session_t *)h;
    if (!bytes_total || !bytes_avail || !bytes_acquired) return -1;

    unsigned int page_size = 0, page_count = 0;
    if (HAP_query_total_VTCM(&page_size, &page_count) == 0) {
        *bytes_total = (long long)page_size * (long long)page_count;
    } else {
        *bytes_total = 0;
    }

    unsigned int avail_block = 0, max_page = 0, num_pages = 0;
    if (HAP_query_avail_VTCM(&avail_block, &max_page, &num_pages) == 0) {
        // avail_block is the largest contiguous request that would
        // succeed right now; close enough to "avail" for our purposes.
        *bytes_avail = (long long)avail_block;
    } else {
        *bytes_avail = 0;
    }

    *bytes_acquired = sess ? (long long)sess->vtcm_bytes : 0;
    FARF(RUNTIME_HIGH,
         "[sp_hex] vtcm_status: total=%lld avail=%lld acquired=%lld",
         *bytes_total, *bytes_avail, *bytes_acquired);
    return 0;
}

int sp_hex_vht2_bench(remote_handle64 h, int head_dim, int iterations,
                       long long *scalar_pcycles, long long *hvx_pcycles) {
    (void)h;
    if (!scalar_pcycles || !hvx_pcycles) return -1;
    if (head_dim < 8 || head_dim > 1024) return -1;
    if ((head_dim & (head_dim - 1)) != 0) return -1;  // pow2 only
    if (iterations < 1) return -1;

    // 128-byte aligned for HVX vmem. 8 KB total — fits unsigned PD stack.
    float input[1024] __attribute__((aligned(128)));
    float buf[1024]   __attribute__((aligned(128)));

    for (int i = 0; i < head_dim; ++i) {
        input[i] = 0.125f + (float)i / (float)head_dim;
    }

    // Time scalar path
    memcpy(buf, input, sizeof(float) * head_dim);
    sp_vht2_forward_f32(buf, head_dim);  // warmup
    uint64_t t0 = HAP_perf_get_pcycles();
    for (int it = 0; it < iterations; ++it) {
        memcpy(buf, input, sizeof(float) * head_dim);
        sp_vht2_forward_f32(buf, head_dim);
    }
    *scalar_pcycles = (long long)(HAP_perf_get_pcycles() - t0);

    // Time HVX path (kept for perf measurement even though dispatcher
    // doesn't currently use it — see note in sp_hex_kernels.c).
#ifdef __HVX__
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) == 0) {
        memcpy(buf, input, sizeof(float) * head_dim);
        sp_hex_vht2_f32_hvx(buf, head_dim);  // warmup
        t0 = HAP_perf_get_pcycles();
        for (int it = 0; it < iterations; ++it) {
            memcpy(buf, input, sizeof(float) * head_dim);
            sp_hex_vht2_f32_hvx(buf, head_dim);
        }
        *hvx_pcycles = (long long)(HAP_perf_get_pcycles() - t0);
        qurt_hvx_unlock();
    } else {
        *hvx_pcycles = *scalar_pcycles;
    }
#else
    *hvx_pcycles = *scalar_pcycles;
#endif
    return 0;
}

// ============================================================================
// IDL methods.
// ============================================================================

// Full SP round-trip on the cDSP:
//   in_vec  --VHT2-->  coeffs  --quantize-->  packed bytes
//   packed  --dequantize-->  coeffs  --VHT2-->  out_vec
// VHT2 is self-inverse so the second VHT2 is the inverse. Reconstruction
// error matches what the math core produces on CPU for the same input —
// not fp32 epsilon (that's the no-quantize path).
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

    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, head_dim, 4, default_bits);

    if (bc.total_bytes > 4096) {
        FARF(ERROR, "[sp_hex] round_trip: packed size %d > scratch 4096",
             bc.total_bytes);
        return -1;
    }
    // 128-byte aligned for HVX vmem (the dispatcher routes large enough
    // head_dim through HVX and the kernels do strict-alignment loads).
    unsigned char packed[4096] __attribute__((aligned(128)));
    float coeffs[1024]         __attribute__((aligned(128)));
    if (head_dim > 1024) {
        FARF(ERROR, "[sp_hex] round_trip: head_dim=%d > scratch 1024",
             head_dim);
        return -1;
    }

    // 1. Copy in, VHT2 forward via dispatcher (currently scalar — HVX
    // VHT2 has a precision drift issue noted in sp_hex_kernels.c).
    memcpy(coeffs, in_vec, sizeof(float) * head_dim);
    sp_hex_vht2_f32(coeffs, head_dim);

    // 2. Quantize via dispatcher — HVX path activates for head_dim ≥ 128
    // multiple-of-128, ternary mask zero (the default).
    int packed_used = 0;
    sp_hex_band_quantize_scalar(coeffs, head_dim, packed, sizeof(packed),
                                &packed_used);

    // 3. Dequantize back to coeffs (still scalar — HVX dequantize TODO).
    sp_band_dequantize(packed, coeffs, &bc);

    // 4. VHT2 inverse via dispatcher.
    sp_hex_vht2_f32(coeffs, head_dim);
    memcpy(out_vec, coeffs, sizeof(float) * head_dim);

    FARF(RUNTIME_HIGH,
         "[sp_hex] round_trip: hd=%d packed=%d in[0]=%f out[0]=%f",
         head_dim, bc.total_bytes, in_vec[0], out_vec[0]);
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

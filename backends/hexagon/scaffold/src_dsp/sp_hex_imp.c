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
#include "qurt_hvx.h"          // qurt_hvx_lock / unlock ? required before HVX
#include "sp_hex.h"            // qaic-generated header from sp_hex.idl
#include "sp_hex_kernels.h"    // forwards to SP math core
#include "shannon_prime.h"     // sp_band_config_t / sp_band_quantize / etc.

// Forward decl ? defined in sp_hex_kernels_hvx.c when HVX is available.
// Used directly by the bench IDL to time the HVX path in isolation,
// vs sp_hex_vht2_f32 which dispatches and would obscure the comparison.
#ifdef __HVX__
void sp_hex_vht2_f32_hvx(float *data, int n);
void sp_hex_vht2_f32_hvx_qf32(float *data, int n);
#endif

// Per-session DSP context. Held behind the FastRPC handle that sp_hex_open
// returns; sp_hex_close releases everything. Sized to fit a few small
// pieces of state ? VTCM pointer, region size, anything else the kernels
// will need at the session level.
//
// VTCM is V69's tightly-coupled SRAM (~8 MB total on this DSP variant).
// Acquiring a small region per session ensures HVX kernels have a fast
// scratch path; HAP_request_VTCM returning NULL means the kernels will
// fall back to DDR, with predictably worse perf.
typedef struct {
    void   *vtcm_ptr;       // NULL if acquisition failed
    int     vtcm_bytes;     // 0 if no region; else bytes acquired
    int     hvx_locked;     // 1 if qurt_hvx_lock succeeded ? required
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
    // instruction executes ? without this lock, the first vector op
    // throws a fault and FastRPC returns a transport error to the host.
    // V69 is 128-byte (1024-bit) HVX so we ask for QURT_HVX_MODE_128B.
    // 0 on success; negative if HVX is unavailable for any reason.
    int hvx_rc = qurt_hvx_lock(QURT_HVX_MODE_128B);
    sess->hvx_locked = (hvx_rc == 0);

    // Try to acquire a VTCM region. single_page_flag=1 asks for one
    // contiguous physical page ? fine for 64 KB which is well under
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

    // Cycle bench ? scalar reference vs HVX-qf32 (the numerically correct
    // HVX path). The IEEE-HVX kernel is structurally broken on V69 so we
    // time qf32; user can A/B by editing the kernel call below.
    float input[1024] __attribute__((aligned(128)));
    float buf[1024]   __attribute__((aligned(128)));
    for (int i = 0; i < head_dim; ++i) {
        input[i] = 0.125f + (float)i / (float)head_dim;
    }

    // Scalar timing
    memcpy(buf, input, sizeof(float) * head_dim);
    sp_vht2_forward_f32(buf, head_dim);  // warmup
    uint64_t t0 = HAP_perf_get_pcycles();
    for (int it = 0; it < iterations; ++it) {
        memcpy(buf, input, sizeof(float) * head_dim);
        sp_vht2_forward_f32(buf, head_dim);
    }
    *scalar_pcycles = (long long)(HAP_perf_get_pcycles() - t0);

    // HVX-qf32 timing
#ifdef __HVX__
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) == 0) {
        memcpy(buf, input, sizeof(float) * head_dim);
        sp_hex_vht2_f32_hvx_qf32(buf, head_dim);  // warmup
        t0 = HAP_perf_get_pcycles();
        for (int it = 0; it < iterations; ++it) {
            memcpy(buf, input, sizeof(float) * head_dim);
            sp_hex_vht2_f32_hvx_qf32(buf, head_dim);
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
// error matches what the math core produces on CPU for the same input ?
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

    // 1. Copy in, VHT2 forward via dispatcher (currently scalar ? HVX
    // VHT2 has a precision drift issue noted in sp_hex_kernels.c).
    memcpy(coeffs, in_vec, sizeof(float) * head_dim);
    sp_hex_vht2_f32(coeffs, head_dim);

    // 2. Quantize via dispatcher ? HVX path activates for head_dim ? 128
    // multiple-of-128, ternary mask zero (the default).
    int packed_used = 0;
    sp_hex_band_quantize_scalar(coeffs, head_dim, packed, sizeof(packed),
                                &packed_used);

    // 3. Dequantize back to coeffs (still scalar ? HVX dequantize TODO).
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

// compress_f32 ? full encode pipeline in one FastRPC dispatch.
// Mirrors the first half of sp_hex_round_trip_f32 (memcpy, VHT2, quantize)
// without the dequantize+inverse-VHT2 tail. Used by the host's
// sp_hexagon_cache_write_{k,v} to land a per-position raw fp32 vector
// into compressed bytes in one call.
//
// The packed_used out-param tells the host exactly how many bytes were
// written into out_packed. Today this is bc.total_bytes for the {5,5,4,3}
// ship config ? fixed for a given head_dim ? but exposing it explicitly
// future-proofs against per-call band-config changes.
int sp_hex_compress_f32(remote_handle64 h,
                         const float *in_vec, int in_len,
                         int head_dim,
                         unsigned char *out_packed, int out_capacity,
                         int *packed_used) {
    (void)h;
    if (!packed_used) return -1;
    *packed_used = 0;
    if (in_len != head_dim) {
        FARF(ERROR, "[sp_hex] compress: length mismatch in=%d hd=%d",
             in_len, head_dim);
        return -1;
    }
    if (head_dim < 8 || head_dim > 1024 ||
        (head_dim & (head_dim - 1)) != 0) {
        FARF(ERROR, "[sp_hex] compress: head_dim=%d must be pow2 in [8,1024]",
             head_dim);
        return -1;
    }

    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, head_dim, 4, default_bits);
    if (bc.total_bytes > out_capacity) {
        FARF(ERROR, "[sp_hex] compress: packed=%d > capacity=%d",
             bc.total_bytes, out_capacity);
        return -1;
    }

    // 128-byte aligned for HVX vmem (the dispatcher routes large enough
    // head_dim through HVX and the kernels do strict-alignment loads).
    float coeffs[1024] __attribute__((aligned(128)));

    memcpy(coeffs, in_vec, sizeof(float) * head_dim);
    sp_hex_vht2_f32(coeffs, head_dim);

    int written = 0;
    int rc = sp_hex_band_quantize_scalar(coeffs, head_dim, out_packed,
                                          out_capacity, &written);
    if (rc != 0) {
        FARF(ERROR, "[sp_hex] compress: band_quantize rc=%d", rc);
        return rc;
    }
    *packed_used = written;
    return 0;
}

// decompress_f32 ? full decode pipeline in one FastRPC dispatch.
// Mirrors the second half of sp_hex_round_trip_f32 (band_dequantize,
// then VHT2 self-inverse). Used by the host's sp_hexagon_cache_read_*
// wrappers. max_bands < 0 (or >= n_bands) means "all bands"; 0 <=
// max_bands < n_bands triggers the partial-fidelity reconstruction
// path used by the phase 3 attention short-circuit.
int sp_hex_decompress_f32(remote_handle64 h,
                           const unsigned char *packed_in, int packed_len,
                           int head_dim, int max_bands,
                           float *out_vec, int out_len) {
    (void)h;
    if (out_len != head_dim) {
        FARF(ERROR, "[sp_hex] decompress: out_len=%d != head_dim=%d",
             out_len, head_dim);
        return -1;
    }
    if (head_dim < 8 || head_dim > 1024 ||
        (head_dim & (head_dim - 1)) != 0) {
        FARF(ERROR, "[sp_hex] decompress: head_dim=%d must be pow2 in [8,1024]",
             head_dim);
        return -1;
    }

    float coeffs[1024] __attribute__((aligned(128)));

    int rc = sp_hex_band_dequantize_scalar(packed_in, packed_len, head_dim,
                                            max_bands, coeffs);
    if (rc != 0) {
        FARF(ERROR, "[sp_hex] decompress: band_dequantize rc=%d", rc);
        return rc;
    }
    // VHT2 is self-inverse ? applying it again returns to the original
    // basis. The HVX-qf32 kernel is bit-equivalent to scalar within
    // fp32 epsilon, so this matches the round-trip path's reconstruction.
    sp_hex_vht2_f32(coeffs, head_dim);
    memcpy(out_vec, coeffs, sizeof(float) * head_dim);
    return 0;
}

// compress_f32_batch ? loop the compress_f32 kernel internally over
// n_vectors so the host pays one FastRPC dispatch per chunk instead
// of one per position. Used by the post-decode hook on prefill, where
// the bridge sees a contiguous run of K/V positions in a single batch.
//
// Layout matches the IDL contract:
//   in_vecs    = n_vectors * head_dim contiguous fp32
//   out_packed = n_vectors * bc.total_bytes contiguous octets
//   packed_used = n_vectors * bc.total_bytes on success
//
// Validation rules: in_len == n_vectors * head_dim, out_capacity >=
// n_vectors * bc.total_bytes. n_vectors must be > 0; an empty batch is
// rejected so the host's chunking logic can't silently drop work.
int sp_hex_compress_f32_batch(remote_handle64 h,
                               const float *in_vecs, int in_len,
                               int head_dim, int n_vectors,
                               unsigned char *out_packed, int out_capacity,
                               int *packed_used) {
    (void)h;
    if (!packed_used) return -1;
    *packed_used = 0;
    if (n_vectors <= 0) {
        FARF(ERROR, "[sp_hex] compress_batch: n_vectors=%d <= 0", n_vectors);
        return -1;
    }
    if (head_dim < 8 || head_dim > 1024 ||
        (head_dim & (head_dim - 1)) != 0) {
        FARF(ERROR, "[sp_hex] compress_batch: head_dim=%d must be pow2 in [8,1024]",
             head_dim);
        return -1;
    }
    if (in_len != n_vectors * head_dim) {
        FARF(ERROR, "[sp_hex] compress_batch: in_len=%d != n_vectors=%d * hd=%d",
             in_len, n_vectors, head_dim);
        return -1;
    }

    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, head_dim, 4, default_bits);
    int per_vec_bytes = bc.total_bytes;
    int total_needed = n_vectors * per_vec_bytes;
    if (total_needed > out_capacity) {
        FARF(ERROR, "[sp_hex] compress_batch: needed=%d > capacity=%d",
             total_needed, out_capacity);
        return -1;
    }

    // Stack scratch reused across iterations. 1024-float aligned coeffs
    // covers head_dim ? 1024 (any production model). The packed stage is
    // written directly into the per-vector offset of out_packed so we
    // don't need a separate per-iter packed scratch.
    float coeffs[1024] __attribute__((aligned(128)));

    for (int i = 0; i < n_vectors; ++i) {
        const float *in_vec = in_vecs + (size_t)i * (size_t)head_dim;
        unsigned char *out_slot = out_packed + (size_t)i * (size_t)per_vec_bytes;

        memcpy(coeffs, in_vec, sizeof(float) * head_dim);
        sp_hex_vht2_f32(coeffs, head_dim);

        int written = 0;
        int rc = sp_hex_band_quantize_scalar(coeffs, head_dim, out_slot,
                                              per_vec_bytes, &written);
        if (rc != 0) {
            FARF(ERROR, "[sp_hex] compress_batch: vec[%d] band_quantize rc=%d",
                 i, rc);
            return rc;
        }
        if (written != per_vec_bytes) {
            FARF(ERROR, "[sp_hex] compress_batch: vec[%d] short write %d != %d",
                 i, written, per_vec_bytes);
            return -1;
        }
    }
    *packed_used = total_needed;
    return 0;
}

// decompress_f32_batch ? loop the decompress_f32 kernel internally
// over n_vectors. max_bands applies to every vector in the batch (the
// host caller is responsible for grouping equal-fidelity reads into
// one call; mixed-fidelity batches would need a per-vector max_bands
// array, which today's host paths don't need).
int sp_hex_decompress_f32_batch(remote_handle64 h,
                                 const unsigned char *packed_in, int packed_len,
                                 int head_dim, int n_vectors, int max_bands,
                                 float *out_vecs, int out_len) {
    (void)h;
    if (n_vectors <= 0) {
        FARF(ERROR, "[sp_hex] decompress_batch: n_vectors=%d <= 0", n_vectors);
        return -1;
    }
    if (head_dim < 8 || head_dim > 1024 ||
        (head_dim & (head_dim - 1)) != 0) {
        FARF(ERROR, "[sp_hex] decompress_batch: head_dim=%d must be pow2 in [8,1024]",
             head_dim);
        return -1;
    }
    if (out_len != n_vectors * head_dim) {
        FARF(ERROR, "[sp_hex] decompress_batch: out_len=%d != n_vectors=%d * hd=%d",
             out_len, n_vectors, head_dim);
        return -1;
    }

    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, head_dim, 4, default_bits);
    int per_vec_bytes = bc.total_bytes;
    if (packed_len != n_vectors * per_vec_bytes) {
        FARF(ERROR, "[sp_hex] decompress_batch: packed_len=%d != n=%d * pv=%d",
             packed_len, n_vectors, per_vec_bytes);
        return -1;
    }

    float coeffs[1024] __attribute__((aligned(128)));

    for (int i = 0; i < n_vectors; ++i) {
        const unsigned char *in_slot = packed_in + (size_t)i * (size_t)per_vec_bytes;
        float *out_vec = out_vecs + (size_t)i * (size_t)head_dim;

        int rc = sp_hex_band_dequantize_scalar(in_slot, per_vec_bytes, head_dim,
                                                max_bands, coeffs);
        if (rc != 0) {
            FARF(ERROR, "[sp_hex] decompress_batch: vec[%d] dequant rc=%d", i, rc);
            return rc;
        }
        sp_hex_vht2_f32(coeffs, head_dim);
        memcpy(out_vec, coeffs, sizeof(float) * head_dim);
    }
    return 0;
}


// ============================================================================
// V-specific compress/decompress (1 band, 3 bits — different from K's 4-band
// 5/5/4/3 config). The kernels are identical to the K path but use V's bc.
// ============================================================================

int sp_hex_compress_f32_v(remote_handle64 h,
                           const float *in_vec, int in_len,
                           int head_dim,
                           unsigned char *out_packed, int out_capacity,
                           int *packed_used) {
    (void)h;
    if (!packed_used) return -1;
    *packed_used = 0;
    if (in_len != head_dim) return -1;
    if (head_dim < 8 || head_dim > 1024 || (head_dim & (head_dim - 1)) != 0) return -1;

    sp_band_config_t bc;
    int v_bits[1] = {3};
    sp_band_config_init(&bc, head_dim, 1, v_bits);
    if (bc.total_bytes > out_capacity) {
        FARF(ERROR, "[sp_hex] compress_v: packed=%d > capacity=%d",
             bc.total_bytes, out_capacity);
        return -1;
    }

    float coeffs[1024] __attribute__((aligned(128)));
    memcpy(coeffs, in_vec, sizeof(float) * head_dim);
    sp_hex_vht2_f32(coeffs, head_dim);

    // Use math core sp_band_quantize directly (takes bc), bypassing
    // sp_hex_band_quantize_scalar which has K's config hard-coded via
    // sp_hex_default_band_config. V's 1-band 3-bit config doesn't satisfy
    // HVX preconditions (head_dim >= 128 && % 128 == 0) anyway, so
    // scalar-only is fine here.
    sp_band_quantize(coeffs, out_packed, &bc);
    *packed_used = bc.total_bytes;
    return 0;
}

int sp_hex_decompress_f32_v(remote_handle64 h,
                             const unsigned char *packed_in, int packed_len,
                             int head_dim, int max_bands,
                             float *out_vec, int out_len) {
    (void)h;
    if (out_len != head_dim) return -1;
    if (head_dim < 8 || head_dim > 1024 || (head_dim & (head_dim - 1)) != 0) return -1;

    sp_band_config_t bc;
    int v_bits[1] = {3};
    sp_band_config_init(&bc, head_dim, 1, v_bits);
    if (packed_len < bc.total_bytes) {
        FARF(ERROR, "[sp_hex] decompress_v: packed_len=%d < expected=%d",
             packed_len, bc.total_bytes);
        return -1;
    }
    float coeffs[1024] __attribute__((aligned(128)));
    if (max_bands < 0 || max_bands >= bc.n_bands) {
        sp_band_dequantize(packed_in, coeffs, &bc);
    } else {
        sp_band_dequantize_partial(packed_in, coeffs, &bc, max_bands);
    }
    sp_hex_vht2_f32(coeffs, head_dim);
    memcpy(out_vec, coeffs, sizeof(float) * head_dim);
    return 0;
}

int sp_hex_compress_f32_v_batch(remote_handle64 h,
                                 const float *in_vecs, int in_len,
                                 int head_dim, int n_vectors,
                                 unsigned char *out_packed, int out_capacity,
                                 int *packed_used) {
    (void)h;
    if (!packed_used) return -1;
    *packed_used = 0;
    if (n_vectors <= 0) return -1;
    if (head_dim < 8 || head_dim > 1024 || (head_dim & (head_dim - 1)) != 0) return -1;
    if (in_len != n_vectors * head_dim) return -1;

    sp_band_config_t bc;
    int v_bits[1] = {3};
    sp_band_config_init(&bc, head_dim, 1, v_bits);
    int per_vec_bytes = bc.total_bytes;
    int total_needed = n_vectors * per_vec_bytes;
    if (total_needed > out_capacity) {
        FARF(ERROR, "[sp_hex] compress_v_batch: needed=%d > cap=%d",
             total_needed, out_capacity);
        return -1;
    }

    float coeffs[1024] __attribute__((aligned(128)));
    for (int i = 0; i < n_vectors; ++i) {
        const float *in_vec = in_vecs + (size_t)i * (size_t)head_dim;
        unsigned char *out_slot = out_packed + (size_t)i * (size_t)per_vec_bytes;
        memcpy(coeffs, in_vec, sizeof(float) * head_dim);
        sp_hex_vht2_f32(coeffs, head_dim);

        sp_band_quantize(coeffs, out_slot, &bc);
    }
    *packed_used = total_needed;
    return 0;
}

int sp_hex_decompress_f32_v_batch(remote_handle64 h,
                                   const unsigned char *packed_in, int packed_len,
                                   int head_dim, int n_vectors, int max_bands,
                                   float *out_vecs, int out_len) {
    (void)h;
    if (n_vectors <= 0) return -1;
    if (head_dim < 8 || head_dim > 1024 || (head_dim & (head_dim - 1)) != 0) return -1;
    if (out_len != n_vectors * head_dim) return -1;

    sp_band_config_t bc;
    int v_bits[1] = {3};
    sp_band_config_init(&bc, head_dim, 1, v_bits);
    int per_vec_bytes = bc.total_bytes;
    if (packed_len != n_vectors * per_vec_bytes) {
        FARF(ERROR, "[sp_hex] decompress_v_batch: packed_len=%d != n=%d * pv=%d",
             packed_len, n_vectors, per_vec_bytes);
        return -1;
    }

    float coeffs[1024] __attribute__((aligned(128)));
    for (int i = 0; i < n_vectors; ++i) {
        const unsigned char *in_slot = packed_in + (size_t)i * (size_t)per_vec_bytes;
        float *out_vec = out_vecs + (size_t)i * (size_t)head_dim;

        if (max_bands < 0 || max_bands >= bc.n_bands) {
            sp_band_dequantize(in_slot, coeffs, &bc);
        } else {
            sp_band_dequantize_partial(in_slot, coeffs, &bc, max_bands);
        }
        sp_hex_vht2_f32(coeffs, head_dim);
        memcpy(out_vec, coeffs, sizeof(float) * head_dim);
    }
    return 0;
}

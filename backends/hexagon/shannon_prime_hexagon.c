// Shannon-Prime VHT2 - Hexagon DSP backend (host-side).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Two configurations:
//
//   1. SP_HEXAGON_FASTRPC defined  →  real FastRPC implementation. Calls
//      the scaffold's sp_hex_* IDL methods (qaic-generated, see
//      backends/hexagon/scaffold/inc/sp_hex.idl). Requires the Hexagon
//      SDK headers and rpcmem/dsprpc libraries to be on the include /
//      link paths — typically only true when building the scaffold's
//      ARM-side target on Android.
//
//   2. SP_HEXAGON_FASTRPC undefined  →  stub fallback. Every entry point
//      returns -1 / NULL with a one-shot diagnostic so the bridge falls
//      back cleanly to Adreno/CPU. This is what x86 / desktop /
//      non-Snapdragon builds get.
//
// The math core (core/shannon_prime.c) is what runs on the DSP; this
// file is just the FastRPC shim that ferries data across the IPC boundary.

#include "shannon_prime_hexagon.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef SP_HEXAGON_FASTRPC

// ============================================================================
// Real FastRPC implementation
// ============================================================================

#include "rpcmem.h"
#include "remote.h"
#include "AEEStdErr.h"
#include "sp_hex.h"            // qaic-generated stub header from sp_hex.idl
#include "shannon_prime.h"     // sp_f16_to_f32 / sp_f32_to_f16 / sp_band_*

#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif

// FastRPC needs an explicit URI for each domain. The qaic-generated
// `sp_hex_URI` macro encodes the interface name; CDSP_DOMAIN appends the
// session selector. Both come from the SDK's remote.idl-derived headers.
#ifndef CDSP_DOMAIN
#define CDSP_DOMAIN "&_dom=cdsp"
#endif

struct sp_hexagon_ctx_s {
    remote_handle64 fastrpc_handle;     // FastRPC session
    sp_config_t     cfg_snapshot;       // Captured at init for reference
    size_t          bytes_in_use;       // Tracked allocations
    long long       last_call_cycles;   // 0 until profiling lands
    int             unsigned_pd_active; // 1 if we enabled it on this domain
};

// One-shot init of rpcmem. rpcmem_init / _deinit are reference-counted so
// repeated init pairs are safe; a static guard keeps us from doubling up.
static int g_rpcmem_inited = 0;

static int sp_hexagon_enable_unsigned_pd(int domain) {
    if (!remote_session_control) return AEE_EUNSUPPORTED;
    struct remote_rpc_control_unsigned_module data;
    data.domain = domain;
    data.enable = 1;
    return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                  (void *)&data, sizeof(data));
}

int sp_hexagon_caps_probe(sp_hexagon_caps_t *caps) {
    if (!caps) return -1;
    memset(caps, 0, sizeof(*caps));
    // Best-effort caps. A future revision could open a probe FastRPC
    // session and read DSP-side hardware regs (HEXAGON_REG_VERSION etc.)
    // For now we hardcode V69+HVX assumptions matching our v69 build.
    caps->has_dsp         = 1;
    caps->dsp_version     = 69;
    caps->has_hvx         = 1;
    caps->hvx_width_bits  = 1024;
    caps->has_hmx         = 0;          // V69 has no HMX (V73+ feature)
    caps->max_threads     = 4;          // V69 typical
    caps->shared_mem_bytes = 16 * 1024 * 1024;  // 16 MB rpcmem budget
    return 0;
}

void sp_hexagon_caps_print(const sp_hexagon_caps_t *caps) {
    if (!caps) return;
    fprintf(stderr, "Hexagon DSP Capabilities:\n");
    fprintf(stderr, "  DSP accessible: %s\n", caps->has_dsp ? "yes" : "no");
    if (caps->has_dsp) {
        fprintf(stderr, "  DSP version:    V%d\n", caps->dsp_version);
        fprintf(stderr, "  HVX:            %s (%d-bit)\n",
                caps->has_hvx ? "yes" : "no", caps->hvx_width_bits);
        fprintf(stderr, "  HMX:            %s\n", caps->has_hmx ? "yes" : "no");
        fprintf(stderr, "  Threads:        %d\n", caps->max_threads);
        fprintf(stderr, "  Shared memory:  %lld bytes\n",
                (long long)caps->shared_mem_bytes);
    }
}

sp_hexagon_ctx_t *sp_hexagon_init(const sp_config_t *cfg) {
    if (!cfg) return NULL;

    if (!g_rpcmem_inited) {
        rpcmem_init();
        g_rpcmem_inited = 1;
    }

    sp_hexagon_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->cfg_snapshot       = *cfg;
    ctx->fastrpc_handle     = (remote_handle64)-1;
    ctx->unsigned_pd_active = 0;

    // Enable unsigned PD on cDSP — required for unsigned developer builds.
    int rc = sp_hexagon_enable_unsigned_pd(CDSP_DOMAIN_ID);
    if (rc != AEE_SUCCESS) {
        fprintf(stderr, "[Shannon-Prime] hexagon: enable_unsigned_pd "
                        "failed 0x%x; trying signed path\n", rc);
    } else {
        ctx->unsigned_pd_active = 1;
    }

    // Open the FastRPC session against libsp_hex_skel.so on the cDSP.
    char uri_buf[128];
    snprintf(uri_buf, sizeof(uri_buf), "%s%s", sp_hex_URI, CDSP_DOMAIN);
    rc = sp_hex_open(uri_buf, &ctx->fastrpc_handle);
    if (rc != AEE_SUCCESS) {
        fprintf(stderr, "[Shannon-Prime] hexagon: sp_hex_open failed 0x%x; "
                        "falling back to Adreno/CPU\n", rc);
        free(ctx);
        return NULL;
    }
    return ctx;
}

void sp_hexagon_free(sp_hexagon_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->fastrpc_handle != (remote_handle64)-1) {
        sp_hex_close(ctx->fastrpc_handle);
    }
    free(ctx);
}

void *sp_hexagon_alloc(sp_hexagon_ctx_t *ctx, size_t n_bytes) {
    if (!ctx) return NULL;
    // Page-align up to 4 KB.
    size_t aligned = (n_bytes + 4095) & ~(size_t)4095;
    void *p = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS,
                           (int)aligned);
    if (p) ctx->bytes_in_use += aligned;
    return p;
}

void sp_hexagon_free_shared(sp_hexagon_ctx_t *ctx, void *ptr) {
    if (!ctx || !ptr) return;
    rpcmem_free(ptr);
    // bytes_in_use bookkeeping is approximate — rpcmem doesn't expose
    // per-pointer size lookup. Rely on the next caps refresh for truth.
}

// fp16 ↔ fp32 bridges using the math core helpers (sp_f16_to_f32,
// sp_f32_to_f16). These are header-inline in shannon_prime.h so no extra
// link footprint.
static void sp_hex_widen_f16_to_f32(const uint16_t *in, float *out, int n) {
    for (int i = 0; i < n; ++i) out[i] = sp_f16_to_f32(in[i]);
}
static void sp_hex_narrow_f32_to_f16(const float *in, uint16_t *out, int n) {
    for (int i = 0; i < n; ++i) out[i] = sp_f32_to_f16(in[i]);
}

// Single-vector round-trip. fp16 in/out, fp32 across the FastRPC boundary.
// Future optimization: widen the IDL to take fp16 directly so we skip the
// host-side widening (saves 2 × head_dim × 4 bytes per call of memory
// traffic plus the conversion loops).
static int sp_hex_round_trip_one(sp_hexagon_ctx_t *ctx,
                                  const uint16_t *in_fp16,
                                  uint16_t *out_fp16) {
    if (!ctx || ctx->fastrpc_handle == (remote_handle64)-1) return -1;
    int hd = ctx->cfg_snapshot.head_dim;
    if (hd <= 0 || hd > 1024) return -1;

    // Stack scratch — 4 KB on a 64-byte stack vs heap thrash.
    float in_f32[1024];
    float out_f32[1024];
    sp_hex_widen_f16_to_f32(in_fp16, in_f32, hd);
    int rc = sp_hex_round_trip_f32(ctx->fastrpc_handle, in_f32, hd, hd,
                                    out_f32, hd);
    if (rc != AEE_SUCCESS) return rc;
    sp_hex_narrow_f32_to_f16(out_f32, out_fp16, hd);
    return 0;
}

int sp_hexagon_round_trip_k(sp_hexagon_ctx_t *ctx,
                             const uint16_t *in_fp16, uint16_t *out_fp16) {
    return sp_hex_round_trip_one(ctx, in_fp16, out_fp16);
}

int sp_hexagon_round_trip_v(sp_hexagon_ctx_t *ctx,
                             const uint16_t *in_fp16, uint16_t *out_fp16) {
    // K and V share the same DSP code path for now. Differentiation
    // (e.g. V using a 1-band 3-bit config) lands when the IDL grows a
    // band-config parameter.
    return sp_hex_round_trip_one(ctx, in_fp16, out_fp16);
}

int sp_hexagon_round_trip_k_batch(sp_hexagon_ctx_t *ctx,
                                   const uint16_t *in_fp16, uint16_t *out_fp16,
                                   int n_vectors) {
    if (!ctx || n_vectors <= 0) return -1;
    int hd = ctx->cfg_snapshot.head_dim;
    // Naive loop. A real batch IDL entry would amortize FastRPC overhead
    // across all n_vectors; we'll add that once the per-call cost shows
    // up in profiles. For now, correctness before perf.
    for (int v = 0; v < n_vectors; ++v) {
        int rc = sp_hex_round_trip_one(ctx, in_fp16 + v * hd,
                                        out_fp16 + v * hd);
        if (rc != AEE_SUCCESS) return rc;
    }
    return 0;
}

int sp_hexagon_band_dequantize_partial(sp_hexagon_ctx_t *ctx,
                                        const uint8_t *in_packed,
                                        uint16_t *out_fp16,
                                        int max_bands) {
    if (!ctx || ctx->fastrpc_handle == (remote_handle64)-1) return -1;
    int hd = ctx->cfg_snapshot.head_dim;
    if (hd <= 0 || hd > 1024) return -1;

    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, hd, 4, default_bits);

    float coeffs[1024];
    int rc = sp_hex_band_dequantize(ctx->fastrpc_handle,
                                     in_packed, bc.total_bytes,
                                     hd, max_bands,
                                     coeffs, hd);
    if (rc != AEE_SUCCESS) return rc;

    // The math core's band_dequantize emits VHT2 coefficients. Inverse
    // VHT2 to land back in the original basis, then narrow to fp16.
    sp_vht2_forward_f32(coeffs, hd);   // self-inverse
    sp_hex_narrow_f32_to_f16(coeffs, out_fp16, hd);
    return 0;
}

size_t sp_hexagon_memory_in_use(const sp_hexagon_ctx_t *ctx) {
    return ctx ? ctx->bytes_in_use : 0;
}

long long sp_hexagon_last_call_cycles(const sp_hexagon_ctx_t *ctx) {
    return ctx ? ctx->last_call_cycles : 0;
}

#else  // !SP_HEXAGON_FASTRPC

// ============================================================================
// Stub fallback (x86 / desktop / no FastRPC headers available).
// The bridge sees -1 / NULL and falls back to Adreno/CPU.
// ============================================================================

int sp_hexagon_caps_probe(sp_hexagon_caps_t *caps) {
    if (!caps) return -1;
    memset(caps, 0, sizeof(*caps));
    return -1;  // signal "DSP unavailable"
}

void sp_hexagon_caps_print(const sp_hexagon_caps_t *caps) {
    if (!caps) return;
    fprintf(stderr, "Hexagon DSP Capabilities:\n");
    fprintf(stderr, "  DSP accessible: %s\n", caps->has_dsp ? "yes" : "no");
    if (!caps->has_dsp) {
        fprintf(stderr, "  (DSP unavailable - SDK not built or device not "
                        "Snapdragon)\n");
    }
}

sp_hexagon_ctx_t *sp_hexagon_init(const sp_config_t *cfg) {
    (void)cfg;
    static int warned = 0;
    if (!warned) {
        fprintf(stderr,
            "[Shannon-Prime] Hexagon backend not built (SP_HEXAGON_FASTRPC "
            "undefined). Build the scaffold ARM target for Android to enable. "
            "Falling back to Adreno or CPU backend.\n");
        warned = 1;
    }
    return NULL;
}

void sp_hexagon_free(sp_hexagon_ctx_t *ctx) { (void)ctx; }

void *sp_hexagon_alloc(sp_hexagon_ctx_t *ctx, size_t n_bytes) {
    (void)ctx; (void)n_bytes; return NULL;
}
void sp_hexagon_free_shared(sp_hexagon_ctx_t *ctx, void *ptr) {
    (void)ctx; (void)ptr;
}

int sp_hexagon_round_trip_k(sp_hexagon_ctx_t *ctx,
                             const uint16_t *in_fp16, uint16_t *out_fp16) {
    (void)ctx; (void)in_fp16; (void)out_fp16; return -1;
}
int sp_hexagon_round_trip_v(sp_hexagon_ctx_t *ctx,
                             const uint16_t *in_fp16, uint16_t *out_fp16) {
    (void)ctx; (void)in_fp16; (void)out_fp16; return -1;
}
int sp_hexagon_round_trip_k_batch(sp_hexagon_ctx_t *ctx,
                                   const uint16_t *in_fp16, uint16_t *out_fp16,
                                   int n_vectors) {
    (void)ctx; (void)in_fp16; (void)out_fp16; (void)n_vectors; return -1;
}
int sp_hexagon_band_dequantize_partial(sp_hexagon_ctx_t *ctx,
                                        const uint8_t *in_packed,
                                        uint16_t *out_fp16, int max_bands) {
    (void)ctx; (void)in_packed; (void)out_fp16; (void)max_bands; return -1;
}

size_t sp_hexagon_memory_in_use(const sp_hexagon_ctx_t *ctx) {
    (void)ctx; return 0;
}
long long sp_hexagon_last_call_cycles(const sp_hexagon_ctx_t *ctx) {
    (void)ctx; return 0;
}

#endif  // SP_HEXAGON_FASTRPC

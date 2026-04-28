// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available - contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#ifndef SHANNON_PRIME_HEXAGON_H
#define SHANNON_PRIME_HEXAGON_H

#include "../../core/shannon_prime.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Hexagon Backend - Qualcomm DSP for KV Cache Compression
// ============================================================================
//
// Targets the Hexagon DSP block on Snapdragon SoCs. Co-processor: runs in
// parallel with the ARM cores and the Adreno GPU at low power. SP's
// banded quantize/dequantize maps cleanly onto HVX vector intrinsics; the
// VHT2 butterfly maps cleanly onto HMX matrix intrinsics on V73+ (S8G3+).
//
// Hardware target ladder:
//
//   Hexagon V69 (Snapdragon 8 Gen 1)     HVX 1024-bit only.        Primary.
//   Hexagon V73 (Snapdragon 8 Gen 2)     HVX + HMX 256x256 int8.   Future.
//   Hexagon V75 (Snapdragon 8 Gen 3)     HVX + HMX + tensor cores. Future.
//
// Primary validation device: Samsung Galaxy S22 Ultra (SM8450, V69).
// SDK requirement: Qualcomm Hexagon SDK 5.x (toolv87 / DSP_ARCH=v69).
//
// Architecture:
//
//   ARM (host)                                    Hexagon DSP (server)
//   -----------                                   --------------------
//   sp_hexagon_init() ----[FastRPC]----> sp_hexagon_dsp_init()
//   sp_hexagon_round_trip_band(K) ---->  HVX kernel: gather + quantize +
//                                         dequantize + scatter
//                                   <----  reconstructed K via shared mem
//
// Communication layer is FastRPC (the Qualcomm-supplied IPC mechanism for
// CPU<->DSP). Buffers are allocated via rpcmem_alloc with the
// RPCMEM_HEAP_ID_CONTIG flag to land in a shared physical region that
// both sides can access without copies.
//
// Compute mapping:
//
//   Compute Unit            VHT2 Role                  Hexagon API
//   --------------------------------------------------------------------
//   ARM Cortex CPU          Orchestration, FastRPC     standard libc
//   Adreno 730 GPU          Prefill batch (Vulkan)     parallel path
//   Hexagon V69 DSP HVX     VHT2 butterfly + bands     this header
//   Hexagon V69 DSP scalar  glue logic, control flow   this header
//
// The DSP kernels are designed to fit in HVX_VECTORS_AT_ONCE (typically 4-8)
// 1024-bit registers per band. A single layer's worth of (head x position)
// vectors streams through the HVX pipeline at one vector per cycle in
// the inner loop.

// ============================================================================
// Capability / feature detection
// ============================================================================

typedef struct {
    int has_dsp;                 // Hexagon DSP accessible via FastRPC
    int dsp_version;             // V69, V73, V75 (raw int)
    int has_hvx;                 // HVX vector eXtension
    int hvx_width_bits;          // 1024 on V69+
    int has_hmx;                 // HMX matrix eXtension (V73+)
    int max_threads;             // DSP hardware threads (typically 4-6)
    long long shared_mem_bytes;  // Shared CPU<->DSP physical buffer budget
} sp_hexagon_caps_t;

// Probe the device capabilities. Returns 0 on success and fills caps;
// returns -1 if the DSP is unreachable (driver missing, not a Snapdragon
// device, or FastRPC service unavailable).
int sp_hexagon_caps_probe(sp_hexagon_caps_t *caps);
void sp_hexagon_caps_print(const sp_hexagon_caps_t *caps);

// ============================================================================
// Lifecycle
// ============================================================================

// Opaque handle to the host-side Hexagon SP context. Holds the FastRPC
// session, the shared-memory pool, and any per-shape kernel cache.
typedef struct sp_hexagon_ctx_s sp_hexagon_ctx_t;

// Initialise a Hexagon SP context. Loads the DSP-side stub, opens the
// FastRPC session, allocates the shared-memory pool sized for one
// layer's worth of K/V vectors plus headroom for HVX scratch.
//
// Returns NULL if the DSP is unavailable (caller should fall back to
// the Adreno or CPU backend). On success, caller must release the
// context with sp_hexagon_free.
sp_hexagon_ctx_t *sp_hexagon_init(const sp_config_t *cfg);
void sp_hexagon_free(sp_hexagon_ctx_t *ctx);

// ============================================================================
// Round-trip operations
// ============================================================================
//
// The same operation triple as the Adreno backend (gather + transform +
// scatter) but executes on the DSP. Buffers are RPC-shared, so the
// caller gets pointers via sp_hexagon_alloc / sp_hexagon_free_shared.
// Pass these pointers, not malloc'd ones - non-shared memory will incur
// an extra copy across FastRPC.

// Allocate / free shared memory. n_bytes must be page-aligned (4 KB);
// the implementation rounds up.
void *sp_hexagon_alloc(sp_hexagon_ctx_t *ctx, size_t n_bytes);
void  sp_hexagon_free_shared(sp_hexagon_ctx_t *ctx, void *ptr);

// Round-trip a single (head, position) vector through the DSP:
// fp16_in -> fp16 promote -> VHT2 -> Mobius -> band-quantize -> bytes
// bytes -> band-dequantize -> Mobius unreorder -> VHT2 inverse -> fp16_out
//
// in / out must be sp_hexagon_alloc'd. Returns 0 on success, non-zero
// on FastRPC failure.
int sp_hexagon_round_trip_k(sp_hexagon_ctx_t *ctx,
                             const uint16_t *in_fp16,   // head_dim fp16 values
                             uint16_t *out_fp16);        // head_dim fp16 values

int sp_hexagon_round_trip_v(sp_hexagon_ctx_t *ctx,
                             const uint16_t *in_fp16,
                             uint16_t *out_fp16);

// Batch round-trip: process n_vectors at once for better HVX
// utilisation. Used during prefill where we have many (head x position)
// vectors to handle in parallel. Buffers must be shared (allocated via
// sp_hexagon_alloc).
int sp_hexagon_round_trip_k_batch(sp_hexagon_ctx_t *ctx,
                                   const uint16_t *in_fp16,
                                   uint16_t *out_fp16,
                                   int n_vectors);

// Progressive partial dequantize on the DSP - same semantics as
// sp_band_dequantize_partial (math core). max_bands is clamped to
// [0, n_bands]. Used by the phase 3 attention short-circuit:
// reconstruct band 0 only first, attention probes for confidence,
// promote to bands 0+1 if needed, etc.
//
// in: packed band bytes (host-side or shared)
// out: head_dim fp16 reconstruction (shared memory required)
int sp_hexagon_band_dequantize_partial(sp_hexagon_ctx_t *ctx,
                                        const uint8_t *in_packed,
                                        uint16_t *out_fp16,
                                        int max_bands);

// ============================================================================
// Diagnostics
// ============================================================================

// Total bytes of shared memory currently allocated by this context.
size_t sp_hexagon_memory_in_use(const sp_hexagon_ctx_t *ctx);

// Approximate cycles spent in the most recent DSP call (read from the
// HEXAGON_REG_TIMER counter on the DSP side). 0 if profiling not
// enabled (compile-time HEXAGON_PROFILE=0). Microsecond-class resolution.
long long sp_hexagon_last_call_cycles(const sp_hexagon_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif // SHANNON_PRIME_HEXAGON_H

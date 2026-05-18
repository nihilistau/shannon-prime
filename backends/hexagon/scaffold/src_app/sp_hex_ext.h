// Shannon-Prime VHT2 - Hexagon DSP FastRPC scaffold (ARM-side header).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Forked from the Hexagon SDK 5.5.6.0 S22U sample. Qualcomm copyright on the
// pattern; SP-specific code is AGPLv3.

#ifndef SP_HEX_EXT_H
#define SP_HEX_EXT_H

#include "AEEStdDef.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Top-level smoke-test driver. Allocates rpcmem, opens a FastRPC session
// to the cDSP, runs round_trip_f32 on a deterministic input, and reports
// the worst-case fp32 error. Returns 0 on full success.
int sp_hex_process(int domain, int head_dim, bool isUnsignedPD_Enabled);

// Second smoke-test path: drives the engine-side hexagon API
// (sp_hexagon_init / sp_hexagon_round_trip_k / sp_hexagon_free) instead
// of the qaic IDL directly. Validates that the engine's host-side
// FastRPC shim works end-to-end on the phone before we wire it into
// llama-cpp-sp's bridge.
int sp_hex_engine_smoke(int head_dim);

// Cycle bench: runs the VHT2 forward butterfly through both the scalar
// reference path and the HVX-vectorised path on the cDSP, prints
// per-call pcycles + speedup ratio across a sweep of head_dim values.
// Uses sp_hex_vht2_bench (FastRPC) under the hood with HAP_perf_get_pcycles
// timing.
int sp_hex_run_bench_sweep(void);

// Disk-tier proof: scalar host-side quantize fills an rpcmem-backed
// packed-bands buffer, sp_hexagon_band_dequantize_partial processes via
// FastRPC zero-copy, output compared to scalar dequantize reference.
// Demonstrates the rpcmem → DSP path that disk I/O (fread into rpcmem)
// would feed into.
int sp_hex_disk_tier_proof(int head_dim);

// Per-element validation harness for the compress_f32 / decompress_f32
// IDL pair. Drives a deterministic input through (a) the DSP-side
// compress + decompress round-trip and (b) the host-side scalar
// reference (sp_band_quantize → sp_band_dequantize on already-VHT2'd
// coeffs), then compares per-element worst-abs.
//
// This is the validation lesson from the 2026-04-29 V69 IEEE-HVX
// debugging episode: round-trip RMS alone is not sufficient — two
// paths can produce the same RMS while differing wildly in their
// intermediate values. A per-element comparator is required before
// any new IDL method goes on the bridge hot path.
int sp_hex_compress_decompress_validate(int head_dim);

// Path A.2 prototype — CPU benchmark of fused decompress-matmul vs vanilla
// matmul. Validates the math + measures CPU-side perf so we know whether
// CPU is fast enough or we need a cDSP-fused kernel.
//   n_kv: number of K rows (e.g., 4096 for Dolphin n_ctx)
//   hd:   head_dim, power of 2 in [8, 1024]
//   n_q:  number of Q queries to dot against each K row
int sp_hex_kq_matmul_bench(int n_kv, int hd, int n_q);

// Phase 2.0 weight streaming probe — standalone validator for the
// "Zero-RAM-Footprint" thesis. Spawns an ARM I/O thread that pread()s
// from a backing weight file into a circular rpcmem ring buffer, calls
// sp_hex_weight_stream_session on the cDSP which polls the ring, fused-
// decompresses each tile and FMA's against a resident activation, and
// reports throughput + wait/compute jitter via qtimer pcycles.
//
// Args:
//   weight_file_path : path to a backing weight file (will be created
//                      with random SP-compressed bytes if missing,
//                      sized to file_bytes_target).
//   file_bytes_target: total bytes to stream end-to-end (e.g., 1 GiB).
//   tile_bytes       : per-tile bytes (default 16384 = 16 KB).
//   n_slots          : ring depth (default 4 = triple-buffer + drain).
//   head_dim         : VHT2 vector length (default 128).
//
// Success criterion: sustained throughput approaches UFS line speed
// (~1.2 GB/s on UFS 3.1) AND wait_pcycles > 0 consistently — that
// confirms compute is free and UFS is the bottleneck (the Phase 2.0
// thesis). If wait_pcycles == 0 (DSP never idle), compute is the
// bottleneck and we need a faster kernel before streaming makes sense.
int sp_hex_weight_stream_bench(const char *weight_file_path,
                                long long file_bytes_target,
                                int tile_bytes,
                                int n_slots,
                                int head_dim);

// Strike 5.5: FastRPC parity test for the new matmul_block_q8 IDL method.
// Builds a synthetic (w_blocks, x_row) input, dispatches to the cDSP via
// FastRPC, compares (acc_a, acc_b) bit-equal against the host's scalar
// reference (sp_hex_matmul_ok_block_q8_inner). Validates:
//   - ARM → cDSP IDL marshalling for sequence<octet> + sequence<long long>
//   - SMMU page handoff (the 64-B aligned w_blocks arrive intact)
//   - V69 vmpyieacc semantics match the scalar reference bit-for-bit
//
// blocks_per_row: how many 32-element blocks per matmul row (1, 4, 16, 64).
// Returns 0 on success (all configurations bit-equal), non-zero on first
// mismatch with details printed to stderr.
int sp_hex_matmul_block_q8_parity(int blocks_per_row);

// Strike 6: FastRPC parity test for the new mobius_scatter_f32 IDL method.
// Generates a deterministic fp32 input vector, dispatches Möbius reorder via
// HVX vscatter on the DSP, compares against the host scalar reference
// (apply inverse permutation directly). Bit-equal contract: this is pure
// data movement, no math involved.
//
// head_dim must be one of {64, 128, 256, 512} — the compile-time tables.
int sp_hex_mobius_scatter_parity(int head_dim);

// Strike 7: FastRPC byte-equal parity test for band_quantize.
// Generates a deterministic fp32 input, dispatches HVX band_quantize on
// the DSP, and compares the packed output BYTE-BY-BYTE against the host's
// sp_band_quantize. This is the stricter contract — not just "decompresses
// to the same fp32" but "the packed disk-tier byte format is identical".
//
// head_dim must be 64/128/256/512 (band_config_init covers these).
int sp_hex_band_quantize_parity(int head_dim);

// Strike 8a: FastRPC parity test for logit_argmax_u16.
// Synthesizes a UFIXED_16 logit row at production vocab sizes (e.g. 151936
// for Qwen3-4B), seeds known max positions, dispatches via FastRPC. Compares
// against the scalar host argmax. Validates: alignment prologue, HVX vmax
// scan, vror tree reduce, scalar index-find second pass.
//
// Tests both well-aligned (vocab_size multiple of 64) and misaligned
// (vocab_size with a remainder) configurations, plus edge cases (max at
// position 0, at the end, and in the middle).
int sp_hex_logit_argmax_parity(void);

// Strike 9: FastRPC parity test for compress_f32_full (the grand fusion).
// Validates that the single DSP-side dispatch (VHT2 + HVX Möbius scatter
// + HVX band_quantize, all chained through VTCM) produces byte-identical
// output to the host-side sequence:
//   sp_vht2_forward_f32 → sp_mobius_reorder_ex → sp_band_quantize.
//
// This is the strictest contract: every fp32 → byte transformation in
// the encode path must match across all 4 supported head_dims.
int sp_hex_compress_f32_full_parity(int head_dim);

// Strike 10b: FastRPC parity test for compress_f32_full_batch (batched grand
// fusion). Stages n_vectors deterministic fp32 inputs, dispatches via ONE
// FastRPC call, asserts byte-equality vs host scalar reference per-vector.
// Validates per-iteration VTCM reuse — i.e. that the DSP's batch loop is
// race-free across iterations (each Möbius scatter overwrites the previous
// before band_quantize touches it).
//
// head_dim must be one of {64, 128, 256, 512} (Möbius compile-time tables).
// n_vectors in [1, 256] (allocation budget; production chunks are typically 32).
int sp_hex_compress_f32_full_batch_parity(int head_dim, int n_vectors);

// Strike 11b: FastRPC parity test for sp_hex_hier_predict_f32 (the W-matrix
// predictor — Hierarchical Spinor entry point). Two checks per skeleton input:
//   1. KERNEL CORRECTNESS — DSP fp32 output must be bit-equal to the host
//      Q15 scalar reference (which runs the IDENTICAL Q15 math).
//   2. QUANT BUDGET — DSP output must be within ~5e-4 of the host pure-fp32
//      reference (sum-of-14-Q15-quant-errors bound).
//
// Sweeps three deterministic skeleton patterns: uniform / alternating / spike.
// Currently tests head_dim=154 (skeleton=14, predicted=140) — the only config
// in the W rodata bank. Returns 0 if all three pass both checks.
int sp_hex_hier_predict_parity(void);

// Strike 12: FastRPC parity test for sp_hex_residual_quantize_spinor.
// Drives 3 deterministic (actual, predicted) patterns through the DSP
// residual packer; asserts byte-equal 71-byte output + bit-equal fp32
// amax vs the host scalar reference. Returns 0 if all three patterns pass.
int sp_hex_residual_spinor_parity(void);

#ifdef __cplusplus
}
#endif

#endif // SP_HEX_EXT_H

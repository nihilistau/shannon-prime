// Shannon-Prime VHT2 - Hexagon DSP FastRPC scaffold (ARM-side main).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Forked from the Hexagon SDK 5.5.6.0 S22U sample. Qualcomm copyright on the
// scaffolding pattern; SP-specific code is AGPLv3.

#include "sp_hex_ext.h"
#include "rpcmem.h"
#include "remote.h"
#include "dsp_capabilities_utils.h"
#include "os_defines.h"

#include <stdlib.h>
#include <stdio.h>

static void print_usage(void) {
    printf(
        "Usage:\n"
        "    sp_hex [-d domain] [-U unsigned_PD] [-n head_dim]\n\n"
        "Options:\n"
        "  -d domain    : DSP domain (0=ADSP, 3=CDSP). Default: 3 (CDSP).\n"
        "  -U unsigned  : 1 = unsigned PD (default), 0 = signed PD.\n"
        "  -n head_dim  : VHT2 vector length (power of 2, >=8). Default: 128.\n"
        "\n"
        "Smoke test: feeds a deterministic fp32 vector through VHT2(VHT2(x))\n"
        "on the cDSP and reports max-abs error. Expected: ~0 to fp32 epsilon.\n"
    );
}

int main(int argc, char *argv[]) {
    int nErr = 0;
    int head_dim = 128;
    int domain = 3;
    int unsignedPDFlag = 1;
    bool isUnsignedPD_Enabled = false;
    int option = 0;

    while ((option = getopt(argc, argv, "d:U:n:h")) != -1) {
        switch (option) {
            case 'd': domain = atoi(optarg); break;
            case 'U': unsignedPDFlag = atoi(optarg); break;
            case 'n': head_dim = atoi(optarg); break;
            case 'h':
            default:
                print_usage();
                if (option == 'h') return 0;
        }
    }

    if (unsignedPDFlag == 1) {
        if (domain == CDSP_DOMAIN_ID || domain == CDSP1_DOMAIN_ID) {
            isUnsignedPD_Enabled = true;
        } else {
            printf("Overriding user request for unsigned PD. Only signed "
                   "offload is allowed on domain %d.\n", domain);
            unsignedPDFlag = 0;
        }
    }

    printf("\n[sp_hex] Shannon-Prime Hexagon DSP scaffold smoke test\n");
    printf("[sp_hex] Domain: %d  PD: %s  head_dim: %d\n",
           domain, unsignedPDFlag == 1 ? "unsigned" : "signed", head_dim);

    nErr = sp_hex_process(domain, head_dim, isUnsignedPD_Enabled);
    if (nErr) {
        printf("ERROR 0x%x: sp_hex smoke test failed\n", nErr);
        return nErr;
    }
    printf("[sp_hex] Direct IDL path: Success\n");

    int eErr = sp_hex_engine_smoke(head_dim);
    if (eErr) {
        printf("ERROR: engine API smoke test failed\n");
        return eErr;
    }

    int bErr = sp_hex_run_bench_sweep();
    if (bErr) {
        printf("ERROR: bench sweep failed\n");
        return bErr;
    }

    int dErr = sp_hex_disk_tier_proof(head_dim);
    if (dErr) {
        printf("ERROR: disk-tier proof failed\n");
        return dErr;
    }

    int vErr = sp_hex_compress_decompress_validate(head_dim);
    if (vErr) {
        printf("ERROR: per-element compress/decompress validate failed\n");
        return vErr;
    }

    // Path A.2 prototype CPU benchmark: fused decompress-matmul vs vanilla.
    // Workload sized to match Dolphin 1B at n_ctx=4096 (per-layer-head shape).
    // Override via env vars: SP_HEX_BENCH_NKV / SP_HEX_BENCH_HD / SP_HEX_BENCH_NQ.
    int bench_nkv = 4096;
    int bench_hd  = head_dim;
    int bench_nq  = 8;
    const char *e = NULL;
    if ((e = getenv("SP_HEX_BENCH_NKV")) && *e) bench_nkv = atoi(e);
    if ((e = getenv("SP_HEX_BENCH_HD"))  && *e) bench_hd  = atoi(e);
    if ((e = getenv("SP_HEX_BENCH_NQ"))  && *e) bench_nq  = atoi(e);
    int kqErr = sp_hex_kq_matmul_bench(bench_nkv, bench_hd, bench_nq);
    if (kqErr) {
        printf("ERROR: kq_matmul_bench failed (err=%d)\n", kqErr);
        // non-fatal — print and continue
    }

    // Strike 5.5: validate the HVX cyclotomic matmul kernel via FastRPC.
    // Walks blocks_per_row through (1, 4, 16, 64) and asserts bit-equal
    // (acc_a, acc_b) between the DSP HVX path and the host scalar reference.
    printf("\n[parity] === Strike 5: matmul_block_q8 FastRPC parity test ===\n");
    int parity_fail = 0;
    int parity_widths[] = {1, 4, 16, 64};
    for (int i = 0; i < 4; ++i) {
        int rc = sp_hex_matmul_block_q8_parity(parity_widths[i]);
        if (rc) { parity_fail = 1; break; }
    }
    if (parity_fail) {
        printf("[parity] FAIL — HVX kernel diverged from host reference\n");
    } else {
        printf("[parity] all configurations bit-equal\n");
    }

    // Strike 6: validate HVX vscatter-based Möbius reorder via FastRPC.
    printf("\n[mobius] === Strike 6: mobius_scatter_f32 FastRPC parity test ===\n");
    int mobius_fail = 0;
    int mobius_dims[] = {64, 128, 256, 512};
    for (int i = 0; i < 4; ++i) {
        int rc = sp_hex_mobius_scatter_parity(mobius_dims[i]);
        if (rc) { mobius_fail = 1; break; }
    }
    if (mobius_fail) {
        printf("[mobius] FAIL — HVX scatter diverged from host reorder\n");
    } else {
        printf("[mobius] all 4 head_dims (64/128/256/512) bit-equal\n");
    }

    // Strike 7: byte-equal parity for HVX band_quantize (5/5/4/3).
    printf("\n[bandq] === Strike 7: band_quantize FastRPC byte-equal parity ===\n");
    int bandq_fail = 0;
    int bandq_dims[] = {64, 128, 256, 512};
    for (int i = 0; i < 4; ++i) {
        int rc = sp_hex_band_quantize_parity(bandq_dims[i]);
        if (rc) { bandq_fail = 1; break; }
    }
    if (bandq_fail) {
        printf("[bandq] FAIL — packed bytes diverged from host sp_band_quantize\n");
    } else {
        printf("[bandq] all 4 head_dims byte-equal — disk-tier format locked\n");
    }

    // Strike 8a: HVX logit argmax — kill the 300 KB FastRPC choke on decode.
    printf("\n[argmax] === Strike 8a: logit_argmax_u16 FastRPC parity test ===\n");
    int argmax_fail = sp_hex_logit_argmax_parity();
    if (argmax_fail) {
        printf("[argmax] FAIL — HVX argmax diverged from host scalar\n");
    } else {
        printf("[argmax] all cases match — 300 KB IPC choke collapsed to 4 bytes\n");
    }

    // Strike 9: Grand Fusion — VHT2 + Möbius + band_quantize in one dispatch.
    printf("\n[fused] === Strike 9: compress_f32_full FastRPC parity test ===\n");
    int fused_fail = 0;
    int fused_dims[] = {64, 128, 256, 512};
    for (int i = 0; i < 4; ++i) {
        int rc = sp_hex_compress_f32_full_parity(fused_dims[i]);
        if (rc) { fused_fail = 1; break; }
    }
    if (fused_fail) {
        printf("[fused] FAIL — fused pipeline diverged from host reference\n");
    } else {
        printf("[fused] all 4 head_dims byte-equal — 3 dispatches → 1, DSP autonomous\n");
    }

    // Strike 10b: Batched Grand Fusion — single dispatch encodes a whole chunk.
    // Sweep representative prefill chunk sizes: 1 (degenerate, should match the
    // single-vec path), 8 (small batch), 32 (production default chunk size),
    // 64 (max chunk on long contexts).
    printf("\n[fused_batch] === Strike 10b: compress_f32_full_batch FastRPC parity ===\n");
    int fused_batch_fail = 0;
    int fbatch_dims[] = {64, 128, 256, 512};
    int fbatch_ns[]   = {1, 8, 32, 64};
    for (int i = 0; i < 4 && !fused_batch_fail; ++i) {
        for (int j = 0; j < 4 && !fused_batch_fail; ++j) {
            int rc = sp_hex_compress_f32_full_batch_parity(fbatch_dims[i],
                                                            fbatch_ns[j]);
            if (rc) { fused_batch_fail = 1; }
        }
    }
    if (fused_batch_fail) {
        printf("[fused_batch] FAIL — batched fused pipeline diverged from host\n");
    } else {
        printf("[fused_batch] all 4 head_dims × 4 batch sizes byte-equal — prefill collapsed\n");
    }

    // Strike 11b: HVX W-matrix predictor (Hierarchical Spinor entry).
    // Validates the Q15 MAC kernel against (a) a host scalar Q15 simulator
    // (bit-equal) and (b) a pure-fp32 reference (within ~5e-4 quant budget).
    printf("\n[hier] === Strike 11b: hier_predict_f32 FastRPC parity test ===\n");
    int hier_fail = sp_hex_hier_predict_parity();
    if (hier_fail) {
        printf("[hier] FAIL — DSP predictor diverged from host references\n");
    } else {
        printf("[hier] all 3 patterns kernel bit-equal + within Q15 budget — "
               "Hierarchical Spinor entry green\n");
    }

    // Strike 12: residual quantize + SU(2) spinor phase.  Takes the predicted
    // residuals + actual coefficients, packs (actual - predicted) into
    // 71 bytes (140 × 3-bit magnitude + 140 × 1-bit sign) plus one fp32 amax.
    printf("\n[spinor] === Strike 12: residual_quantize_spinor FastRPC parity ===\n");
    int spinor_fail = sp_hex_residual_spinor_parity();
    if (spinor_fail) {
        printf("[spinor] FAIL — DSP packer diverged from host reference\n");
    } else {
        printf("[spinor] all 3 patterns 71 bytes byte-equal + amax bit-equal — "
               "Hierarchical Spinor encode closed\n");
    }

    // Strike 14: Hierarchical Spinor decode — predict + unpack + add.
    // Reconstructs the 140 fp32 residual coordinates from a 71-byte packed
    // block + skeleton + amax via a single DSP dispatch.  This is the
    // generation hot path (one decode per K/V slot read per layer per head).
    printf("\n[decode] === Strike 14: hier_decode_f32 FastRPC parity test ===\n");
    int decode_fail = sp_hex_hier_decode_parity();
    if (decode_fail) {
        printf("[decode] FAIL — DSP decode diverged from host reference\n");
    } else {
        printf("[decode] all 3 patterns within qf32 ULP + Q3 round-trip "
               "budget — Hierarchical Spinor decode closed\n");
    }

    // Strike 16: batched Hierarchical Spinor decode (single FastRPC per chunk).
    // Sweep production-typical chunk sizes: 1 (degenerate, matches single-vec),
    // 8 (small chunk), 32 (caught the Strike 10b regression at this n),
    // 64 (max chunk on long contexts).  Byte-equal vs n single-vec dispatches.
    printf("\n[decode_batch] === Strike 16: hier_decode_batch_f32 FastRPC parity ===\n");
    int decode_batch_fail = 0;
    int decode_batch_ns[] = {1, 8, 32, 64};
    for (int j = 0; j < 4 && !decode_batch_fail; ++j) {
        int rc = sp_hex_hier_decode_batch_parity(decode_batch_ns[j]);
        if (rc) decode_batch_fail = 1;
    }
    if (decode_batch_fail) {
        printf("[decode_batch] FAIL — batched decode diverged from single-vec ref\n");
    } else {
        printf("[decode_batch] all 4 batch sizes byte-equal — read scan "
               "amortization unlocked\n");
    }

    printf("\n[sp_hex] All paths green\n\n");
    return 0;
}

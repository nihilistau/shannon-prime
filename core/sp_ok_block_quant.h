/* sp_ok_block_quant.h — Phase 15: block-scale fused storage.
 *
 * Reads GGUF block-quant formats (Q8_0, Q4_0) and fuses each block's
 * fp16 scale with our Frobenius element π^k into a per-block pair of
 * O_K integers (B_a, B_b). The int4/int8 codepoints inside the GGUF
 * file are copied byte-for-byte — no dequant, no re-encode.
 *
 * Math (full derivation in PHASE_15_BLOCK_QUANT.md):
 *
 *   GGUF stores: W_continuous[k] = w_int[k] · block_scale          (scalar)
 *   Frobenius:   W_ring[k]       = W_continuous[k] · π^k           (O_K element)
 *                                = w_int[k] · (block_scale · π^k)
 *
 *   Let π^k = (π_a, π_b) in O_K. Define per-block fused integers:
 *
 *     B_a = round(scale_recip · block_scale · π_a)
 *     B_b = round(scale_recip · block_scale · π_b)
 *
 *   At decode time, the lifted coordinate of weight k is simply:
 *     W_ring[k] = (w_int[k] · B_a, w_int[k] · B_b)
 *
 *   So w_int[k] stays a tiny scalar, the (a, b) lift is a per-block constant.
 *
 * Storage layout: AoS at the block granularity, one block per cache line.
 *   Q8: 64-byte block (32 int8 + 16 bytes (B_a, B_b) + 16 reserved)
 *   Q4: 32-byte block (16 packed nybbles + 16 bytes (B_a, B_b))
 *
 * This file is C11, depends only on stdint + sp_ok_arith.h.
 */

#ifndef SP_OK_BLOCK_QUANT_H
#define SP_OK_BLOCK_QUANT_H

#include "sp_ok_arith.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SP_OK_BLOCK_SIZE 32

/* ---------- Q8 block (32 int8 codepoints, one cache line) -------------- */

#if defined(__GNUC__) || defined(__clang__)
#  define SP_OK_BLK_ALIGN __attribute__((aligned(64)))
#elif defined(_MSC_VER)
#  define SP_OK_BLK_ALIGN __declspec(align(64))
#else
#  define SP_OK_BLK_ALIGN
#endif

typedef struct SP_OK_BLK_ALIGN sp_ok_q8_block_s {
    int64_t B_a;                            /* 8 B: π_a^k · block_scale · scale_recip */
    int64_t B_b;                            /* 8 B: π_b^k · block_scale · scale_recip */
    int64_t reserved_block_min_a;           /* 8 B: future asymmetric zero-point (Q4_K etc.) */
    int64_t reserved_block_min_b;           /* 8 B: future */
    int8_t  packed[SP_OK_BLOCK_SIZE];       /* 32 B: GGUF int8 codepoints, untouched */
} sp_ok_q8_block_t;

/* Q4 block: 32 packed int4 codepoints (16 bytes) + (B_a, B_b). */

#if defined(__GNUC__) || defined(__clang__)
#  define SP_OK_BLK_Q4_ALIGN __attribute__((aligned(32)))
#elif defined(_MSC_VER)
#  define SP_OK_BLK_Q4_ALIGN __declspec(align(32))
#else
#  define SP_OK_BLK_Q4_ALIGN
#endif

typedef struct SP_OK_BLK_Q4_ALIGN sp_ok_q4_block_s {
    int64_t B_a;                            /* 8 B */
    int64_t B_b;                            /* 8 B */
    uint8_t packed[SP_OK_BLOCK_SIZE / 2];   /* 16 B: two int4 nybbles per byte */
} sp_ok_q4_block_t;

/* ---------- Tensor descriptors ----------------------------------------- */

typedef struct {
    sp_ok_q8_block_t* blocks;    /* numel / SP_OK_BLOCK_SIZE blocks */
    size_t            numel;     /* must be multiple of SP_OK_BLOCK_SIZE */
    size_t            n_blocks;
    int16_t           frobenius_p;
    int16_t           frobenius_k;
    int32_t           reserved;
} sp_ok_block_q8_tensor;

typedef struct {
    sp_ok_q4_block_t* blocks;
    size_t            numel;
    size_t            n_blocks;
    int16_t           frobenius_p;
    int16_t           frobenius_k;
    int32_t           reserved;
} sp_ok_block_q4_tensor;

/* ---------- GGUF block layouts (mirror of ggml-common.h) --------------- */
/* Note: we don't include ggml-common.h here to keep the math submodule
 * standalone. The caller passes pointers to these structs; we redeclare
 * them with identical layouts so the import functions can stride through
 * the GGUF tensor data without a vendor dependency. */

typedef struct {
    uint16_t d;                                  /* fp16 block scale */
    int8_t   qs[SP_OK_BLOCK_SIZE];               /* 32 int8 quants */
} sp_gguf_block_q8_0;  /* 34 bytes per block */

typedef struct {
    uint16_t d;                                  /* fp16 block scale */
    uint8_t  qs[SP_OK_BLOCK_SIZE / 2];           /* 16 bytes, two int4 nybbles per byte */
} sp_gguf_block_q4_0;  /* 18 bytes per block */

/* ---------- Importers --------------------------------------------------- */

/* Convert N GGUF Q8_0 blocks (continuous in src) into N fused
 * sp_ok_q8_block_t blocks in dst. Caller has already allocated
 * dst.blocks[0..n_blocks). Returns false on numel mismatch. */
int sp_ok_block_q8_from_gguf_q8_0(
    sp_ok_block_q8_tensor* dst,
    const sp_gguf_block_q8_0* src,
    size_t n_blocks,
    int64_t scale_recip,
    int64_t p,
    int64_t k);

/* Same for GGUF Q4_0. The 16-byte nybble buffer is copied byte-for-byte,
 * including GGUF's specific {low4 of byte 0 = elem 0, high4 of byte 0 =
 * elem 16, low4 of byte 1 = elem 1, ...} interleave — see
 * sp_ok_block_q4_decode_pair for the matching decoder. */
int sp_ok_block_q4_from_gguf_q4_0(
    sp_ok_block_q4_tensor* dst,
    const sp_gguf_block_q4_0* src,
    size_t n_blocks,
    int64_t scale_recip,
    int64_t p,
    int64_t k);

/* ---------- Inline decoders for kernels -------------------------------- */

/* GGUF Q4_0 stores 32 int4 codepoints as 16 bytes:
 *   byte i  bits [3:0] = elem i      (low nybble)
 *   byte i  bits [7:4] = elem i+16   (high nybble)
 * Each codepoint is signed int4 in [-8, 7] after the unbiased shift
 * (GGUF stores them with a +8 bias; subtract 8 to recover signed).
 *
 * Returns the signed int4 value for codepoint `idx` (0..31). */
static inline int8_t sp_ok_block_q4_decode_codepoint(const uint8_t* packed,
                                                       int idx) {
    /* GGUF layout: idx in [0, 16) -> low nybble of packed[idx];
     *              idx in [16, 32) -> high nybble of packed[idx - 16]. */
    uint8_t byte;
    if (idx < 16) {
        byte = packed[idx] & 0x0F;
    } else {
        byte = (packed[idx - 16] >> 4) & 0x0F;
    }
    /* GGUF stores int4 with +8 bias; subtract to recover [-8, 7]. */
    return (int8_t)((int)byte - 8);
}

#ifdef __cplusplus
}
#endif

#endif /* SP_OK_BLOCK_QUANT_H */

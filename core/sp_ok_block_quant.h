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

/* Phase 15b: Q4_1 block. GGUF Q4_1 stores per-block (d, m) where
 *   W[k] = d * x_int_unbiased[k] + m
 * After Frobenius lift:
 *   W_ring[k] = x_int_unbiased[k] * (d·π^k) + (m·π^k)
 *             = x_int_unbiased[k] * (B_a, B_b) + (M_a, M_b)
 *
 * 48 bytes total per 32 elements = 1.5 B/elem. Half-cache-line on
 * 64-B-line systems, fits perfectly. Q4_1 nybbles are UNSIGNED [0, 15]
 * — the +8 bias is absorbed in m, no decode-side subtraction needed. */

#if defined(__GNUC__) || defined(__clang__)
#  define SP_OK_BLK_Q4_1_ALIGN __attribute__((aligned(64)))
#elif defined(_MSC_VER)
#  define SP_OK_BLK_Q4_1_ALIGN __declspec(align(64))
#else
#  define SP_OK_BLK_Q4_1_ALIGN
#endif

typedef struct SP_OK_BLK_Q4_1_ALIGN sp_ok_q4_1_block_s {
    int64_t B_a;                            /* 8 B: d·π_a · scale_recip */
    int64_t B_b;                            /* 8 B */
    int64_t M_a;                            /* 8 B: m·π_a · scale_recip */
    int64_t M_b;                            /* 8 B */
    uint8_t packed[SP_OK_BLOCK_SIZE / 2];   /* 16 B: unsigned 4-bit nybbles */
} sp_ok_q4_1_block_t;

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

typedef struct {
    sp_ok_q4_1_block_t* blocks;
    size_t              numel;
    size_t              n_blocks;
    int16_t             frobenius_p;
    int16_t             frobenius_k;
    int32_t             reserved;
} sp_ok_block_q4_1_tensor;

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

typedef struct {
    uint16_t d;                                  /* fp16 block scale */
    uint16_t m;                                  /* fp16 block min */
    uint8_t  qs[SP_OK_BLOCK_SIZE / 2];           /* 16 bytes, unsigned 4-bit nybbles */
} sp_gguf_block_q4_1;  /* 20 bytes per block */

/* Q4_K super-block: 256 elements = 8 sub-blocks of 32 elements each.
 * Each sub-block carries a 6-bit scale and 6-bit min, encoded in the
 * 12-byte `scales` field via the get_scale_min_k4 helper layout. The
 * super-block-level d and dmin are fp16. Dequant per element:
 *   W[k] = d·sc[sub_idx]·q_int[k] − dmin·m[sub_idx]
 * where q_int[k] is the unsigned 4-bit codepoint. */
#define SP_OK_Q4_K_SUPER         256
#define SP_OK_Q4_K_SCALES_BYTES  12
#define SP_OK_Q4_K_SUBBLOCKS     (SP_OK_Q4_K_SUPER / SP_OK_BLOCK_SIZE)  /* 8 */
typedef struct {
    uint16_t d;                                       /* fp16 super-block scale */
    uint16_t dmin;                                    /* fp16 super-block min scale */
    uint8_t  scales[SP_OK_Q4_K_SCALES_BYTES];         /* 6-bit packed (sc, m) per sub-block */
    uint8_t  qs[SP_OK_Q4_K_SUPER / 2];                /* 128 bytes, 4-bit packed quants */
} sp_gguf_block_q4_K;  /* 144 bytes per 256 elements = 4.5 bits/elem */

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

/* Phase 15b: GGUF Q4_1 importer. Q4_1 nybbles are UNSIGNED (no -8 bias
 * subtraction at decode); the asymmetric offset lives in the per-block
 * `m` term which gets fused into M_a / M_b at load time. */
int sp_ok_block_q4_1_from_gguf_q4_1(
    sp_ok_block_q4_1_tensor* dst,
    const sp_gguf_block_q4_1* src,
    size_t n_blocks,
    int64_t scale_recip,
    int64_t p,
    int64_t k);

/* Phase 15c: GGUF Q4_K importer. Fans each 256-element super-block out
 * to 8 sp_ok_q4_1_block_t sub-blocks (32 elements each). Per-sub-block
 *   B_sub = d · sc · π^k         (positive contribution)
 *   M_sub = − dmin · m · π^k     (note the sign — Q4_K dequant is
 *                                  W = d·sc·q − dmin·m, so to match the
 *                                  Q4_1 kernel's convention q·B + M we
 *                                  store M = −dmin·m·π^k)
 * The nybble layout is repacked into our standard Q4_1 byte-i-low /
 * byte-i-high-+16 ordering. */
int sp_ok_block_q4_K_from_gguf_q4_K(
    sp_ok_block_q4_1_tensor* dst,
    const sp_gguf_block_q4_K* src,
    size_t n_super_blocks,             /* numel / 256 */
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
    /* Q4_0: GGUF stores int4 with +8 bias; subtract to recover [-8, 7]. */
    return (int8_t)((int)byte - 8);
}

/* Q4_1 decoder: same interleaved layout as Q4_0 but the bias is absorbed
 * in the per-block `m` term, not in the codepoint. Returns the UNSIGNED
 * nybble in [0, 15]. */
static inline uint8_t sp_ok_block_q4_1_decode_codepoint(const uint8_t* packed,
                                                         int idx) {
    if (idx < 16) return (uint8_t)(packed[idx] & 0x0F);
    return (uint8_t)((packed[idx - 16] >> 4) & 0x0F);
}

#ifdef __cplusplus
}
#endif

#endif /* SP_OK_BLOCK_QUANT_H */

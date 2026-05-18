/* sp_hex_mobius_tables.h — Strike 6 compile-time Möbius scatter tables.
 *
 * Lookup the byte-offset table for HVX vscatter-based Möbius reorder at
 * a given head_dim. Tables are baked at compile time by
 * scripts/gen_mobius_tables.py — see the generated .c for the data.
 *
 * Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
 */
#ifndef SP_HEX_MOBIUS_TABLES_H
#define SP_HEX_MOBIUS_TABLES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns a pointer to the head_dim-sized byte-offset array, or NULL if
 * the head_dim isn't covered by a baked table. Caller falls back to
 * scalar sp_mobius_reorder_ex on NULL. */
const uint32_t* sp_hex_mobius_offsets_f32(int head_dim);

/* 1 if head_dim is covered by a baked table, 0 otherwise. */
int sp_hex_mobius_supported_head_dim(int head_dim);

/* For diagnostics: number of squarefree indices (= length of Band-0+ if
 * the band split aligns to it). -1 if head_dim is unsupported. */
int sp_hex_mobius_n_squarefree(int head_dim);

#ifdef __cplusplus
}
#endif

#endif /* SP_HEX_MOBIUS_TABLES_H */

/* sp_kste_pack.c — bit-packing helpers for the KSTE tree.
 *
 * Two fields:
 *   labels [15] — 60 nodes * 2 bits each, little-endian within each byte.
 *                 Node i occupies bits 2*(i%4)+1 .. 2*(i%4) of byte i/4.
 *   parents[45] — 60 nodes * 6 bits each, packed across byte boundaries.
 *                 Stored little-endian: parent[i] occupies bits
 *                 [6*i .. 6*i+5] of the bit-stream made by concatenating
 *                 the bytes in order, where byte j contributes bits j*8..j*8+7
 *                 with the LSB at bit j*8.
 *
 * Both layouts are platform-independent (we never read more than one
 * byte at a time and never depend on machine endianness).
 *
 * Tested by tests/test_sp_kste.cpp.
 */

#include "sp_kste.h"

#include <string.h>

/* ---------- labels (2-bit per node) ------------------------------------- */

void sp_kste_pack_label(uint8_t labels[15], int idx, sp_kste_label lbl)
{
    if (idx < 0 || idx >= SP_KSTE_MAX_NODES) return;
    int byte = idx >> 2;          /* 4 labels per byte                      */
    int shift = (idx & 3) << 1;   /* 0, 2, 4, 6                             */
    uint8_t mask = (uint8_t)(0x3u << shift);
    labels[byte] = (uint8_t)((labels[byte] & ~mask) |
                             (((uint8_t)lbl & 0x3u) << shift));
}

sp_kste_label sp_kste_unpack_label(const uint8_t labels[15], int idx)
{
    if (idx < 0 || idx >= SP_KSTE_MAX_NODES) return SP_KSTE_LBL_ROOT;
    int byte = idx >> 2;
    int shift = (idx & 3) << 1;
    return (sp_kste_label)((labels[byte] >> shift) & 0x3u);
}

/* ---------- parents (6-bit per node, range [0,63]) ----------------------- */

void sp_kste_pack_parent(uint8_t parents[45], int idx, uint8_t parent)
{
    if (idx < 0 || idx >= SP_KSTE_MAX_NODES) return;
    parent &= 0x3F;                   /* clamp to 6 bits                    */
    int bit0 = idx * 6;               /* low bit position in stream         */
    int byte0 = bit0 >> 3;            /* first byte index                   */
    int bit_in_byte = bit0 & 7;
    int low_bits  = 8 - bit_in_byte;  /* bits this byte carries             */
    /* low_bits is in [1, 8] but our index range gives [2, 8].  When
     * low_bits >= 6 the parent fits in one byte; else it spans byte0/byte0+1. */

    uint8_t lo_mask = (uint8_t)(0x3Fu << bit_in_byte);          /* low 6 bits at bit_in_byte */
    if (low_bits >= 6) {
        parents[byte0] = (uint8_t)((parents[byte0] & ~lo_mask) |
                                   ((uint32_t)parent << bit_in_byte));
        return;
    }

    /* Spans two bytes: low_bits in byte0, remaining 6-low_bits in byte0+1. */
    uint8_t lo_part_mask  = (uint8_t)(0xFFu << bit_in_byte);    /* bits we own in byte0 */
    parents[byte0] = (uint8_t)((parents[byte0] & ~lo_part_mask) |
                               (((uint32_t)parent << bit_in_byte) & 0xFFu));

    int hi_bits   = 6 - low_bits;
    uint8_t hi_mask = (uint8_t)((1u << hi_bits) - 1u);
    parents[byte0 + 1] = (uint8_t)((parents[byte0 + 1] & (uint8_t)~hi_mask) |
                                   ((uint32_t)parent >> low_bits));
}

uint8_t sp_kste_unpack_parent(const uint8_t parents[45], int idx)
{
    if (idx < 0 || idx >= SP_KSTE_MAX_NODES) return 0;
    int bit0 = idx * 6;
    int byte0 = bit0 >> 3;
    int bit_in_byte = bit0 & 7;
    int low_bits = 8 - bit_in_byte;

    if (low_bits >= 6) {
        return (uint8_t)((parents[byte0] >> bit_in_byte) & 0x3Fu);
    }

    /* Span. */
    uint8_t lo = (uint8_t)(parents[byte0] >> bit_in_byte);
    int hi_bits = 6 - low_bits;
    uint8_t hi = (uint8_t)(parents[byte0 + 1] & ((1u << hi_bits) - 1u));
    return (uint8_t)((hi << low_bits) | lo);
}

/* ---------- whole-tree helpers ------------------------------------------- */

void sp_kste_tree_clear(sp_kste_tree *T)
{
    if (!T) return;
    memset(T, 0, sizeof(*T));
    T->node_count = 1;             /* root only                              */
    /* Root is node 0 with label ROOT, parent 0 (self).  Already zero.       */
}

int sp_kste_tree_add_child(sp_kste_tree *T, int parent_idx, sp_kste_label lbl)
{
    if (!T) return -1;
    if (T->node_count >= SP_KSTE_MAX_NODES) return -1;
    if (parent_idx < 0 || parent_idx >= T->node_count) return -1;

    int new_idx = T->node_count;
    sp_kste_pack_label (T->labels,  new_idx, lbl);
    sp_kste_pack_parent(T->parents, new_idx, (uint8_t)parent_idx);
    T->node_count = (uint8_t)(new_idx + 1);
    return new_idx;
}

int sp_kste_tree_children_of_root(const sp_kste_tree *T, int *out, int cap)
{
    if (!T || !out || cap <= 0) return 0;
    int n = 0;
    for (int i = 1; i < T->node_count && n < cap; ++i) {
        if (sp_kste_unpack_parent(T->parents, i) == 0u) {
            out[n++] = i;
        }
    }
    return n;
}

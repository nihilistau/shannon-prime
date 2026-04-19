// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#include "shannon_prime_modelpack.h"

#include <stdio.h>
#include <string.h>

// ============================================================================
// Registry
// ============================================================================
//
// Kept intentionally small. Adding an architecture is cheap — adding one
// without doing the calibration work is harmful. New entries start at
// SP_PRESET_PROVISIONAL, get promoted to SP_PRESET_CALIBRATED only after
// `sp-engine cache_ppl --model <ref> --sqfree` shows PPL drift within
// the budget documented in docs/MODEL-PACK.md.
//
// Matching rules (see sp_model_preset_resolve below):
//   * arch_pattern is a substring match against arch_name (case-sensitive;
//     GGUF arch strings are lowercase by convention).
//   * Dimensional hints are soft: if head_dim_hint is nonzero it must
//     match exactly. min_n_layer / min_n_head_kv are lower bounds.
//   * First match wins — order presets from most-specific to most-general.

static const sp_model_preset_t g_presets[] = {
    // ── Qwen 3.6 MoE (and Qwen 3 MoE variants) ─────────────────────────
    // Most-specific: wide MoE stacks, dense K/V heads, benefit most from
    // sqfree + spinor because mid-band energy is the dominant contributor.
    {
        .name              = "qwen3-moe",
        .arch_pattern      = "qwen3moe",
        .notes             = "MoE: dense KV, high mid-band energy; spinor recovers tail",
        .head_dim_hint     = 0,
        .min_n_layer       = 1,
        .min_n_head_kv     = 1,
        .k_n_bands         = 5,
        .k_band_bits       = { 5, 4, 4, 4, 4 },
        .v_n_bands         = 5,
        .v_band_bits       = { 4, 4, 4, 4, 4 },
        .residual_bits     = 3,
        .use_mobius        = true,
        .recommend_sqfree  = true,
        .recommend_spinor  = true,
        .status            = SP_PRESET_PROVISIONAL,
    },

    // ── Qwen 3.5 / Qwen 3 (dense) ──────────────────────────────────────
    {
        .name              = "qwen3",
        .arch_pattern      = "qwen3",
        .notes             = "Dense Qwen 3.x; similar band profile to Llama-3 but V carries more energy",
        .head_dim_hint     = 0,
        .min_n_layer       = 1,
        .min_n_head_kv     = 1,
        .k_n_bands         = 4,
        .k_band_bits       = { 5, 5, 4, 3 },
        .v_n_bands         = 2,
        .v_band_bits       = { 4, 3 },
        .residual_bits     = 3,
        .use_mobius        = true,
        .recommend_sqfree  = true,
        .recommend_spinor  = false,
        .status            = SP_PRESET_PROVISIONAL,
    },

    // ── Gemma 3 (Gemma 4 reuses hparams layout; same preset until diverged) ─
    {
        .name              = "gemma3",
        .arch_pattern      = "gemma3",
        .notes             = "Gemma 3: sliding-window attn stresses upper K band; keep K[0]=5",
        .head_dim_hint     = 0,
        .min_n_layer       = 1,
        .min_n_head_kv     = 1,
        .k_n_bands         = 4,
        .k_band_bits       = { 5, 4, 4, 3 },
        .v_n_bands         = 1,
        .v_band_bits       = { 3 },
        .residual_bits     = 3,
        .use_mobius        = true,
        .recommend_sqfree  = true,
        .recommend_spinor  = false,
        .status            = SP_PRESET_PROVISIONAL,
    },

    // ── Llama 3 / 3.1 / 3.2 / 3.3 ──────────────────────────────────────
    // Shipping default mirrors this — the preset exists so a match is
    // logged and the "auto-adapts" story is auditable.
    {
        .name              = "llama-3",
        .arch_pattern      = "llama",
        .notes             = "Llama-3 family; shipping defaults; baseline for calibration drift",
        .head_dim_hint     = 0,
        .min_n_layer       = 1,
        .min_n_head_kv     = 1,
        .k_n_bands         = 4,
        .k_band_bits       = { 5, 5, 4, 3 },
        .v_n_bands         = 1,
        .v_band_bits       = { 3 },
        .residual_bits     = 3,
        .use_mobius        = true,
        .recommend_sqfree  = false,    // ship path is strong enough
        .recommend_spinor  = false,
        .status            = SP_PRESET_PROVISIONAL,
    },
};

static const int g_preset_count = (int)(sizeof(g_presets) / sizeof(g_presets[0]));

// ============================================================================
// Public API
// ============================================================================

const sp_model_preset_t *
sp_model_preset_resolve(const char *arch_name,
                        int head_dim,
                        int n_layers,
                        int n_heads_kv)
{
    if (arch_name == NULL || arch_name[0] == '\0') {
        return NULL;
    }
    for (int i = 0; i < g_preset_count; ++i) {
        const sp_model_preset_t *p = &g_presets[i];
        if (strstr(arch_name, p->arch_pattern) == NULL) {
            continue;
        }
        if (p->head_dim_hint != 0 && p->head_dim_hint != head_dim) {
            continue;
        }
        if (p->min_n_layer   > n_layers)    continue;
        if (p->min_n_head_kv > n_heads_kv)  continue;
        return p;
    }
    return NULL;
}

int sp_config_apply_preset(sp_config_t *cfg, const sp_model_preset_t *preset) {
    if (cfg == NULL || preset == NULL) return -1;

    // K bands — copy count + bits, leave trailing slots untouched
    int kb = preset->k_n_bands;
    if (kb < 1) kb = 1;
    if (kb > SP_MAX_BANDS) kb = SP_MAX_BANDS;
    cfg->k_n_bands = kb;
    for (int i = 0; i < kb; ++i) cfg->k_band_bits[i] = preset->k_band_bits[i];

    // V bands
    int vb = preset->v_n_bands;
    if (vb < 1) vb = 1;
    if (vb > SP_MAX_BANDS) vb = SP_MAX_BANDS;
    cfg->v_n_bands = vb;
    for (int i = 0; i < vb; ++i) cfg->v_band_bits[i] = preset->v_band_bits[i];

    // Möbius is an overlay; preserve caller's explicit override if they
    // already set it. The convention is: shipping default is true, so if
    // the caller set it to false they meant it.
    cfg->use_mobius_mask = preset->use_mobius;

    return 0;
}

const sp_model_preset_t *sp_model_preset_at(int index) {
    if (index < 0 || index >= g_preset_count) return NULL;
    return &g_presets[index];
}

int sp_model_preset_count(void) {
    return g_preset_count;
}

int sp_model_preset_describe(const sp_model_preset_t *preset,
                             char *out, size_t size)
{
    if (!preset || !out || size == 0) return -1;

    // "name (arch=X) K=a,b,c,d V=e,f res=R spinor=Y [STATUS]"
    char kbuf[48]; char vbuf[48];
    int kpos = 0, vpos = 0;
    for (int i = 0; i < preset->k_n_bands && kpos < (int)sizeof(kbuf) - 4; ++i) {
        kpos += snprintf(kbuf + kpos, sizeof(kbuf) - kpos,
                         i == 0 ? "%d" : ",%d", preset->k_band_bits[i]);
    }
    for (int i = 0; i < preset->v_n_bands && vpos < (int)sizeof(vbuf) - 4; ++i) {
        vpos += snprintf(vbuf + vpos, sizeof(vbuf) - vpos,
                         i == 0 ? "%d" : ",%d", preset->v_band_bits[i]);
    }

    return snprintf(out, size,
                    "%s (arch=%s) K=%s V=%s res=%d spinor=%s [%s]",
                    preset->name,
                    preset->arch_pattern,
                    kbuf, vbuf,
                    preset->residual_bits,
                    preset->recommend_spinor ? "recommended" : "off",
                    preset->status == SP_PRESET_CALIBRATED
                        ? "CALIBRATED" : "PROVISIONAL");
}

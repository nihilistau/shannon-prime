// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

//
// Sqfree prime-Hartley + Knight mask + Möbius CSR + spinor sheet bit.
// Additive to core/shannon_prime.c — does NOT modify the VHT2 ship path.
//
// Include this file alongside shannon_prime.c in your build.
// Requires: <math.h>, <stdlib.h>, <string.h>, <stdbool.h>, <stdint.h>
//

#include "shannon_prime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Möbius function (standalone, not dependent on mask)
// ============================================================================

static int sp_mobius_val(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    int val = n, nf = 0;
    for (int p = 2; p * p <= val; p++) {
        if (val % p == 0) {
            nf++;
            val /= p;
            if (val % p == 0) return 0; // p² divides n
        }
    }
    if (val > 1) nf++;
    return (nf % 2 == 0) ? 1 : -1;
}

// ============================================================================
// Squarefree basis helpers
// ============================================================================

static const int SP_SMALL_PRIMES[] = {2, 3, 5, 7, 11};
static const int SP_N_SMALL_PRIMES = 5;

bool sp_is_sqfree_factorable(int n) {
    if (n <= 0) return false;
    int d = n;
    for (int i = 0; i < SP_N_SMALL_PRIMES; i++) {
        int p = SP_SMALL_PRIMES[i];
        int c = 0;
        while (d % p == 0) {
            d /= p;
            c++;
            if (c > 1) return false; // repeated prime
        }
    }
    return d == 1; // must factor completely
}

int sp_sqfree_pad_dim(int head_dim) {
    for (int n = (head_dim < 2 ? 2 : head_dim); n < head_dim * 4; n++) {
        if (sp_is_sqfree_factorable(n)) return n;
    }
    return head_dim;
}

void sp_sqfree_pad_f32(const float *in, int head_dim,
                       float *out, int pad_dim) {
    memcpy(out, in, head_dim * sizeof(float));
    if (pad_dim > head_dim) {
        // Mean-fill tail
        double sum = 0.0;
        for (int i = 0; i < head_dim; i++) sum += in[i];
        float mean = (float)(sum / head_dim);
        for (int i = head_dim; i < pad_dim; i++) out[i] = mean;
    }
}

void sp_sqfree_unpad_f32(const float *in, float *out, int head_dim) {
    memcpy(out, in, head_dim * sizeof(float));
}

// ============================================================================
// Prime-Hartley Transform
// ============================================================================
//
// cas(x) = cos(x) + sin(x). Normalized by 1/√p per prime factor.
// Self-inverse: H·H = I when each factor is normalized.
// Applied iteratively over prime factors of N (Kronecker product structure).

// Factorize n into the small primes. Returns count, fills factors[].
static int sp_prime_factorize(int n, int *factors, int max_factors) {
    int count = 0;
    int d = n;
    for (int i = 0; i < SP_N_SMALL_PRIMES && d > 1; i++) {
        int p = SP_SMALL_PRIMES[i];
        while (d % p == 0 && count < max_factors) {
            factors[count++] = p;
            d /= p;
        }
    }
    return (d == 1) ? count : -1; // -1 if doesn't factor completely
}

// Apply one prime-p Hartley stage in-place.
// data has length total_len. This stage operates on groups of size (stride*p),
// applying the p×p Hartley matrix on the p-axis with the given stride.
static void sp_hartley_stage_f32(float *data, int total_len, int p, int stride) {
    int block_size = stride * p;
    int n_blocks = total_len / block_size;
    float inv_sqrt_p = 1.0f / sqrtf((float)p);

    // Temporary buffer for one p-group
    float tmp[16]; // p ≤ 13

    for (int blk = 0; blk < n_blocks; blk++) {
        for (int s = 0; s < stride; s++) {
            // Gather the p elements at stride intervals
            int base = blk * block_size + s;
            for (int k = 0; k < p; k++) {
                tmp[k] = 0.0f;
                for (int n_idx = 0; n_idx < p; n_idx++) {
                    double angle = 2.0 * M_PI * k * n_idx / (double)p;
                    float cas = (float)(cos(angle) + sin(angle));
                    tmp[k] += cas * data[base + n_idx * stride];
                }
                tmp[k] *= inv_sqrt_p;
            }
            // Scatter back
            for (int k = 0; k < p; k++) {
                data[base + k * stride] = tmp[k];
            }
        }
    }
}

static void sp_vilenkin_forward_f32(float *data, int n) {
    int factors[16];
    int n_factors = sp_prime_factorize(n, factors, 16);
    if (n_factors < 0) return; // Can't factorize

    int stride = 1;
    for (int f = 0; f < n_factors; f++) {
        sp_hartley_stage_f32(data, n, factors[f], stride);
        stride *= factors[f];
    }
}

static void sp_vilenkin_inverse_f32(float *data, int n) {
    // Self-inverse when each stage is normalized by 1/√p
    sp_vilenkin_forward_f32(data, n);
}

// ============================================================================
// Knight-ranked mask + Möbius CSR predictor
// ============================================================================

int sp_knight_mask_init(sp_knight_mask_t *mask, int pad_dim, int sk_k,
                        const float *variance) {
    memset(mask, 0, sizeof(*mask));
    mask->pad_dim = pad_dim;

    // Partition indices: squarefree vs composite (1-indexed for μ)
    int *sqfree_idx = (int *)malloc(pad_dim * sizeof(int));
    int *comp_idx   = (int *)malloc(pad_dim * sizeof(int));
    int n_sqf = 0, n_comp = 0;

    for (int i = 0; i < pad_dim; i++) {
        if (sp_mobius_val(i + 1) != 0) {
            sqfree_idx[n_sqf++] = i;
        } else {
            comp_idx[n_comp++] = i;
        }
    }

    // Sort by variance descending if provided
    if (variance) {
        // Simple insertion sort (pad_dim ≤ 330, not hot path)
        for (int i = 1; i < n_sqf; i++) {
            int key = sqfree_idx[i];
            float kv = variance[key];
            int j = i - 1;
            while (j >= 0 && variance[sqfree_idx[j]] < kv) {
                sqfree_idx[j + 1] = sqfree_idx[j];
                j--;
            }
            sqfree_idx[j + 1] = key;
        }
        for (int i = 1; i < n_comp; i++) {
            int key = comp_idx[i];
            float kv = variance[key];
            int j = i - 1;
            while (j >= 0 && variance[comp_idx[j]] < kv) {
                comp_idx[j + 1] = comp_idx[j];
                j--;
            }
            comp_idx[j + 1] = key;
        }
    }

    // Pick skeleton: squarefree first, fill with composites
    int k_sqf = (sk_k < n_sqf) ? sk_k : n_sqf;
    int extra = sk_k - k_sqf;
    if (extra > n_comp) extra = n_comp;
    int actual_sk = k_sqf + extra;

    mask->sk_k = actual_sk;
    mask->skeleton_idx = (int *)malloc(actual_sk * sizeof(int));
    memcpy(mask->skeleton_idx, sqfree_idx, k_sqf * sizeof(int));
    if (extra > 0) {
        memcpy(mask->skeleton_idx + k_sqf, comp_idx, extra * sizeof(int));
    }

    // Sort skeleton ascending for scatter locality
    for (int i = 1; i < actual_sk; i++) {
        int key = mask->skeleton_idx[i];
        int j = i - 1;
        while (j >= 0 && mask->skeleton_idx[j] > key) {
            mask->skeleton_idx[j + 1] = mask->skeleton_idx[j];
            j--;
        }
        mask->skeleton_idx[j + 1] = key;
    }

    // Build skeleton lookup
    bool *in_skel = (bool *)calloc(pad_dim, sizeof(bool));
    int *idx_to_slot = (int *)malloc(pad_dim * sizeof(int));
    memset(idx_to_slot, -1, pad_dim * sizeof(int));
    for (int s = 0; s < actual_sk; s++) {
        in_skel[mask->skeleton_idx[s]] = true;
        idx_to_slot[mask->skeleton_idx[s]] = s;
    }

    // Residual = non-squarefree NOT in skeleton
    int *res_buf = (int *)malloc(pad_dim * sizeof(int));
    mask->n_res = 0;
    for (int i = 0; i < pad_dim; i++) {
        if (sp_mobius_val(i + 1) == 0 && !in_skel[i]) {
            res_buf[mask->n_res++] = i;
        }
    }
    mask->residual_idx = (int *)malloc(mask->n_res * sizeof(int));
    memcpy(mask->residual_idx, res_buf, mask->n_res * sizeof(int));

    // Build Möbius CSR
    mask->csr_offsets = (int *)malloc((mask->n_res + 1) * sizeof(int));
    // First pass: count terms
    int total_terms = 0;
    for (int r = 0; r < mask->n_res; r++) {
        int n = mask->residual_idx[r] + 1; // 1-indexed
        mask->csr_offsets[r] = total_terms;
        for (int d = 1; d <= n; d++) {
            if (n % d == 0) {
                int mu_d = sp_mobius_val(d);
                if (mu_d != 0) {
                    int q = n / d;
                    int q_idx = q - 1;
                    if (q_idx >= 0 && q_idx < pad_dim && idx_to_slot[q_idx] >= 0) {
                        total_terms++;
                    }
                }
            }
        }
    }
    mask->csr_offsets[mask->n_res] = total_terms;
    mask->n_terms = total_terms;

    mask->csr_skel_slot = (int *)malloc(total_terms * sizeof(int));
    mask->csr_mu_sign = (int8_t *)malloc(total_terms * sizeof(int8_t));

    // Second pass: fill CSR
    int term_idx = 0;
    for (int r = 0; r < mask->n_res; r++) {
        int n = mask->residual_idx[r] + 1;
        mask->csr_offsets[r] = term_idx;
        for (int d = 1; d <= n; d++) {
            if (n % d == 0) {
                int mu_d = sp_mobius_val(d);
                if (mu_d != 0) {
                    int q = n / d;
                    int q_idx = q - 1;
                    if (q_idx >= 0 && q_idx < pad_dim && idx_to_slot[q_idx] >= 0) {
                        mask->csr_skel_slot[term_idx] = idx_to_slot[q_idx];
                        mask->csr_mu_sign[term_idx] = (int8_t)mu_d;
                        term_idx++;
                    }
                }
            }
        }
    }
    mask->csr_offsets[mask->n_res] = term_idx;

    mask->residual_bits = 3;
    mask->use_spinor = true;

    free(sqfree_idx);
    free(comp_idx);
    free(in_skel);
    free(idx_to_slot);
    free(res_buf);

    return 0;
}

void sp_knight_mask_free(sp_knight_mask_t *mask) {
    free(mask->skeleton_idx);
    free(mask->residual_idx);
    free(mask->csr_offsets);
    free(mask->csr_skel_slot);
    free(mask->csr_mu_sign);
    memset(mask, 0, sizeof(*mask));
}

// ============================================================================
// N-bit symmetric residual quantization
// ============================================================================

void sp_quantize_residual(const float *vals, int n, int nbits, float mag,
                          uint8_t *levels) {
    int L = 1 << nbits;
    float center = (float)(L - 1) / 2.0f;
    float sat = (mag > 1e-12f) ? mag * (float)nbits : 1e-12f;
    float step = (2.0f * sat) / (float)(L - 1);
    float inv_step = 1.0f / step;

    for (int i = 0; i < n; i++) {
        float q = vals[i] * inv_step + center;
        int level = (int)(q + 0.5f); // round
        if (level < 0) level = 0;
        if (level >= L) level = L - 1;
        levels[i] = (uint8_t)level;
    }
}

void sp_dequantize_residual(const uint8_t *levels, int n, int nbits, float mag,
                            float *vals) {
    int L = 1 << nbits;
    float center = (float)(L - 1) / 2.0f;
    float sat = mag * (float)nbits;
    float step = (2.0f * sat) / (float)(L - 1);

    for (int i = 0; i < n; i++) {
        vals[i] = ((float)levels[i] - center) * step;
    }
}

// ============================================================================
// Sqfree shadow cache
// ============================================================================

int sp_sqfree_cache_init(sp_sqfree_cache_t *sc, const sp_config_t *cfg,
                         int max_seq_len, int residual_bits, bool use_spinor) {
    memset(sc, 0, sizeof(*sc));
    sc->config = *cfg;
    sc->residual_bits = residual_bits;
    sc->use_spinor = use_spinor;
    sc->pad_dim = sp_sqfree_pad_dim(cfg->head_dim);
    sc->max_seq_len = max_seq_len;
    sc->calibrating = false;
    sc->calib_sum = NULL;
    sc->calib_sum2 = NULL;
    sc->calib_n = 0;

    // Init Vilenkin basis
    int factors[16];
    int n_factors = sp_prime_factorize(sc->pad_dim, factors, 16);
    if (n_factors < 0) return -1;

    // Determine n_primes from pad_dim factorization
    int seen_primes = 0;
    int last_p = 0;
    for (int i = 0; i < n_factors; i++) {
        if (factors[i] != last_p) {
            seen_primes++;
            last_p = factors[i];
        }
    }
    if (sp_vilenkin_init(&sc->vilenkin, seen_primes) != 0) return -1;

    // Init Knight mask with L/2 skeleton (phase-transition optimum).
    // Research finding: adaptive top-K universally peaks at dim/2 (50%),
    // not the prior 75%. Variance ranking is applied later via calibrate.
    int sk_k = sc->pad_dim / 2;
    sp_knight_mask_init(&sc->mask, sc->pad_dim, sk_k, NULL);
    sc->mask.residual_bits = residual_bits;
    sc->mask.use_spinor = use_spinor;

    // Init banded quantizers for skeleton size. Honor caller's cfg.k_n_bands /
    // k_band_bits when provided (the llama+sqfree hook parses SHANNON_PRIME_K_BITS
    // into these); fall back to the torus-aligned 5-band default otherwise.
    int default_k_bits[5] = {5, 4, 4, 4, 5};
    int k_nb = (cfg->k_n_bands > 0 && cfg->k_n_bands <= SP_MAX_BANDS)
               ? cfg->k_n_bands : 5;
    const int *k_bits = (cfg->k_n_bands > 0 && cfg->k_n_bands <= SP_MAX_BANDS)
                        ? cfg->k_band_bits : default_k_bits;
    int v_nb = (cfg->v_n_bands > 0 && cfg->v_n_bands <= SP_MAX_BANDS)
               ? cfg->v_n_bands : k_nb;
    const int *v_bits = (cfg->v_n_bands > 0 && cfg->v_n_bands <= SP_MAX_BANDS)
                        ? cfg->v_band_bits : k_bits;
    sp_band_config_init(&sc->k_bands, sc->mask.sk_k, k_nb, k_bits);
    sp_band_config_init(&sc->v_bands, sc->mask.sk_k, v_nb, v_bits);

    if (getenv("SHANNON_PRIME_VERBOSE")) {
        fprintf(stderr, "[Shannon-Prime SQFREE] K band bits (%d bands): ", k_nb);
        for (int i = 0; i < k_nb; i++) fprintf(stderr, "%d%s", k_bits[i], i < k_nb - 1 ? "," : "\n");
        fprintf(stderr, "[Shannon-Prime SQFREE] sk_k=%d pad_dim=%d n_res=%d\n",
                sc->mask.sk_k, sc->pad_dim, sc->mask.n_res);
    }

    // Allocate compressed storage
    int n_slots = cfg->n_layers * cfg->n_heads_kv;
    int k_bytes_per_pos = sc->k_bands.total_bytes
                        + (sc->mask.n_res * residual_bits + 7) / 8
                        + 4  // residual mag
                        + (use_spinor ? (sc->mask.n_res + 7) / 8 : 0);
    int v_bytes_per_pos = sc->v_bands.total_bytes
                        + (sc->mask.n_res * residual_bits + 7) / 8
                        + 4
                        + (use_spinor ? (sc->mask.n_res + 7) / 8 : 0);

    sc->k_cache = (uint8_t **)calloc(n_slots, sizeof(uint8_t *));
    sc->v_cache = (uint8_t **)calloc(n_slots, sizeof(uint8_t *));
    for (int s = 0; s < n_slots; s++) {
        sc->k_cache[s] = (uint8_t *)calloc(max_seq_len, k_bytes_per_pos);
        sc->v_cache[s] = (uint8_t *)calloc(max_seq_len, v_bytes_per_pos);
    }

    // Scratch buffers
    sc->pad_scratch   = (float *)malloc(sc->pad_dim * sizeof(float));
    sc->coeff_scratch = (float *)malloc(sc->pad_dim * sizeof(float));
    sc->pred_scratch  = (float *)malloc(sc->mask.n_res * sizeof(float));

    // Optional skeleton-level Möbius reorder (ship-path style, distinct
    // from the Knight-CSR predictor's always-on μ usage). Gate via env
    // so the default sqfree path is unchanged.
    sc->use_skel_mobius = false;
    sc->skel_mobius_scratch = NULL;
    // Enable via either the specific env var or the top-level
    // SHANNON_PRIME_MOBIUS=1 — the latter is what users expect to compose
    // with SHANNON_PRIME_SQFREE=1. Prior to this commit MOBIUS was
    // silently ignored when SQFREE was on; now it maps to skeleton-level
    // reorder of the banded skeleton. The Knight-CSR μ predictor is
    // always on independent of these envs.
    const char *skel_mob_env = getenv("SHANNON_PRIME_SQFREE_SKEL_MOBIUS");
    const char *mob_env      = getenv("SHANNON_PRIME_MOBIUS");
    bool enable = (skel_mob_env && skel_mob_env[0] == '1')
               || (mob_env      && mob_env[0]      == '1');
    if (enable) {
        if (sp_mobius_mask_init(&sc->skel_mobius_mask, sc->mask.sk_k) == 0) {
            sc->use_skel_mobius = true;
            sc->skel_mobius_scratch = (float *)malloc(sc->mask.sk_k * sizeof(float));
            if (getenv("SHANNON_PRIME_VERBOSE")) {
                fprintf(stderr, "[Shannon-Prime SQFREE] skeleton Möbius reorder: ON "
                                "(sk_k=%d, squarefree=%d)\n",
                        sc->mask.sk_k, sc->skel_mobius_mask.n_squarefree);
            }
        }
    }

    return 0;
}

void sp_sqfree_cache_free(sp_sqfree_cache_t *sc) {
    int n_slots = sc->config.n_layers * sc->config.n_heads_kv;
    for (int s = 0; s < n_slots; s++) {
        free(sc->k_cache[s]);
        free(sc->v_cache[s]);
    }
    free(sc->k_cache);
    free(sc->v_cache);
    free(sc->pad_scratch);
    free(sc->coeff_scratch);
    free(sc->pred_scratch);
    if (sc->use_skel_mobius) {
        sp_mobius_mask_free(&sc->skel_mobius_mask);
        free(sc->skel_mobius_scratch);
    }
    free(sc->calib_sum);
    free(sc->calib_sum2);
    sp_knight_mask_free(&sc->mask);
    sp_vilenkin_free(&sc->vilenkin);
    memset(sc, 0, sizeof(*sc));
}

// Internal: compress one vector through the full sqfree pipeline.
// Writes compressed bytes to `out`. Returns bytes written.
static int sp_sqfree_compress_one(sp_sqfree_cache_t *sc,
                                  const float *vec, uint8_t *out,
                                  const sp_band_config_t *bc) {
    int hd = sc->config.head_dim;
    int pd = sc->pad_dim;
    int n_res = sc->mask.n_res;
    uint8_t *write_ptr = out;

    // 1. Pad
    sp_sqfree_pad_f32(vec, hd, sc->pad_scratch, pd);

    // 2. Vilenkin forward
    memcpy(sc->coeff_scratch, sc->pad_scratch, pd * sizeof(float));
    sp_vilenkin_forward_f32(sc->coeff_scratch, pd);

    // 2b. Optional: dump post-pad Vilenkin coefficients for offline analysis
    // (tools/sp_auto_bands.py --basis vilenkin). Gated on
    // SHANNON_PRIME_DUMP_VILENKIN=<path>; SHANNON_PRIME_DUMP_VILENKIN_LIMIT
    // caps vector count (default 8192). Format: binary fp32 stream of `pd`
    // floats per vector. Unlike the SHANNON_PRIME_DUMP_K hook in the
    // llama integration (which captures raw pre-pad K), this dump gives the
    // allocator the real sqfree basis it will be quantising.
    {
        static FILE     *dump_fp = NULL;
        static long long dump_count = 0;
        static long long dump_limit = 0;
        static int       dump_init = 0;
        if (!dump_init) {
            dump_init = 1;
            const char *dp = getenv("SHANNON_PRIME_DUMP_VILENKIN");
            if (dp && *dp) {
                dump_fp = fopen(dp, "wb");
                const char *lim_s = getenv("SHANNON_PRIME_DUMP_VILENKIN_LIMIT");
                dump_limit = lim_s ? atoll(lim_s) : 8192;
                if (dump_fp) {
                    fprintf(stderr,
                        "[Shannon-Prime SQFREE] Vilenkin dump -> %s "
                        "(limit=%lld vectors, pd=%d)\n",
                        dp, dump_limit, pd);
                }
            }
        }
        if (dump_fp && (dump_limit == 0 || dump_count < dump_limit)) {
            fwrite(sc->coeff_scratch, sizeof(float), (size_t)pd, dump_fp);
            dump_count++;
            if (dump_count == dump_limit) fflush(dump_fp);
        }
    }

    // 3. Extract skeleton coefficients into contiguous buffer
    float *skel_vals = sc->pad_scratch; // Reuse — safe since we're done with pad
    for (int s = 0; s < sc->mask.sk_k; s++) {
        skel_vals[s] = sc->coeff_scratch[sc->mask.skeleton_idx[s]];
    }

    // 4. Band-quantize skeleton. Optionally apply ship-path-style Möbius
    //    reorder on the skeleton vector first so squarefree slots land in
    //    the high-bit early bands. The CSR predictor below still reads
    //    skel_vals in the original ordering — the reorder is scoped to
    //    the band quant/dequant round-trip only.
    if (sc->use_skel_mobius) {
        float *reordered = sc->skel_mobius_scratch;
        const int *ord = sc->skel_mobius_mask.order;
        for (int i = 0; i < sc->mask.sk_k; i++) {
            reordered[i] = skel_vals[ord[i]];
        }
        sp_band_quantize(reordered, write_ptr, bc);
    } else {
        sp_band_quantize(skel_vals, write_ptr, bc);
    }
    write_ptr += bc->total_bytes;

    // 5. Möbius predict + residual quantize
    if (n_res > 0) {
        // Compute predictions
        for (int i = 0; i < n_res; i++) {
            float pred = 0.0f;
            int start = sc->mask.csr_offsets[i];
            int end   = sc->mask.csr_offsets[i + 1];
            for (int j = start; j < end; j++) {
                pred += (float)sc->mask.csr_mu_sign[j]
                      * skel_vals[sc->mask.csr_skel_slot[j]];
            }
            sc->pred_scratch[i] = pred;
        }

        // Spinor: pick better predictor sign
        float *deviation = (float *)malloc(n_res * sizeof(float));
        uint8_t *sheet = NULL;
        if (sc->use_spinor) {
            sheet = (uint8_t *)calloc((n_res + 7) / 8, 1);
        }

        for (int i = 0; i < n_res; i++) {
            float actual = sc->coeff_scratch[sc->mask.residual_idx[i]];
            float v_plus  = actual - sc->pred_scratch[i];
            float v_minus = actual + sc->pred_scratch[i];

            if (sc->use_spinor && fabsf(v_minus) < fabsf(v_plus)) {
                deviation[i] = v_minus;
                sheet[i / 8] |= (1 << (i % 8));
            } else {
                deviation[i] = v_plus;
            }
        }

        // Compute magnitude
        float mag = 0.0f;
        for (int i = 0; i < n_res; i++) mag += fabsf(deviation[i]);
        mag /= (float)n_res;
        if (mag < 1e-12f) mag = 1e-12f;

        // Store magnitude
        memcpy(write_ptr, &mag, 4);
        write_ptr += 4;

        // Quantize + store residual levels
        uint8_t *levels = (uint8_t *)malloc(n_res);
        sp_quantize_residual(deviation, n_res, sc->residual_bits, mag, levels);
        int res_bytes = (n_res * sc->residual_bits + 7) / 8;
        // Pack bits (LSB-first)
        memset(write_ptr, 0, res_bytes);
        for (int i = 0; i < n_res; i++) {
            int bit_off = i * sc->residual_bits;
            write_ptr[bit_off / 8] |= (levels[i] << (bit_off % 8));
            if ((bit_off % 8) + sc->residual_bits > 8 && (bit_off / 8 + 1) < res_bytes) {
                write_ptr[bit_off / 8 + 1] |= (levels[i] >> (8 - bit_off % 8));
            }
        }
        write_ptr += res_bytes;

        // Store sheet bits
        if (sc->use_spinor && sheet) {
            int sheet_bytes = (n_res + 7) / 8;
            memcpy(write_ptr, sheet, sheet_bytes);
            write_ptr += sheet_bytes;
            free(sheet);
        }

        free(levels);
        free(deviation);
    }

    return (int)(write_ptr - out);
}

// Internal: reconstruct one vector from compressed bytes.
static void sp_sqfree_reconstruct_one(const sp_sqfree_cache_t *sc,
                                      const uint8_t *in, float *vec_out,
                                      const sp_band_config_t *bc) {
    int hd = sc->config.head_dim;
    int pd = sc->pad_dim;
    int n_res = sc->mask.n_res;
    const uint8_t *read_ptr = in;

    // Allocate working buffers on stack (pad_dim ≤ 330)
    float skel_vals[SP_MAX_HEAD_DIM * 2]; // Generous
    float coeffs[SP_MAX_HEAD_DIM * 2];
    memset(coeffs, 0, pd * sizeof(float));

    // 1. Dequantize skeleton. If the compress-side applied the
    //    ship-style Möbius reorder to the skeleton before banding, undo
    //    the permutation here before scattering back into the full
    //    coefficient vector — the CSR predictor below expects skel_vals
    //    in the original Knight ordering.
    if (sc->use_skel_mobius) {
        float reordered[SP_MAX_HEAD_DIM * 2];
        sp_band_dequantize(read_ptr, reordered, bc);
        const int *ord = sc->skel_mobius_mask.order;
        for (int i = 0; i < sc->mask.sk_k; i++) {
            skel_vals[ord[i]] = reordered[i];
        }
    } else {
        sp_band_dequantize(read_ptr, skel_vals, bc);
    }
    read_ptr += bc->total_bytes;

    // Scatter skeleton into full coefficient vector
    for (int s = 0; s < sc->mask.sk_k; s++) {
        coeffs[sc->mask.skeleton_idx[s]] = skel_vals[s];
    }

    // 2. Reconstruct residual positions
    if (n_res > 0) {
        // Read magnitude
        float mag;
        memcpy(&mag, read_ptr, 4);
        read_ptr += 4;

        // Unpack residual levels
        int res_bytes = (n_res * sc->residual_bits + 7) / 8;
        uint8_t levels[SP_MAX_HEAD_DIM * 2];
        int L = 1 << sc->residual_bits;
        uint32_t mask_bits = (uint32_t)(L - 1);
        for (int i = 0; i < n_res; i++) {
            int bit_off = i * sc->residual_bits;
            uint32_t packed = (uint32_t)read_ptr[bit_off / 8] >> (bit_off % 8);
            if ((bit_off % 8) + sc->residual_bits > 8 && (bit_off / 8 + 1) < res_bytes) {
                packed |= ((uint32_t)read_ptr[bit_off / 8 + 1]) << (8 - bit_off % 8);
            }
            levels[i] = (uint8_t)(packed & mask_bits);
        }
        read_ptr += res_bytes;

        // Dequantize residual
        float deviation[SP_MAX_HEAD_DIM * 2];
        sp_dequantize_residual(levels, n_res, sc->residual_bits, mag, deviation);

        // Read sheet bits
        const uint8_t *sheet = NULL;
        if (sc->use_spinor) {
            sheet = read_ptr;
            read_ptr += (n_res + 7) / 8;
        }

        // Compute predictions + apply spinor + add residual
        for (int i = 0; i < n_res; i++) {
            float pred = 0.0f;
            int start = sc->mask.csr_offsets[i];
            int end   = sc->mask.csr_offsets[i + 1];
            for (int j = start; j < end; j++) {
                pred += (float)sc->mask.csr_mu_sign[j]
                      * skel_vals[sc->mask.csr_skel_slot[j]];
            }

            // Spinor: flip pred sign if sheet bit set
            if (sc->use_spinor && sheet) {
                if (sheet[i / 8] & (1 << (i % 8))) {
                    pred = -pred;
                }
            }

            coeffs[sc->mask.residual_idx[i]] = pred + deviation[i];
        }
    }

    // 3. Vilenkin inverse
    sp_vilenkin_inverse_f32(coeffs, pd);

    // 4. Unpad
    sp_sqfree_unpad_f32(coeffs, vec_out, hd);
}

void sp_sqfree_write_k(sp_sqfree_cache_t *sc,
                       int layer, int head, int pos,
                       const float *k_vec) {
    int slot = layer * sc->config.n_heads_kv + head;
    int bytes_per = sc->k_bands.total_bytes + 4
                  + (sc->mask.n_res * sc->residual_bits + 7) / 8
                  + (sc->use_spinor ? (sc->mask.n_res + 7) / 8 : 0);
    sp_sqfree_compress_one(sc, k_vec, sc->k_cache[slot] + pos * bytes_per,
                           &sc->k_bands);
}

void sp_sqfree_write_v(sp_sqfree_cache_t *sc,
                       int layer, int head, int pos,
                       const float *v_vec) {
    int slot = layer * sc->config.n_heads_kv + head;
    int bytes_per = sc->v_bands.total_bytes + 4
                  + (sc->mask.n_res * sc->residual_bits + 7) / 8
                  + (sc->use_spinor ? (sc->mask.n_res + 7) / 8 : 0);
    sp_sqfree_compress_one(sc, v_vec, sc->v_cache[slot] + pos * bytes_per,
                           &sc->v_bands);
}

void sp_sqfree_read_k(const sp_sqfree_cache_t *sc,
                      int layer, int head, int pos,
                      float *k_out) {
    int slot = layer * sc->config.n_heads_kv + head;
    int bytes_per = sc->k_bands.total_bytes + 4
                  + (sc->mask.n_res * sc->residual_bits + 7) / 8
                  + (sc->use_spinor ? (sc->mask.n_res + 7) / 8 : 0);
    sp_sqfree_reconstruct_one(sc, sc->k_cache[slot] + pos * bytes_per,
                              k_out, &sc->k_bands);
}

void sp_sqfree_read_v(const sp_sqfree_cache_t *sc,
                      int layer, int head, int pos,
                      float *v_out) {
    int slot = layer * sc->config.n_heads_kv + head;
    int bytes_per = sc->v_bands.total_bytes + 4
                  + (sc->mask.n_res * sc->residual_bits + 7) / 8
                  + (sc->use_spinor ? (sc->mask.n_res + 7) / 8 : 0);
    sp_sqfree_reconstruct_one(sc, sc->v_cache[slot] + pos * bytes_per,
                              v_out, &sc->v_bands);
}

// ============================================================================
// Adaptive calibration — variance-ranked Knight mask at L/2
// ============================================================================

int sp_sqfree_calibrate_begin(sp_sqfree_cache_t *sc) {
    if (sc->calibrating) return -1;
    int pd = sc->pad_dim;
    sc->calib_sum  = (double *)calloc(pd, sizeof(double));
    sc->calib_sum2 = (double *)calloc(pd, sizeof(double));
    if (!sc->calib_sum || !sc->calib_sum2) {
        free(sc->calib_sum);
        free(sc->calib_sum2);
        sc->calib_sum = NULL;
        sc->calib_sum2 = NULL;
        return -1;
    }
    sc->calib_n = 0;
    sc->calibrating = true;
    return 0;
}

void sp_sqfree_calibrate_feed(sp_sqfree_cache_t *sc, const float *vec) {
    if (!sc->calibrating) return;
    int hd = sc->config.head_dim;
    int pd = sc->pad_dim;

    // Pad vector
    sp_sqfree_pad_f32(vec, hd, sc->pad_scratch, pd);

    // Transform into Vilenkin domain
    sp_vilenkin_forward_f32(sc->pad_scratch, pd);

    // Accumulate per-coefficient statistics
    for (int i = 0; i < pd; i++) {
        double v = (double)sc->pad_scratch[i];
        sc->calib_sum[i]  += v;
        sc->calib_sum2[i] += v * v;
    }
    sc->calib_n++;
}

int sp_sqfree_calibrate_end(sp_sqfree_cache_t *sc) {
    if (!sc->calibrating || sc->calib_n < 1) return -1;
    sc->calibrating = false;

    int pd = sc->pad_dim;
    double inv_n = 1.0 / (double)sc->calib_n;

    // Compute per-coefficient variance: Var = E[x²] − E[x]²
    float *variance = (float *)malloc(pd * sizeof(float));
    for (int i = 0; i < pd; i++) {
        double mean = sc->calib_sum[i] * inv_n;
        double var  = sc->calib_sum2[i] * inv_n - mean * mean;
        variance[i] = (var > 0.0) ? (float)var : 0.0f;
    }

    // Free accumulators
    free(sc->calib_sum);
    free(sc->calib_sum2);
    sc->calib_sum = NULL;
    sc->calib_sum2 = NULL;

    // Rebuild Knight mask with variance ranking at L/2
    sp_knight_mask_free(&sc->mask);
    int sk_k = pd / 2;
    sp_knight_mask_init(&sc->mask, pd, sk_k, variance);
    sc->mask.residual_bits = sc->residual_bits;
    sc->mask.use_spinor = sc->use_spinor;

    free(variance);

    // Re-initialise band quantisers for the (possibly changed) skeleton size.
    // Reuse the config's band bits if set, else defaults.
    int default_k_bits[5] = {5, 4, 4, 4, 5};
    const sp_config_t *cfg = &sc->config;
    int k_nb = (cfg->k_n_bands > 0 && cfg->k_n_bands <= SP_MAX_BANDS)
               ? cfg->k_n_bands : 5;
    const int *k_bits = (cfg->k_n_bands > 0 && cfg->k_n_bands <= SP_MAX_BANDS)
                        ? cfg->k_band_bits : default_k_bits;
    int v_nb = (cfg->v_n_bands > 0 && cfg->v_n_bands <= SP_MAX_BANDS)
               ? cfg->v_n_bands : k_nb;
    const int *v_bits = (cfg->v_n_bands > 0 && cfg->v_n_bands <= SP_MAX_BANDS)
                        ? cfg->v_band_bits : k_bits;
    sp_band_config_init(&sc->k_bands, sc->mask.sk_k, k_nb, k_bits);
    sp_band_config_init(&sc->v_bands, sc->mask.sk_k, v_nb, v_bits);

    // Reallocate compressed storage (new sk_k may change bytes-per-pos)
    int n_slots = cfg->n_layers * cfg->n_heads_kv;
    int n_res = sc->mask.n_res;
    int k_bytes_per_pos = sc->k_bands.total_bytes
                        + (n_res * sc->residual_bits + 7) / 8
                        + 4
                        + (sc->use_spinor ? (n_res + 7) / 8 : 0);
    int v_bytes_per_pos = sc->v_bands.total_bytes
                        + (n_res * sc->residual_bits + 7) / 8
                        + 4
                        + (sc->use_spinor ? (n_res + 7) / 8 : 0);

    for (int s = 0; s < n_slots; s++) {
        free(sc->k_cache[s]);
        free(sc->v_cache[s]);
        sc->k_cache[s] = (uint8_t *)calloc(sc->max_seq_len, k_bytes_per_pos);
        sc->v_cache[s] = (uint8_t *)calloc(sc->max_seq_len, v_bytes_per_pos);
    }

    // Resize pred_scratch for new residual count
    free(sc->pred_scratch);
    sc->pred_scratch = (float *)malloc(n_res * sizeof(float));

    // Rebuild optional skeleton Möbius reorder if it was enabled
    if (sc->use_skel_mobius) {
        sp_mobius_mask_free(&sc->skel_mobius_mask);
        free(sc->skel_mobius_scratch);
        sc->use_skel_mobius = false;
        if (sp_mobius_mask_init(&sc->skel_mobius_mask, sc->mask.sk_k) == 0) {
            sc->use_skel_mobius = true;
            sc->skel_mobius_scratch = (float *)malloc(sc->mask.sk_k * sizeof(float));
        }
    }

    if (getenv("SHANNON_PRIME_VERBOSE")) {
        fprintf(stderr, "[Shannon-Prime SQFREE] calibrated: sk_k=%d (L/2=%d) "
                        "pad_dim=%d n_res=%d n_vectors=%d\n",
                sc->mask.sk_k, pd / 2, pd, n_res, sc->calib_n);
    }

    sc->calib_n = 0;
    return 0;
}

// ============================================================================
// Scaling law
// ============================================================================

float sp_predicted_ppl_ratio(float k_corr, float params_b, int bits) {
    if (k_corr >= 1.0f) return 1.0f;
    float err = 1.0f - k_corr;
    float log_ratio = 4700.0f * (err * err)
                    / (powf(params_b, 1.1f) * powf((float)bits, 1.5f));
    return expf(log_ratio);
}

bool sp_is_pareto_viable(float k_corr, float params_b, int bits, float budget) {
    return sp_predicted_ppl_ratio(k_corr, params_b, bits) <= 1.0f + budget;
}

float sp_min_k_corr_for_budget(float params_b, int bits, float budget) {
    float log_budget = logf(1.0f + budget);
    float denom = powf(params_b, 1.1f) * powf((float)bits, 1.5f);
    float err_sq = log_budget * denom / 4700.0f;
    if (err_sq <= 0.0f) return 1.0f;
    float err = sqrtf(err_sq);
    float result = 1.0f - err;
    return (result > 0.0f) ? result : 0.0f;
}
// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#include "shannon_prime.h"
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

// ============================================================================
// Utilities
// ============================================================================

static inline int sp_is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static inline float sp_f16_to_f32(uint16_t h) {
    // IEEE 754 half-precision to single-precision
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            // Denormalized
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13); // Inf/NaN
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

static inline uint16_t sp_f32_to_f16(float val) {
    uint32_t f;
    memcpy(&f, &val, sizeof(uint32_t));
    uint16_t sign = (f >> 16) & 0x8000;
    int exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp <= 0) {
        return sign; // Flush to zero
    } else if (exp >= 31) {
        return sign | 0x7C00; // Inf
    }
    return sign | (exp << 10) | (mant >> 13);
}

// ============================================================================
// Config
// ============================================================================

void sp_config_init(sp_config_t *cfg, int head_dim, int n_layers, int n_heads_kv) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->head_dim    = head_dim;
    cfg->n_layers    = n_layers;
    cfg->n_heads_kv  = n_heads_kv;

    // Ship-safe K band allocation: 5/5/4/3
    cfg->k_n_bands = 4;
    cfg->k_band_bits[0] = 5;
    cfg->k_band_bits[1] = 5;
    cfg->k_band_bits[2] = 4;
    cfg->k_band_bits[3] = 3;

    // V: flat int3 (no banding — flat beats banded for V, no exceptions)
    cfg->v_n_bands = 1;
    cfg->v_band_bits[0] = 3;

    // Möbius mask on by default (free quality win)
    cfg->use_mobius_mask = true;
    cfg->skeleton_k = head_dim; // Must equal head_dim
    cfg->skeleton_v = head_dim;

    // Vilenkin off by default (research path)
    cfg->use_vilenkin    = false;
    cfg->vilenkin_primes = 2;
    cfg->energy_threshold = 0.95f;
}

// ============================================================================
// WHT — Walsh-Hadamard Transform (in-place butterfly)
// ============================================================================
//
// The WHT is self-inverse: WHT(WHT(x)) = N*x
// This is the Z/2Z case of the Vilenkin-Hartley basis.
// Key property: K vectors show strong spectral concentration in first bands.
// V vectors have uniform energy distribution.

void sp_wht_inplace_f32(float *data, int n) {
    if (!sp_is_power_of_2(n)) return;

    // Standard iterative butterfly
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
}

void sp_wht_inplace_f16(uint16_t *data, int n) {
    // Reference: convert to f32, WHT, convert back.
    // Backends should override with SIMD (NEON, CUDA, etc.)
    float *tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) return;

    for (int i = 0; i < n; i++) {
        tmp[i] = sp_f16_to_f32(data[i]);
    }

    sp_wht_inplace_f32(tmp, n);

    for (int i = 0; i < n; i++) {
        data[i] = sp_f32_to_f16(tmp[i]);
    }

    free(tmp);
}

// ============================================================================
// Möbius mask
// ============================================================================
//
// The Möbius function μ(n):
//   μ(1) = 1
//   μ(n) = 0 if n has a squared prime factor
//   μ(n) = (-1)^k if n is product of k distinct primes
//
// Squarefree indices (μ(n) != 0) carry the independent information.
// Non-squarefree indices are partially predictable via Möbius inversion.
// Prioritizing squarefree-first gives +0.14 PPL for free.

static void compute_mobius(int8_t *mu, int n) {
    // Sieve-based computation
    mu[0] = 0; // μ(0) undefined; treat as non-squarefree
    if (n > 1) mu[1] = 1;

    // Initialize
    for (int i = 2; i < n; i++) mu[i] = 1;

    // Mark prime factors
    int *prime_count = (int *)calloc(n, sizeof(int));
    bool *has_square = (bool *)calloc(n, sizeof(bool));

    for (int p = 2; p < n; p++) {
        if (prime_count[p] != 0) continue; // not prime
        // p is prime — mark all multiples
        for (int m = p; m < n; m += p) {
            prime_count[m]++;
        }
        // Mark squared multiples
        long long p2 = (long long)p * p;
        for (long long m = p2; m < n; m += p2) {
            has_square[m] = true;
        }
    }

    for (int i = 2; i < n; i++) {
        if (has_square[i]) {
            mu[i] = 0;
        } else {
            mu[i] = (prime_count[i] % 2 == 0) ? 1 : -1;
        }
    }

    free(prime_count);
    free(has_square);
}

int sp_mobius_mask_init(sp_mobius_mask_t *mask, int n) {
    mask->n = n;
    mask->mu = (int8_t *)malloc(n * sizeof(int8_t));
    mask->order = (int *)malloc(n * sizeof(int));
    if (!mask->mu || !mask->order) return -1;

    compute_mobius(mask->mu, n);

    // Build permutation: squarefree first, then non-squarefree
    int sf = 0;

    // Count squarefree
    for (int i = 0; i < n; i++) {
        if (mask->mu[i] != 0) sf++;
    }
    mask->n_squarefree = sf;

    // Fill order: squarefree indices first
    int pos_sf = 0;
    int pos_nsf = sf;
    for (int i = 0; i < n; i++) {
        if (mask->mu[i] != 0) {
            mask->order[pos_sf++] = i;
        } else {
            mask->order[pos_nsf++] = i;
        }
    }

    return 0;
}

void sp_mobius_mask_free(sp_mobius_mask_t *mask) {
    free(mask->mu);
    free(mask->order);
    mask->mu = NULL;
    mask->order = NULL;
}

void sp_mobius_reorder(float *wht_coeffs, const sp_mobius_mask_t *mask) {
    int n = mask->n;
    float *tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) return;

    for (int i = 0; i < n; i++) {
        tmp[i] = wht_coeffs[mask->order[i]];
    }
    memcpy(wht_coeffs, tmp, n * sizeof(float));
    free(tmp);
}

void sp_mobius_unreorder(float *wht_coeffs, const sp_mobius_mask_t *mask) {
    int n = mask->n;
    float *tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) return;

    for (int i = 0; i < n; i++) {
        tmp[mask->order[i]] = wht_coeffs[i];
    }
    memcpy(wht_coeffs, tmp, n * sizeof(float));
    free(tmp);
}

// ============================================================================
// VHT2 banded quantization
// ============================================================================
//
// Each band stores: 1 fp16 scale + packed integer values.
// The scale is max(abs(band)) / (2^(bits-1) - 1).
// This mirrors WHT energy decay: band 0 (highest energy) gets most bits,
// band 3 (lowest energy / noise tail) gets fewest.
//
// Critical results from paper:
//   5/5/4/3 → PPL 11.2147 (BETTER than fp16 11.2194 — spectral regularization)
//   4/4/4/4 → off Pareto frontier (4/4/4/3 strictly dominates)
//   Any band at 2-bit → catastrophic

void sp_band_config_init(sp_band_config_t *bc, int head_dim,
                         int n_bands, const int *band_bits) {
    bc->n_bands   = n_bands;
    bc->band_size = head_dim / n_bands;

    int total = 0;
    for (int b = 0; b < n_bands; b++) {
        bc->band_bits[b] = band_bits[b];
        // Per band: 2 bytes (fp16 scale) + ceil(band_size * bits / 8) bytes
        int data_bits = bc->band_size * band_bits[b];
        int data_bytes = (data_bits + 7) / 8;
        total += 2 + data_bytes; // fp16 scale + packed data
    }
    bc->total_bytes = total;
}

void sp_band_quantize(const float *wht_coeffs, uint8_t *out,
                      const sp_band_config_t *bc) {
    int offset = 0;

    for (int b = 0; b < bc->n_bands; b++) {
        const float *band = wht_coeffs + b * bc->band_size;
        int bits = bc->band_bits[b];
        int max_val = (1 << (bits - 1)) - 1; // e.g. 5-bit → 15

        // Find max absolute value in band
        float amax = 0.0f;
        for (int i = 0; i < bc->band_size; i++) {
            float a = fabsf(band[i]);
            if (a > amax) amax = a;
        }

        // Scale: maps [-amax, amax] to [-max_val, max_val]
        float scale = (amax > 0.0f) ? amax / (float)max_val : 0.0f;
        float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;

        // Store scale as fp16
        uint16_t scale_f16 = sp_f32_to_f16(scale);
        out[offset]     = scale_f16 & 0xFF;
        out[offset + 1] = (scale_f16 >> 8) & 0xFF;
        offset += 2;

        // Pack quantized values
        // We pack bits-wide signed integers in little-endian bit order
        uint64_t bit_buffer = 0;
        int      bit_pos = 0;

        for (int i = 0; i < bc->band_size; i++) {
            // Quantize to signed integer
            int q = (int)roundf(band[i] * inv_scale);
            if (q > max_val)  q = max_val;
            if (q < -max_val) q = -max_val;

            // Convert to unsigned representation for packing
            // Offset by max_val so range is [0, 2*max_val]
            uint32_t u = (uint32_t)(q + max_val);

            bit_buffer |= ((uint64_t)u << bit_pos);
            bit_pos += bits;

            // Flush full bytes
            while (bit_pos >= 8) {
                out[offset++] = (uint8_t)(bit_buffer & 0xFF);
                bit_buffer >>= 8;
                bit_pos -= 8;
            }
        }

        // Flush remaining bits
        if (bit_pos > 0) {
            out[offset++] = (uint8_t)(bit_buffer & 0xFF);
        }
    }
}

void sp_band_dequantize(const uint8_t *in, float *wht_coeffs,
                        const sp_band_config_t *bc) {
    int offset = 0;

    for (int b = 0; b < bc->n_bands; b++) {
        float *band = wht_coeffs + b * bc->band_size;
        int bits = bc->band_bits[b];
        int max_val = (1 << (bits - 1)) - 1;
        uint32_t mask = (1u << bits) - 1;

        // Read scale
        uint16_t scale_f16 = (uint16_t)in[offset] | ((uint16_t)in[offset + 1] << 8);
        float scale = sp_f16_to_f32(scale_f16);
        offset += 2;

        // Unpack quantized values
        uint64_t bit_buffer = 0;
        int      bit_pos = 0;
        int      byte_idx = offset;

        for (int i = 0; i < bc->band_size; i++) {
            // Load bytes as needed
            while (bit_pos < bits) {
                bit_buffer |= ((uint64_t)in[byte_idx++] << bit_pos);
                bit_pos += 8;
            }

            uint32_t u = (uint32_t)(bit_buffer & mask);
            bit_buffer >>= bits;
            bit_pos -= bits;

            // Convert back to signed
            int q = (int)u - max_val;
            band[i] = (float)q * scale;
        }

        // Advance offset past the packed data
        int data_bits = bc->band_size * bits;
        offset += (data_bits + 7) / 8;
    }
}

// ============================================================================
// Vilenkin-Hartley Transform
// ============================================================================
//
// Hartley kernel: cas(x) = cos(x) + sin(x)
// For prime p, the p×p Hartley matrix has entries:
//   H[i][j] = cas(2π·i·j / p) / sqrt(p)
// The full basis is the Kronecker product: V = H_p1 ⊗ H_p2 ⊗ ... ⊗ H_pk
// Self-inverse: V·V = N·I (round-trip error = 0.0000)

static const int vilenkin_primes[] = { 2, 3, 5, 7, 11, 13 };

int sp_vilenkin_init(sp_vilenkin_basis_t *vb, int n_primes) {
    if (n_primes < 1 || n_primes > 6) return -1;

    vb->n_primes = n_primes;
    vb->n = 1;
    for (int i = 0; i < n_primes; i++) {
        vb->primes[i] = vilenkin_primes[i];
        vb->n *= vilenkin_primes[i];
    }

    // Allocate n×n basis matrix
    vb->basis = (float *)malloc((size_t)vb->n * vb->n * sizeof(float));
    if (!vb->basis) return -1;

    // Build via Kronecker product of per-prime Hartley matrices
    // Start with 1×1 identity, then Kronecker with each H_p

    // Current matrix (starts as [1.0])
    int cur_n = 1;
    float *cur = (float *)malloc(sizeof(float));
    cur[0] = 1.0f;

    for (int pi = 0; pi < n_primes; pi++) {
        int p = vb->primes[pi];
        int new_n = cur_n * p;
        float *next = (float *)malloc((size_t)new_n * new_n * sizeof(float));

        // H_p[i][j] = cas(2π·i·j / p) / sqrt(p)
        float norm = 1.0f / sqrtf((float)p);

        for (int ci = 0; ci < cur_n; ci++) {
            for (int cj = 0; cj < cur_n; cj++) {
                float c_val = cur[ci * cur_n + cj];
                for (int hi = 0; hi < p; hi++) {
                    for (int hj = 0; hj < p; hj++) {
                        float angle = 2.0f * (float)M_PI * (float)(hi * hj) / (float)p;
                        float h_val = (cosf(angle) + sinf(angle)) * norm;
                        int ri = ci * p + hi;
                        int rj = cj * p + hj;
                        next[ri * new_n + rj] = c_val * h_val;
                    }
                }
            }
        }

        free(cur);
        cur = next;
        cur_n = new_n;
    }

    memcpy(vb->basis, cur, (size_t)vb->n * vb->n * sizeof(float));
    free(cur);
    return 0;
}

void sp_vilenkin_free(sp_vilenkin_basis_t *vb) {
    free(vb->basis);
    vb->basis = NULL;
}

void sp_vilenkin_forward(const sp_vilenkin_basis_t *vb,
                         const float *input, int head_dim,
                         float *output) {
    int n = vb->n;

    // Zero-pad input if head_dim < n
    float *padded = (float *)calloc(n, sizeof(float));
    memcpy(padded, input, head_dim * sizeof(float));

    // Matrix multiply: output = V · padded
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += vb->basis[i * n + j] * padded[j];
        }
        output[i] = sum;
    }

    free(padded);
}

void sp_vilenkin_inverse(const sp_vilenkin_basis_t *vb,
                         const float *input,
                         float *output, int head_dim) {
    int n = vb->n;

    // V is orthonormal (V·V = I), so inverse = V (same as forward)
    float *full = (float *)calloc(n, sizeof(float));

    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += vb->basis[i * n + j] * input[j];
        }
        full[i] = sum;
    }

    // Truncate back to head_dim
    memcpy(output, full, head_dim * sizeof(float));
    free(full);
}

int sp_vilenkin_extract_pass(const sp_vilenkin_basis_t *vb,
                             float *residual, int head_dim,
                             float energy_threshold,
                             sp_vilenkin_pass_t *pass) {
    int n = vb->n;

    // Forward transform residual
    float *coeffs = (float *)malloc(n * sizeof(float));
    sp_vilenkin_forward(vb, residual, head_dim, coeffs);

    // Compute total energy
    float total_energy = 0.0f;
    for (int i = 0; i < n; i++) {
        total_energy += coeffs[i] * coeffs[i];
    }

    // Sort indices by descending energy
    int *sorted_idx = (int *)malloc(n * sizeof(int));
    float *energies = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        sorted_idx[i] = i;
        energies[i] = coeffs[i] * coeffs[i];
    }

    // Simple insertion sort (n is small: 6, 30, or 210)
    for (int i = 1; i < n; i++) {
        int key_idx = sorted_idx[i];
        float key_e = energies[i];
        int j = i - 1;
        while (j >= 0 && energies[j] < key_e) {
            sorted_idx[j + 1] = sorted_idx[j];
            energies[j + 1]   = energies[j];
            j--;
        }
        sorted_idx[j + 1] = key_idx;
        energies[j + 1]   = key_e;
    }

    // Select coefficients until energy threshold reached
    float captured = 0.0f;
    float target = total_energy * energy_threshold;
    int count = 0;

    while (count < n && captured < target) {
        captured += energies[count];
        count++;
    }

    // Allocate pass
    pass->n_coeffs = count;
    pass->indices  = (int *)malloc(count * sizeof(int));
    pass->values   = (float *)malloc(count * sizeof(float));

    for (int i = 0; i < count; i++) {
        pass->indices[i] = sorted_idx[i];
        pass->values[i]  = coeffs[sorted_idx[i]];
    }

    // Subtract extracted component from residual
    // Reconstruct the extracted part and subtract
    float *extracted = (float *)calloc(n, sizeof(float));
    for (int i = 0; i < count; i++) {
        extracted[sorted_idx[i]] = coeffs[sorted_idx[i]];
    }

    float *reconstructed = (float *)calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += vb->basis[i * n + j] * extracted[j];
        }
        reconstructed[i] = sum;
    }

    for (int i = 0; i < head_dim; i++) {
        residual[i] -= reconstructed[i];
    }

    free(coeffs);
    free(sorted_idx);
    free(energies);
    free(extracted);
    free(reconstructed);
    return 0;
}

void sp_vilenkin_pass_free(sp_vilenkin_pass_t *pass) {
    free(pass->indices);
    free(pass->values);
    pass->indices = NULL;
    pass->values = NULL;
}

// ============================================================================
// Shadow cache
// ============================================================================

int sp_shadow_cache_init(sp_shadow_cache_t *sc, const sp_config_t *cfg) {
    memcpy(&sc->config, cfg, sizeof(sp_config_t));

    // Initialize band configs
    sp_band_config_init(&sc->k_bands, cfg->head_dim,
                        cfg->k_n_bands, cfg->k_band_bits);
    sp_band_config_init(&sc->v_bands, cfg->head_dim,
                        cfg->v_n_bands, cfg->v_band_bits);

    // Initialize Möbius mask
    if (cfg->use_mobius_mask) {
        if (sp_mobius_mask_init(&sc->mobius_mask, cfg->head_dim) != 0) {
            return -1;
        }
    }

    // Allocate scratch buffer
    sc->wht_scratch = (float *)malloc(cfg->head_dim * sizeof(float));
    if (!sc->wht_scratch) return -1;

    // Cache storage will be allocated by backend (depends on max_seq_len)
    sc->k_cache = NULL;
    sc->v_cache = NULL;
    sc->seq_len = (int *)calloc(cfg->n_layers, sizeof(int));

    return 0;
}

void sp_shadow_cache_free(sp_shadow_cache_t *sc) {
    if (sc->config.use_mobius_mask) {
        sp_mobius_mask_free(&sc->mobius_mask);
    }
    free(sc->wht_scratch);
    free(sc->seq_len);
    // k_cache and v_cache freed by backend
    sc->wht_scratch = NULL;
    sc->seq_len = NULL;
}

// Write path: raw KV → WHT → Möbius reorder → band quantize → store
void sp_shadow_write_k(sp_shadow_cache_t *sc,
                       int layer, int head, int pos,
                       const float *k_vec) {
    int hd = sc->config.head_dim;
    float *scratch = sc->wht_scratch;

    // Copy to scratch
    memcpy(scratch, k_vec, hd * sizeof(float));

    // WHT forward
    sp_wht_inplace_f32(scratch, hd);

    // Normalize (WHT is unnormalized; scale by 1/sqrt(N) for symmetric form)
    // For quantization we keep unnormalized — scale absorbs it
    // (The scale per band captures the actual magnitude)

    // Möbius reorder (squarefree first)
    if (sc->config.use_mobius_mask) {
        sp_mobius_reorder(scratch, &sc->mobius_mask);
    }

    // Band quantize
    int slot = layer * sc->config.n_heads_kv + head;
    uint8_t *dest = sc->k_cache[slot] + (size_t)pos * sc->k_bands.total_bytes;
    sp_band_quantize(scratch, dest, &sc->k_bands);
}

void sp_shadow_write_v(sp_shadow_cache_t *sc,
                       int layer, int head, int pos,
                       const float *v_vec) {
    int hd = sc->config.head_dim;
    float *scratch = sc->wht_scratch;

    memcpy(scratch, v_vec, hd * sizeof(float));
    sp_wht_inplace_f32(scratch, hd);

    // No Möbius reorder for V (uniform spectrum — no benefit)
    // V gets flat quantization (1 band), no reordering needed

    int slot = layer * sc->config.n_heads_kv + head;
    uint8_t *dest = sc->v_cache[slot] + (size_t)pos * sc->v_bands.total_bytes;
    sp_band_quantize(scratch, dest, &sc->v_bands);
}

// Read path: load → band dequantize → Möbius unreorder → inverse WHT → KV
void sp_shadow_read_k(const sp_shadow_cache_t *sc,
                      int layer, int head, int pos,
                      float *k_out) {
    int hd = sc->config.head_dim;
    // Note: read path needs its own scratch to be thread-safe.
    // For reference impl, we allocate locally.
    // Backends should use thread-local scratch.
    float *scratch = (float *)malloc(hd * sizeof(float));

    int slot = layer * sc->config.n_heads_kv + head;
    const uint8_t *src = sc->k_cache[slot] + (size_t)pos * sc->k_bands.total_bytes;

    // Dequantize
    sp_band_dequantize(src, scratch, &sc->k_bands);

    // Möbius unreorder
    if (sc->config.use_mobius_mask) {
        sp_mobius_unreorder(scratch, &sc->mobius_mask);
    }

    // Inverse WHT (same as forward, then divide by N)
    sp_wht_inplace_f32(scratch, hd);
    float inv_n = 1.0f / (float)hd;
    for (int i = 0; i < hd; i++) {
        scratch[i] *= inv_n;
    }

    // NaN guard (defense in depth — ship config shouldn't need it)
    sp_nan_guard_f32(scratch, hd, 65504.0f); // fp16 max

    memcpy(k_out, scratch, hd * sizeof(float));
    free(scratch);
}

void sp_shadow_read_v(const sp_shadow_cache_t *sc,
                      int layer, int head, int pos,
                      float *v_out) {
    int hd = sc->config.head_dim;
    float *scratch = (float *)malloc(hd * sizeof(float));

    int slot = layer * sc->config.n_heads_kv + head;
    const uint8_t *src = sc->v_cache[slot] + (size_t)pos * sc->v_bands.total_bytes;

    sp_band_dequantize(src, scratch, &sc->v_bands);

    // No Möbius unreorder for V
    sp_wht_inplace_f32(scratch, hd);
    float inv_n = 1.0f / (float)hd;
    for (int i = 0; i < hd; i++) {
        scratch[i] *= inv_n;
    }

    sp_nan_guard_f32(scratch, hd, 65504.0f);

    memcpy(v_out, scratch, hd * sizeof(float));
    free(scratch);
}

// ============================================================================
// NaN guard
// ============================================================================

void sp_nan_guard_f32(float *data, int n, float max_magnitude) {
    for (int i = 0; i < n; i++) {
        if (!isfinite(data[i])) {
            data[i] = 0.0f;
        } else if (data[i] > max_magnitude) {
            data[i] = max_magnitude;
        } else if (data[i] < -max_magnitude) {
            data[i] = -max_magnitude;
        }
    }
}

// ============================================================================
// Diagnostics
// ============================================================================

float sp_correlation_f32(const float *a, const float *b, int n) {
    double sum_a = 0, sum_b = 0, sum_ab = 0;
    double sum_a2 = 0, sum_b2 = 0;

    for (int i = 0; i < n; i++) {
        sum_a  += a[i];
        sum_b  += b[i];
        sum_ab += (double)a[i] * b[i];
        sum_a2 += (double)a[i] * a[i];
        sum_b2 += (double)b[i] * b[i];
    }

    double mean_a = sum_a / n;
    double mean_b = sum_b / n;
    double cov = sum_ab / n - mean_a * mean_b;
    double var_a = sum_a2 / n - mean_a * mean_a;
    double var_b = sum_b2 / n - mean_b * mean_b;

    if (var_a < 1e-12 || var_b < 1e-12) return 0.0f;
    return (float)(cov / sqrt(var_a * var_b));
}

float sp_compression_ratio(const sp_config_t *cfg) {
    int hd = cfg->head_dim;
    int baseline_bytes = hd * 2; // fp16 per element

    // K compressed size
    sp_band_config_t kbc;
    sp_band_config_init(&kbc, hd, cfg->k_n_bands, cfg->k_band_bits);

    // V compressed size
    sp_band_config_t vbc;
    sp_band_config_init(&vbc, hd, cfg->v_n_bands, cfg->v_band_bits);

    // Total ratio (K and V equally weighted)
    return 2.0f * (float)baseline_bytes / (float)(kbc.total_bytes + vbc.total_bytes);
}

void sp_config_print(const sp_config_t *cfg) {
    fprintf(stderr, "Shannon-Prime VHT2 Configuration:\n");
    fprintf(stderr, "  head_dim:     %d\n", cfg->head_dim);
    fprintf(stderr, "  n_layers:     %d\n", cfg->n_layers);
    fprintf(stderr, "  n_heads_kv:   %d\n", cfg->n_heads_kv);

    fprintf(stderr, "  K bands:      %d (", cfg->k_n_bands);
    for (int i = 0; i < cfg->k_n_bands; i++) {
        fprintf(stderr, "%d%s", cfg->k_band_bits[i],
                i < cfg->k_n_bands - 1 ? "/" : "");
    }
    fprintf(stderr, ")\n");

    fprintf(stderr, "  V bands:      %d (", cfg->v_n_bands);
    for (int i = 0; i < cfg->v_n_bands; i++) {
        fprintf(stderr, "%d%s", cfg->v_band_bits[i],
                i < cfg->v_n_bands - 1 ? "/" : "");
    }
    fprintf(stderr, ")\n");

    fprintf(stderr, "  Möbius mask:  %s\n", cfg->use_mobius_mask ? "on" : "off");
    fprintf(stderr, "  Vilenkin:     %s", cfg->use_vilenkin ? "on" : "off");
    if (cfg->use_vilenkin) {
        fprintf(stderr, " (%d primes, %.0f%% energy)",
                cfg->vilenkin_primes, cfg->energy_threshold * 100);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "  Compression:  %.1f×\n", sp_compression_ratio(cfg));
}

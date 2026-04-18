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
// VHT2 — Vilenkin-Hartley Transform (the single transform)
// ============================================================================
//
// Staged application of p × p Hartley kernels cas(2πkj/p)/√p for each prime
// factor p of the vector length n (factors in {2,3,5,7,11}). Each stage is
// orthonormal, so the whole transform is self-inverse: VHT2(VHT2(x)) = x.
//
// At n = 2^k this collapses to the Walsh-Hadamard butterfly scaled by 1/√2
// per stage — the spectral structure is that of the p=2 Hartley butterfly but
// there is no 1/N to undo on the inverse.

#define SP_VHT2_MAX_P 11

static const int _sp_vht2_primes[] = {2, 3, 5, 7, 11};
static const int _sp_vht2_n_primes = 5;

// Hartley kernels for each supported prime, lazily initialised on first use.
static float _sp_H2[2][2];
static float _sp_H3[3][3];
static float _sp_H5[5][5];
static float _sp_H7[7][7];
static float _sp_H11[11][11];
static int   _sp_vht2_initialised = 0;

static void _sp_vht2_init_kernels(void) {
    if (_sp_vht2_initialised) return;
    float *kernels[5] = {
        &_sp_H2[0][0], &_sp_H3[0][0], &_sp_H5[0][0],
        &_sp_H7[0][0], &_sp_H11[0][0]
    };
    for (int pi = 0; pi < _sp_vht2_n_primes; pi++) {
        int p = _sp_vht2_primes[pi];
        float *H = kernels[pi];
        double inv_sqrt_p = 1.0 / sqrt((double)p);
        for (int k = 0; k < p; k++) {
            for (int j = 0; j < p; j++) {
                double angle = 2.0 * M_PI * (double)k * (double)j / (double)p;
                H[k * p + j] = (float)((cos(angle) + sin(angle)) * inv_sqrt_p);
            }
        }
    }
    _sp_vht2_initialised = 1;
}

static const float *_sp_hartley_kernel(int p) {
    switch (p) {
        case 2:  return &_sp_H2[0][0];
        case 3:  return &_sp_H3[0][0];
        case 5:  return &_sp_H5[0][0];
        case 7:  return &_sp_H7[0][0];
        case 11: return &_sp_H11[0][0];
        default: return NULL;
    }
}

void sp_vht2_forward_f32(float *data, int n) {
    if (n <= 0) return;
    _sp_vht2_init_kernels();

    int stride = 1;
    int residue = n;
    for (int pi = 0; pi < _sp_vht2_n_primes; pi++) {
        int p = _sp_vht2_primes[pi];
        while (residue % p == 0) {
            const float *H = _sp_hartley_kernel(p);
            const int block = p * stride;
            for (int i = 0; i < n; i += block) {
                for (int j = 0; j < stride; j++) {
                    float in[SP_VHT2_MAX_P];
                    for (int k = 0; k < p; k++) {
                        in[k] = data[i + k * stride + j];
                    }
                    for (int k = 0; k < p; k++) {
                        float sum = 0.0f;
                        const float *row = H + k * p;
                        for (int m = 0; m < p; m++) sum += row[m] * in[m];
                        data[i + k * stride + j] = sum;
                    }
                }
            }
            residue /= p;
            stride  *= p;
        }
    }
    // If residue != 1 here, n had a prime factor > 11 — caller should have
    // padded via sqfree_pad_dim. Leave data unchanged so the failure is loud.
}

void sp_vht2_forward_f16(uint16_t *data, int n) {
    // Reference: float promote, transform, quantise back. Backends with native
    // fp16 support (CUDA, NEON fp16) should override.
    float *tmp = (float *)malloc((size_t)n * sizeof(float));
    if (!tmp) return;
    for (int i = 0; i < n; i++) tmp[i] = sp_f16_to_f32(data[i]);
    sp_vht2_forward_f32(tmp, n);
    for (int i = 0; i < n; i++) data[i] = sp_f32_to_f16(tmp[i]);
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

// Caller-owned-scratch variants. Used on the hot path (shadow cache,
// Adreno backend) to avoid malloc per KV vector.
void sp_mobius_reorder_ex(float *vht2_coeffs, const sp_mobius_mask_t *mask,
                          float *scratch) {
    int n = mask->n;
    for (int i = 0; i < n; i++) {
        scratch[i] = vht2_coeffs[mask->order[i]];
    }
    memcpy(vht2_coeffs, scratch, n * sizeof(float));
}

void sp_mobius_unreorder_ex(float *vht2_coeffs, const sp_mobius_mask_t *mask,
                            float *scratch) {
    int n = mask->n;
    for (int i = 0; i < n; i++) {
        scratch[mask->order[i]] = vht2_coeffs[i];
    }
    memcpy(vht2_coeffs, scratch, n * sizeof(float));
}

// Malloc-owning variants retained for existing callers (tests, research code).
void sp_mobius_reorder(float *vht2_coeffs, const sp_mobius_mask_t *mask) {
    int n = mask->n;
    float *tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) return;
    sp_mobius_reorder_ex(vht2_coeffs, mask, tmp);
    free(tmp);
}

void sp_mobius_unreorder(float *vht2_coeffs, const sp_mobius_mask_t *mask) {
    int n = mask->n;
    float *tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) return;
    sp_mobius_unreorder_ex(vht2_coeffs, mask, tmp);
    free(tmp);
}

// ============================================================================
// VHT2 banded quantization
// ============================================================================
//
// Each band stores: 1 fp16 scale + packed integer values.
// The scale is max(abs(band)) / (2^(bits-1) - 1).
// This mirrors VHT2 energy decay: band 0 (highest energy) gets most bits,
// band 3 (lowest energy / noise tail) gets fewest.
//
// Critical results from paper:
//   5/5/4/3 → PPL 11.2147 (BETTER than fp16 11.2194 — spectral regularization)
//   4/4/4/4 → off Pareto frontier (4/4/4/3 strictly dominates)
//   Any band at 2-bit → catastrophic

void sp_band_config_init(sp_band_config_t *bc, int head_dim,
                         int n_bands, const int *band_bits) {
    bc->n_bands   = n_bands;
    bc->head_dim  = head_dim;
    bc->band_size = head_dim / n_bands;

    int total = 0;
    for (int b = 0; b < n_bands; b++) {
        bc->band_bits[b] = band_bits[b];
        int off, sz;
        sp_band_span(bc, b, &off, &sz);
        // Per band: 2 bytes (fp16 scale) + ceil(sz * bits / 8) bytes
        int data_bits = sz * band_bits[b];
        int data_bytes = (data_bits + 7) / 8;
        total += 2 + data_bytes; // fp16 scale + packed data
    }
    bc->total_bytes = total;
}

void sp_band_quantize(const float *vht2_coeffs, uint8_t *out,
                      const sp_band_config_t *bc) {
    int offset = 0;

    for (int b = 0; b < bc->n_bands; b++) {
        int band_off, band_sz;
        sp_band_span(bc, b, &band_off, &band_sz);
        const float *band = vht2_coeffs + band_off;
        int bits = bc->band_bits[b];
        int max_val = (1 << (bits - 1)) - 1; // e.g. 5-bit → 15

        // Find max absolute value in band
        float amax = 0.0f;
        for (int i = 0; i < band_sz; i++) {
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

        for (int i = 0; i < band_sz; i++) {
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

void sp_band_dequantize(const uint8_t *in, float *vht2_coeffs,
                        const sp_band_config_t *bc) {
    int offset = 0;

    for (int b = 0; b < bc->n_bands; b++) {
        int band_off, band_sz;
        sp_band_span(bc, b, &band_off, &band_sz);
        float *band = vht2_coeffs + band_off;
        int bits = bc->band_bits[b];
        int max_val = (1 << (bits - 1)) - 1;
        uint32_t mask = (1u << bits) - 1;

        // Read scale. Sanitise: fp16 round-trip can produce +Inf (amax overflowed
        // the fp16 range on encode) or NaN (corrupted bytes). An Inf or NaN scale
        // would poison every value in this band and then cascade through the
        // inverse VHT2, so we clamp to 0 here — the band decodes as all zeros,
        // which is the same outcome the old blanket NaN guard used to produce,
        // but applied at the root cause rather than the output tail.
        uint16_t scale_f16 = (uint16_t)in[offset] | ((uint16_t)in[offset + 1] << 8);
        float scale = sp_f16_to_f32(scale_f16);
        if (!isfinite(scale)) scale = 0.0f;
        offset += 2;

        // Unpack quantized values
        uint64_t bit_buffer = 0;
        int      bit_pos = 0;
        int      byte_idx = offset;

        for (int i = 0; i < band_sz; i++) {
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
        int data_bits = band_sz * bits;
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
    memset(sc, 0, sizeof(*sc));
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

    // Allocate persistent scratch buffers so the hot path never mallocs.
    // vht2_scratch    : write-path VHT2 buffer
    // mobius_scratch  : Möbius reorder/unreorder tmp (shared between write+read)
    // read_scratch    : read-path VHT2 buffer (independent of write path)
    sc->vht2_scratch    = (float *)malloc(cfg->head_dim * sizeof(float));
    sc->mobius_scratch = (float *)malloc(cfg->head_dim * sizeof(float));
    sc->read_scratch   = (float *)malloc(cfg->head_dim * sizeof(float));
    if (!sc->vht2_scratch || !sc->mobius_scratch || !sc->read_scratch) return -1;

    // Cache storage will be allocated by backend (depends on max_seq_len)
    sc->k_cache = NULL;
    sc->v_cache = NULL;
    sc->seq_len = (int *)calloc(cfg->n_layers, sizeof(int));

    // Variance-ranked reorder: off until calibrated
    sc->use_var_reorder = false;
    sc->var_order = NULL;
    sc->var_unorder = NULL;
    sc->calibrating = false;
    sc->calib_sum = NULL;
    sc->calib_sum2 = NULL;
    sc->calib_n = 0;

    return 0;
}

void sp_shadow_cache_free(sp_shadow_cache_t *sc) {
    if (sc->config.use_mobius_mask) {
        sp_mobius_mask_free(&sc->mobius_mask);
    }
    free(sc->vht2_scratch);
    free(sc->mobius_scratch);
    free(sc->read_scratch);
    free(sc->seq_len);
    free(sc->var_order);
    free(sc->var_unorder);
    free(sc->calib_sum);
    free(sc->calib_sum2);
    // k_cache and v_cache freed by backend
    sc->vht2_scratch    = NULL;
    sc->mobius_scratch = NULL;
    sc->read_scratch   = NULL;
    sc->seq_len        = NULL;
    sc->var_order      = NULL;
    sc->var_unorder    = NULL;
    sc->calib_sum      = NULL;
    sc->calib_sum2     = NULL;
}

// Write path: raw KV → VHT2 → Möbius reorder → band quantize → store
// Hot path: uses persistent sc->vht2_scratch + sc->mobius_scratch. No malloc.
void sp_shadow_write_k(sp_shadow_cache_t *sc,
                       int layer, int head, int pos,
                       const float *k_vec) {
    int hd = sc->config.head_dim;
    float *scratch = sc->vht2_scratch;

    memcpy(scratch, k_vec, hd * sizeof(float));
    sp_vht2_forward_f32(scratch, hd);

    // Reorder: variance-ranked (if calibrated) > Möbius (if enabled) > none
    if (sc->use_var_reorder) {
        // Permute by variance: high-variance → front → high-bit bands
        float *tmp = sc->mobius_scratch;
        for (int i = 0; i < hd; i++) tmp[i] = scratch[sc->var_order[i]];
        memcpy(scratch, tmp, hd * sizeof(float));
    } else if (sc->config.use_mobius_mask) {
        sp_mobius_reorder_ex(scratch, &sc->mobius_mask, sc->mobius_scratch);
    }

    int slot = layer * sc->config.n_heads_kv + head;
    uint8_t *dest = sc->k_cache[slot] + (size_t)pos * sc->k_bands.total_bytes;
    sp_band_quantize(scratch, dest, &sc->k_bands);
}

void sp_shadow_write_v(sp_shadow_cache_t *sc,
                       int layer, int head, int pos,
                       const float *v_vec) {
    int hd = sc->config.head_dim;
    float *scratch = sc->vht2_scratch;

    memcpy(scratch, v_vec, hd * sizeof(float));
    sp_vht2_forward_f32(scratch, hd);

    // No Möbius reorder for V (uniform spectrum — no benefit)
    // V gets flat quantization (1 band), no reordering needed

    int slot = layer * sc->config.n_heads_kv + head;
    uint8_t *dest = sc->v_cache[slot] + (size_t)pos * sc->v_bands.total_bytes;
    sp_band_quantize(scratch, dest, &sc->v_bands);
}

// Read path: load → band dequantize → Möbius unreorder → VHT2 (self-inverse) → KV
// Hot path: uses persistent sc->read_scratch + sc->mobius_scratch. No malloc.
// The `const` on sc is a contract for thread-call-safety, not true immutability
// — we write into sc->read_scratch / sc->mobius_scratch. Callers must serialize.
void sp_shadow_read_k(const sp_shadow_cache_t *sc,
                      int layer, int head, int pos,
                      float *k_out) {
    int hd = sc->config.head_dim;
    float *scratch = sc->read_scratch;

    int slot = layer * sc->config.n_heads_kv + head;
    const uint8_t *src = sc->k_cache[slot] + (size_t)pos * sc->k_bands.total_bytes;

    sp_band_dequantize(src, scratch, &sc->k_bands);

    // Inverse reorder: variance-ranked (if calibrated) > Möbius > none
    if (sc->use_var_reorder) {
        float *tmp = sc->mobius_scratch;
        for (int i = 0; i < hd; i++) tmp[sc->var_order[i]] = scratch[i];
        memcpy(scratch, tmp, hd * sizeof(float));
    } else if (sc->config.use_mobius_mask) {
        sp_mobius_unreorder_ex(scratch, &sc->mobius_mask, sc->mobius_scratch);
    }

    // Inverse transform: VHT2 is self-inverse (1/√p per stage absorbs the
    // 1/N the old unnormalised p=2 butterfly required).
    sp_vht2_forward_f32(scratch, hd);

    memcpy(k_out, scratch, hd * sizeof(float));
}

void sp_shadow_read_v(const sp_shadow_cache_t *sc,
                      int layer, int head, int pos,
                      float *v_out) {
    int hd = sc->config.head_dim;
    float *scratch = sc->read_scratch;

    int slot = layer * sc->config.n_heads_kv + head;
    const uint8_t *src = sc->v_cache[slot] + (size_t)pos * sc->v_bands.total_bytes;

    sp_band_dequantize(src, scratch, &sc->v_bands);

    // Inverse transform: VHT2 is self-inverse.
    sp_vht2_forward_f32(scratch, hd);

    memcpy(v_out, scratch, hd * sizeof(float));
}

// Batch variants. Tight loop reusing the persistent scratch. Zero mallocs
// per batch, amortizes the "copy → transform → store/load → transform
// → copy" pipeline setup across n_pos vectors.
void sp_shadow_write_k_batch(sp_shadow_cache_t *sc,
                             int layer, int head,
                             int start_pos, int n_pos,
                             const float *k_vecs) {
    int hd = sc->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_shadow_write_k(sc, layer, head, start_pos + i, k_vecs + (size_t)i * hd);
    }
}

void sp_shadow_write_v_batch(sp_shadow_cache_t *sc,
                             int layer, int head,
                             int start_pos, int n_pos,
                             const float *v_vecs) {
    int hd = sc->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_shadow_write_v(sc, layer, head, start_pos + i, v_vecs + (size_t)i * hd);
    }
}

void sp_shadow_read_k_batch(const sp_shadow_cache_t *sc,
                            int layer, int head,
                            int start_pos, int n_pos,
                            float *k_out) {
    int hd = sc->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_shadow_read_k(sc, layer, head, start_pos + i, k_out + (size_t)i * hd);
    }
}

void sp_shadow_read_v_batch(const sp_shadow_cache_t *sc,
                            int layer, int head,
                            int start_pos, int n_pos,
                            float *v_out) {
    int hd = sc->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_shadow_read_v(sc, layer, head, start_pos + i, v_out + (size_t)i * hd);
    }
}

// ============================================================================
// Ship-path variance-ranked calibration
// ============================================================================

int sp_shadow_calibrate_begin(sp_shadow_cache_t *sc) {
    if (sc->calibrating) return -1;
    int hd = sc->config.head_dim;
    sc->calib_sum  = (double *)calloc(hd, sizeof(double));
    sc->calib_sum2 = (double *)calloc(hd, sizeof(double));
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

void sp_shadow_calibrate_feed(sp_shadow_cache_t *sc, const float *vec) {
    if (!sc->calibrating) return;
    int hd = sc->config.head_dim;

    // Transform to VHT2 domain using the persistent scratch
    memcpy(sc->vht2_scratch, vec, hd * sizeof(float));
    sp_vht2_forward_f32(sc->vht2_scratch, hd);

    for (int i = 0; i < hd; i++) {
        double v = (double)sc->vht2_scratch[i];
        sc->calib_sum[i]  += v;
        sc->calib_sum2[i] += v * v;
    }
    sc->calib_n++;
}

int sp_shadow_calibrate_end(sp_shadow_cache_t *sc) {
    if (!sc->calibrating || sc->calib_n < 1) return -1;
    sc->calibrating = false;

    int hd = sc->config.head_dim;
    double inv_n = 1.0 / (double)sc->calib_n;

    // Compute per-coefficient variance
    float *variance = (float *)malloc(hd * sizeof(float));
    for (int i = 0; i < hd; i++) {
        double mean = sc->calib_sum[i] * inv_n;
        double var  = sc->calib_sum2[i] * inv_n - mean * mean;
        variance[i] = (var > 0.0) ? (float)var : 0.0f;
    }

    free(sc->calib_sum);
    free(sc->calib_sum2);
    sc->calib_sum = NULL;
    sc->calib_sum2 = NULL;

    // Build variance-ranked permutation: indices sorted by variance descending
    // so highest-variance coefficients land in band 0 (highest bits).
    // Free any prior allocation (safe even on first call — they start NULL).
    free(sc->var_order);
    free(sc->var_unorder);
    sc->var_order   = (int *)malloc(hd * sizeof(int));
    sc->var_unorder = (int *)malloc(hd * sizeof(int));
    for (int i = 0; i < hd; i++) sc->var_order[i] = i;

    // Insertion sort (head_dim ≤ 256, not hot path)
    for (int i = 1; i < hd; i++) {
        int key = sc->var_order[i];
        float kv = variance[key];
        int j = i - 1;
        while (j >= 0 && variance[sc->var_order[j]] < kv) {
            sc->var_order[j + 1] = sc->var_order[j];
            j--;
        }
        sc->var_order[j + 1] = key;
    }

    // Build inverse permutation for the read path
    for (int i = 0; i < hd; i++) {
        sc->var_unorder[sc->var_order[i]] = i;
    }

    sc->use_var_reorder = true;
    free(variance);

    if (getenv("SHANNON_PRIME_VERBOSE")) {
        fprintf(stderr, "[Shannon-Prime SHADOW] variance-ranked reorder calibrated "
                        "(head_dim=%d, n_vectors=%d)\n", hd, sc->calib_n);
    }

    sc->calib_n = 0;
    return 0;
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

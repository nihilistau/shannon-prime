// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

//
// CUDA kernels for the sqfree + spinor aggressive compression path.
//
// Additive to backends/cuda/shannon_prime_cuda.cu — the existing WHT
// butterfly kernels are untouched. These new kernels handle:
//
//   1. Prime-Hartley (Vilenkin) transform — successive stages per prime factor
//   2. Möbius CSR prediction — segment-sum over skeleton values
//   3. Spinor sheet bit — pick min(|v_plus|, |v_minus|), store 1-bit flag
//   4. Sqfree pad/unpad — mean-fill padding for non-power-of-2 dimensions
//
// All kernels operate on device memory. No host↔device transfers needed
// when the shadow cache lives on GPU (same as the WHT CUDA path).

#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Vilenkin prime-Hartley kernel
// ============================================================================
//
// One block per vector. Shared memory for the coefficients.
// Successive stages: for each prime factor p of N, apply the p×p Hartley
// matrix at the current stride. Self-inverse (normalized by 1/√p per stage).
//
// Max pad_dim = 330 (hd=256 → 330 = 2·3·5·11). Fits in 48KB shared memory.

__device__ float cas_val(float angle) {
    return __cosf(angle) + __sinf(angle);
}

__global__ void kernel_vilenkin_inplace(
    float *data,        // [n_vecs × pad_dim]
    int    pad_dim,     // Product of prime factors
    int    n_vecs,
    int    n_factors,
    int   *factors      // Device array of prime factors
) {
    extern __shared__ float smem[];

    int vec_idx = blockIdx.x;
    int tid     = threadIdx.x;
    if (vec_idx >= n_vecs) return;

    float *vec = data + vec_idx * pad_dim;

    // Load into shared memory
    if (tid < pad_dim) {
        smem[tid] = vec[tid];
    }
    __syncthreads();

    // Successive Hartley stages
    int stride = 1;
    for (int f = 0; f < n_factors; f++) {
        int p = factors[f];
        int block_size = stride * p;
        int n_blocks_inner = pad_dim / block_size;
        float inv_sqrt_p = rsqrtf((float)p);

        // Each thread handles one (block, stride_pos) pair
        int work_items = n_blocks_inner * stride;
        for (int wi = tid; wi < work_items; wi += blockDim.x) {
            int blk = wi / stride;
            int s   = wi % stride;
            int base = blk * block_size + s;

            // Gather p elements
            float gathered[16]; // Max prime = 13
            for (int k = 0; k < p; k++) {
                gathered[k] = smem[base + k * stride];
            }

            // Apply p×p Hartley matrix
            float result[16];
            for (int k = 0; k < p; k++) {
                float sum = 0.0f;
                for (int j = 0; j < p; j++) {
                    float angle = 2.0f * (float)M_PI * (float)(k * j) / (float)p;
                    sum += cas_val(angle) * gathered[j];
                }
                result[k] = sum * inv_sqrt_p;
            }

            // Scatter back
            for (int k = 0; k < p; k++) {
                smem[base + k * stride] = result[k];
            }
        }

        stride *= p;
        __syncthreads();
    }

    // Write back
    if (tid < pad_dim) {
        vec[tid] = smem[tid];
    }
}

// ============================================================================
// Sqfree pad/unpad kernels
// ============================================================================

__global__ void kernel_sqfree_pad(
    const float *in,    // [n_vecs × head_dim]
    float       *out,   // [n_vecs × pad_dim]
    int          head_dim,
    int          pad_dim,
    int          n_vecs
) {
    int vec_idx = blockIdx.x;
    int tid     = threadIdx.x;
    if (vec_idx >= n_vecs) return;

    const float *src = in  + vec_idx * head_dim;
    float       *dst = out + vec_idx * pad_dim;

    // Compute mean for padding (warp reduction)
    float sum = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        sum += src[i];
    }
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    __shared__ float s_mean;
    if (tid == 0) {
        s_mean = sum / (float)head_dim;
    }
    __syncthreads();

    // Copy data + pad
    for (int i = tid; i < pad_dim; i += blockDim.x) {
        dst[i] = (i < head_dim) ? src[i] : s_mean;
    }
}

__global__ void kernel_sqfree_unpad(
    const float *in,    // [n_vecs × pad_dim]
    float       *out,   // [n_vecs × head_dim]
    int          head_dim,
    int          pad_dim,
    int          n_vecs
) {
    int vec_idx = blockIdx.x;
    int tid     = threadIdx.x;
    if (vec_idx >= n_vecs) return;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        out[vec_idx * head_dim + i] = in[vec_idx * pad_dim + i];
    }
}

// ============================================================================
// Möbius CSR prediction kernel
// ============================================================================
//
// For each residual position i, compute:
//   pred[i] = Σ csr_mu_sign[j] · skel_vals[csr_skel_slot[j]]
//             for j in [csr_offsets[i], csr_offsets[i+1])
//
// One thread per residual position. CSR tables are device-resident constants.

__global__ void kernel_mobius_predict(
    const float *skel_vals,    // [n_vecs × sk_k]
    float       *pred_out,     // [n_vecs × n_res]
    const int   *csr_offsets,  // [n_res + 1]
    const int   *csr_skel_slot,// [n_terms]
    const int   *csr_mu_sign,  // [n_terms] — ±1 as int
    int          sk_k,
    int          n_res,
    int          n_vecs
) {
    int vec_idx = blockIdx.x;
    int res_i   = threadIdx.x;
    if (vec_idx >= n_vecs || res_i >= n_res) return;

    const float *sv = skel_vals + vec_idx * sk_k;
    int start = csr_offsets[res_i];
    int end   = csr_offsets[res_i + 1];

    float pred = 0.0f;
    for (int j = start; j < end; j++) {
        pred += (float)csr_mu_sign[j] * sv[csr_skel_slot[j]];
    }

    pred_out[vec_idx * n_res + res_i] = pred;
}

// ============================================================================
// Spinor sheet bit kernel (compress path)
// ============================================================================
//
// For each residual position, compare |actual - pred| vs |actual + pred|.
// Store the sheet bit (1 if flipped) and the winning deviation.

__global__ void kernel_spinor_extract(
    const float *actual_res,   // [n_vecs × n_res] — actual coeff at residual positions
    const float *pred,         // [n_vecs × n_res] — Möbius predictions
    float       *deviation,    // [n_vecs × n_res] — output: winning deviation
    uint8_t     *sheet_packed, // [n_vecs × (n_res+7)/8] — output: packed sheet bits
    int          n_res,
    int          n_vecs
) {
    int vec_idx = blockIdx.x;
    int res_i   = threadIdx.x;
    if (vec_idx >= n_vecs || res_i >= n_res) return;

    int idx = vec_idx * n_res + res_i;
    float a = actual_res[idx];
    float p = pred[idx];

    float v_plus  = a - p;
    float v_minus = a + p;

    bool use_minus = fabsf(v_minus) < fabsf(v_plus);
    deviation[idx] = use_minus ? v_minus : v_plus;

    // Pack sheet bit (atomicOr for thread safety within a byte)
    if (use_minus) {
        int byte_idx = vec_idx * ((n_res + 7) / 8) + res_i / 8;
        atomicOr((unsigned int *)(sheet_packed + (byte_idx & ~3)),
                 (1u << (res_i % 8)) << ((byte_idx & 3) * 8));
    }
}

// ============================================================================
// Spinor correction kernel (reconstruct path)
// ============================================================================
//
// Flip pred sign where sheet bit is set, add dequantized residual,
// scatter to coefficient vector.

__global__ void kernel_spinor_reconstruct(
    float       *coeffs,       // [n_vecs × pad_dim] — output coefficient vector
    const float *skel_vals,    // [n_vecs × sk_k] — dequantized skeleton
    const float *deviation,    // [n_vecs × n_res] — dequantized residual
    const float *pred,         // [n_vecs × n_res] — Möbius predictions (recomputed)
    const uint8_t *sheet_packed,// [n_vecs × (n_res+7)/8]
    const int   *residual_idx, // [n_res] — where in coeffs[] to write
    int          pad_dim,
    int          sk_k,
    int          n_res,
    int          n_vecs,
    int          use_spinor
) {
    int vec_idx = blockIdx.x;
    int res_i   = threadIdx.x;
    if (vec_idx >= n_vecs || res_i >= n_res) return;

    float p = pred[vec_idx * n_res + res_i];

    // Apply spinor sign flip
    if (use_spinor) {
        int byte_idx = vec_idx * ((n_res + 7) / 8) + res_i / 8;
        uint8_t byte_val = sheet_packed[byte_idx];
        if (byte_val & (1 << (res_i % 8))) {
            p = -p;
        }
    }

    float val = p + deviation[vec_idx * n_res + res_i];
    coeffs[vec_idx * pad_dim + residual_idx[res_i]] = val;
}

// ============================================================================
// Host-side dispatch helpers
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Launch Vilenkin transform on a batch of vectors
void sp_cuda_vilenkin_inplace(float *d_data, int pad_dim, int n_vecs,
                              int *d_factors, int n_factors,
                              cudaStream_t stream) {
    int smem_bytes = pad_dim * sizeof(float);
    int block_size = (pad_dim < 256) ? pad_dim : 256;
    kernel_vilenkin_inplace<<<n_vecs, block_size, smem_bytes, stream>>>(
        d_data, pad_dim, n_vecs, n_factors, d_factors
    );
}

// Launch sqfree pad
void sp_cuda_sqfree_pad(const float *d_in, float *d_out,
                        int head_dim, int pad_dim, int n_vecs,
                        cudaStream_t stream) {
    kernel_sqfree_pad<<<n_vecs, 256, 0, stream>>>(
        d_in, d_out, head_dim, pad_dim, n_vecs
    );
}

// Launch Möbius prediction
void sp_cuda_mobius_predict(const float *d_skel, float *d_pred,
                           const int *d_offsets, const int *d_slots,
                           const int *d_signs,
                           int sk_k, int n_res, int n_vecs,
                           cudaStream_t stream) {
    int block = (n_res < 256) ? n_res : 256;
    kernel_mobius_predict<<<n_vecs, block, 0, stream>>>(
        d_skel, d_pred, d_offsets, d_slots, d_signs,
        sk_k, n_res, n_vecs
    );
}

// Launch spinor extract (compress path)
void sp_cuda_spinor_extract(const float *d_actual, const float *d_pred,
                            float *d_deviation, uint8_t *d_sheet,
                            int n_res, int n_vecs,
                            cudaStream_t stream) {
    // Zero sheet bits first
    int sheet_bytes = n_vecs * ((n_res + 7) / 8);
    cudaMemsetAsync(d_sheet, 0, sheet_bytes, stream);

    int block = (n_res < 256) ? n_res : 256;
    kernel_spinor_extract<<<n_vecs, block, 0, stream>>>(
        d_actual, d_pred, d_deviation, d_sheet, n_res, n_vecs
    );
}

#ifdef __cplusplus
}
#endif
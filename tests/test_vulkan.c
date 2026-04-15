// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

// Vulkan backend validation.
// Without Vulkan SDK, exercises the CPU fallback path to validate
// the API surface and ensure correct results through the Vulkan abstraction.

#include "../backends/vulkan/shannon_prime_vulkan.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int tests_run = 0, tests_passed = 0;
#define CHECK(cond, msg) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("  [%s] %s\n", PASS, msg); } \
    else { printf("  [%s] %s\n", FAIL, msg); } \
} while(0)

int main(void) {
    srand(42);
    printf("Shannon-Prime Vulkan Backend Validation\n");
    printf("========================================\n");

    // Test 1: Init with NULL device → CPU fallback
    printf("\n== Vulkan Init (CPU fallback) ==\n");
    {
        sp_config_t cfg;
        sp_config_init(&cfg, 128, 4, 2);

        sp_vulkan_cache_t *cc = NULL;
        int rc = sp_vulkan_cache_init(&cc, &cfg, 1024, NULL, NULL);
        CHECK(rc == 0 && cc != NULL, "Init with CPU fallback succeeds");

        sp_vulkan_check_device(cc);

        // Test 2: Full pipeline through Vulkan API
        printf("\n== Vulkan Pipeline (via CPU fallback) ==\n");
        int hd = 128;
        float *k_orig  = (float *)malloc(hd * sizeof(float));
        float *k_recon = (float *)malloc(hd * sizeof(float));
        for (int i = 0; i < hd; i++) {
            k_orig[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }

        sp_vulkan_write_k(cc, 0, 0, 0, k_orig);
        sp_vulkan_read_k(cc, 0, 0, 0, k_recon);

        float k_corr = sp_correlation_f32(k_orig, k_recon, hd);
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "K pipeline: correlation=%.4f (need >0.990)", k_corr);
        CHECK(k_corr > 0.990f, msg);

        float *v_orig  = (float *)malloc(hd * sizeof(float));
        float *v_recon = (float *)malloc(hd * sizeof(float));
        for (int i = 0; i < hd; i++) {
            v_orig[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }

        sp_vulkan_write_v(cc, 0, 0, 0, v_orig);
        sp_vulkan_read_v(cc, 0, 0, 0, v_recon);

        float v_corr = sp_correlation_f32(v_orig, v_recon, hd);
        snprintf(msg, sizeof(msg),
                 "V pipeline: correlation=%.4f (need >0.950)", v_corr);
        CHECK(v_corr > 0.950f, msg);

        // Test 3: Batch write/read
        printf("\n== Vulkan Batch Ops ==\n");
        int n_pos = 8;
        float *k_batch = (float *)malloc(n_pos * hd * sizeof(float));
        float *k_batch_recon = (float *)malloc(n_pos * hd * sizeof(float));

        for (int i = 0; i < n_pos * hd; i++) {
            k_batch[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }

        sp_vulkan_write_k_batch(cc, 1, 0, 0, n_pos, k_batch);
        sp_vulkan_read_k_batch(cc, 1, 0, 0, n_pos, k_batch_recon);

        float total_corr = 0;
        for (int p = 0; p < n_pos; p++) {
            total_corr += sp_correlation_f32(
                k_batch + p * hd, k_batch_recon + p * hd, hd);
        }
        float avg_corr = total_corr / n_pos;
        snprintf(msg, sizeof(msg),
                 "Batch K (%d pos): avg_corr=%.4f (need >0.990)", n_pos, avg_corr);
        CHECK(avg_corr > 0.990f, msg);

        // Test 4: Memory reporting
        printf("\n== Memory ==\n");
        sp_vulkan_print_memory(cc);

        free(k_orig); free(k_recon);
        free(v_orig); free(v_recon);
        free(k_batch); free(k_batch_recon);
        sp_vulkan_cache_free(cc);
    }

    printf("\n========================================\n");
    printf("Results: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}

# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
# Licensed under AGPLv3. Commercial license available.

CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -O2
LDFLAGS = -lm

NVCC       = nvcc
NVCC_ARCH  = -arch=sm_75
NVCC_FLAGS = -O2 $(NVCC_ARCH) -Icore

CORE_SRC       = core/shannon_prime.c core/shannon_prime_cauchy.c
SQFREE_SRC     = core/shannon_prime_sqfree.c
MODELPACK_SRC  = core/shannon_prime_modelpack.c
CORE_HDR       = core/shannon_prime.h core/shannon_prime_modelpack.h
ADRENO_SRC = backends/adreno/shannon_prime_adreno.c
VULKAN_SRC = backends/vulkan/shannon_prime_vulkan.c
CUDA_SRC   = backends/cuda/shannon_prime_cuda.cu
CUDA_SQFREE_SRC = backends/cuda/shannon_prime_sqfree.cu
CUDA_HIER_SRC   = backends/cuda/shannon_prime_hier.cu
LLAMA_SRC  = tools/shannon_prime_llama.c

# ── Vulkan SDK (optional) ────────────────────────────────────────
# Enable the real GPU path when the SDK is present. Otherwise the
# backend builds as before with the CPU fallback (which routes through
# the core staged VHT2, so sqfree dims still work — just on CPU).
ifneq ($(strip $(VULKAN_SDK)),)
  VULKAN_ENABLED     = 1
  VULKAN_INC         = -I"$(VULKAN_SDK)/Include"
  VULKAN_LIB         = -L"$(VULKAN_SDK)/Lib" -lvulkan-1
  GLSLC              = "$(VULKAN_SDK)/Bin/glslc"
  VULKAN_CFLAGS      = -DSHANNON_PRIME_VULKAN_ENABLED=1 $(VULKAN_INC)
  SHADER_SRC_DIR     = backends/vulkan/shaders
  SHADER_BUILD_DIR   = build/shaders
  SHADERS            = vilenkin mobius_reorder band_quantize band_dequantize
  SHADER_SPVS        = $(patsubst %,$(SHADER_BUILD_DIR)/%.spv,$(SHADERS))
else
  VULKAN_ENABLED     =
  VULKAN_CFLAGS      =
  VULKAN_LIB         =
  SHADER_SPVS        =
endif

.PHONY: all test test-core test-torch test-adreno test-vulkan test-cuda \
        test-cuda-advanced test-integration test-comfyui test-sqfree \
        test-modelpack test-all clean

all: test-all

# ── C backends ───────────────────────────────────────────────────
build/test_core: tests/test_core.c $(CORE_SRC) $(SQFREE_SRC) $(MODELPACK_SRC) $(CORE_HDR)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tests/test_core.c $(CORE_SRC) $(SQFREE_SRC) $(MODELPACK_SRC) $(LDFLAGS)

build/test_modelpack: tests/test_modelpack.c $(CORE_SRC) $(MODELPACK_SRC) $(CORE_HDR)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tests/test_modelpack.c $(CORE_SRC) $(MODELPACK_SRC) $(LDFLAGS)

build/test_adreno: tests/test_adreno.c $(ADRENO_SRC) $(CORE_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tests/test_adreno.c $(ADRENO_SRC) $(CORE_SRC) $(LDFLAGS)

# Compile each compute shader to SPIR-V when the Vulkan SDK is available
$(SHADER_BUILD_DIR)/%.spv: $(SHADER_SRC_DIR)/%.comp
	@mkdir -p $(SHADER_BUILD_DIR)
	$(GLSLC) -O $< -o $@

build/test_vulkan: tests/test_vulkan.c $(VULKAN_SRC) $(CORE_SRC) $(SQFREE_SRC) $(SHADER_SPVS)
	@mkdir -p build
	$(CC) $(CFLAGS) $(VULKAN_CFLAGS) -o $@ tests/test_vulkan.c $(VULKAN_SRC) $(CORE_SRC) $(SQFREE_SRC) $(LDFLAGS) $(VULKAN_LIB)

build/test_cuda: tests/test_cuda.c $(CUDA_SRC) $(CUDA_SQFREE_SRC) $(CUDA_HIER_SRC) $(CORE_SRC) $(SQFREE_SRC) $(CORE_HDR)
	@mkdir -p build
	$(NVCC) $(NVCC_FLAGS) -o $@ tests/test_cuda.c $(CUDA_SRC) $(CUDA_SQFREE_SRC) $(CUDA_HIER_SRC) $(CORE_SRC) $(SQFREE_SRC) -lcudart

build/test_cuda_advanced: tests/test_cuda_advanced.c $(CUDA_SRC) $(CUDA_SQFREE_SRC) $(CUDA_HIER_SRC) $(CORE_SRC) $(SQFREE_SRC) $(CORE_HDR)
	@mkdir -p build
	$(NVCC) $(NVCC_FLAGS) -o $@ tests/test_cuda_advanced.c $(CUDA_SRC) $(CUDA_SQFREE_SRC) $(CUDA_HIER_SRC) $(CORE_SRC) $(SQFREE_SRC) -lcudart

build/test_integration: tests/test_integration.c $(LLAMA_SRC) $(CORE_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tests/test_integration.c $(LLAMA_SRC) $(CORE_SRC) $(LDFLAGS)

# ── Test targets ─────────────────────────────────────────────────
test-core: build/test_core
	@echo "── Core math (31 tests) ──"
	@./build/test_core

test-modelpack: build/test_modelpack
	@echo "── Model-pack registry ──"
	@./build/test_modelpack

test-adreno: build/test_adreno
	@echo "── Adreno/ARM backend (14 tests) ──"
	@./build/test_adreno

test-vulkan: build/test_vulkan
	@echo "── Vulkan backend (4 tests) ──"
	@./build/test_vulkan

test-cuda: build/test_cuda
	@echo "── CUDA backend (7 tests) ──"
	@./build/test_cuda

test-cuda-advanced: build/test_cuda_advanced
	@echo "── CUDA advanced (sqfree/hier/cold/stress) ──"
	@./build/test_cuda_advanced

test-integration: build/test_integration
	@echo "── llama.cpp integration (7 tests) ──"
	@./build/test_integration

test-torch:
	@echo "── PyTorch backend (28 tests) ──"
	@python3 tests/test_torch.py

test-comfyui:
	@echo "── ComfyUI integration (25 tests) ──"
	@python3 tests/test_comfyui.py

test-sqfree:
	@echo "── Sqfree+spinor path (69 tests) ──"
	@PYTHONIOENCODING=utf-8 python3 tests/test_sqfree.py

test-all: test-core test-adreno test-vulkan test-cuda test-integration test-torch test-comfyui test-sqfree
	@echo ""
	@echo "════════════════════════════════════════"
	@echo " All 185 tests passed across 8 suites."
	@echo "════════════════════════════════════════"

test: test-core

clean:
	rm -rf build/

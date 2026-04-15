# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
# Licensed under AGPLv3. Commercial license available.

CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -O2
LDFLAGS = -lm

CORE_SRC = core/shannon_prime.c
CORE_HDR = core/shannon_prime.h
ADRENO_SRC = backends/adreno/shannon_prime_adreno.c
VULKAN_SRC = backends/vulkan/shannon_prime_vulkan.c
LLAMA_SRC  = tools/shannon_prime_llama.c

.PHONY: all test test-core test-torch test-adreno test-vulkan test-integration \
        test-comfyui test-all clean

all: test-all

# ── C backends ───────────────────────────────────────────────────
build/test_core: tests/test_core.c $(CORE_SRC) $(CORE_HDR)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tests/test_core.c $(CORE_SRC) $(LDFLAGS)

build/test_adreno: tests/test_adreno.c $(ADRENO_SRC) $(CORE_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tests/test_adreno.c $(ADRENO_SRC) $(CORE_SRC) $(LDFLAGS)

build/test_vulkan: tests/test_vulkan.c $(VULKAN_SRC) $(CORE_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tests/test_vulkan.c $(VULKAN_SRC) $(CORE_SRC) $(LDFLAGS)

build/test_integration: tests/test_integration.c $(LLAMA_SRC) $(CORE_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tests/test_integration.c $(LLAMA_SRC) $(CORE_SRC) $(LDFLAGS)

# ── Test targets ─────────────────────────────────────────────────
test-core: build/test_core
	@echo "── Core math (31 tests) ──"
	@./build/test_core

test-adreno: build/test_adreno
	@echo "── Adreno/ARM backend (14 tests) ──"
	@./build/test_adreno

test-vulkan: build/test_vulkan
	@echo "── Vulkan backend (4 tests) ──"
	@./build/test_vulkan

test-integration: build/test_integration
	@echo "── llama.cpp integration (7 tests) ──"
	@./build/test_integration

test-torch:
	@echo "── PyTorch backend (28 tests) ──"
	@python3 tests/test_torch.py

test-comfyui:
	@echo "── ComfyUI integration (25 tests) ──"
	@python3 tests/test_comfyui.py

test-all: test-core test-adreno test-vulkan test-integration test-torch test-comfyui
	@echo ""
	@echo "════════════════════════════════════════"
	@echo " All 109 tests passed across 6 suites."
	@echo "════════════════════════════════════════"

test: test-core

clean:
	rm -rf build/

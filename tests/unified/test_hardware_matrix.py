#!/usr/bin/env python3
# Shannon-Prime VHT2: Hardware Matrix Tests
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
# Licensed under AGPLv3.

import sys
import os
import ctypes
import platform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shannon-prime', 'backends', 'torch'))

try:
    import torch
    torch.manual_seed(42)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from run_tests import register_suite


def _find_sp_lib():
    """Try to find compiled shannon_prime shared library."""
    base = os.path.join(os.path.dirname(__file__), '..', 'shannon-prime')
    candidates = [
        os.path.join(base, 'build', 'libshannon_prime.so'),
        os.path.join(base, 'build', 'shannon_prime.dll'),
        os.path.join(base, 'build', 'libshannon_prime.dylib'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


@register_suite("hardware", "Hardware matrix: CPU/CUDA/Vulkan numerical agreement")
def suite_hardware(t):
    if not HAS_TORCH:
        t.check(False, "PyTorch required for hardware tests")
        return

    from shannon_prime_torch import vht2, BandedQuantizer, ShadowCache, correlation

    # ========================================================================
    # SECTION 1: Platform Detection
    # ========================================================================
    t.set_category("platform")
    print("\n  -- Platform Detection --")

    t.check(True, f"OS: {platform.system()} {platform.release()}")
    t.check(True, f"Arch: {platform.machine()}")
    t.check(True, f"Python: {platform.python_version()}")
    t.check(True, f"PyTorch: {torch.__version__}")

    has_cuda = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if has_cuda else 0
    t.check(True, f"CUDA: {'yes' if has_cuda else 'no'} ({cuda_count} devices)")

    if has_cuda:
        for i in range(cuda_count):
            props = torch.cuda.get_device_properties(i)
            t.check(True, f"  GPU {i}: {props.name} ({props.total_mem/1e9:.1f}GB, "
                    f"SM {props.major}.{props.minor})")
            # Detect Blackwell (SM 10.x)
            if props.major >= 10:
                t.check(True, f"  GPU {i}: Blackwell detected (SM {props.major}.{props.minor})")

    # ========================================================================
    # SECTION 2: CPU Baseline
    # ========================================================================
    t.set_category("cpu")
    print("\n  -- CPU Baseline --")

    # VHT2 round-trip on CPU
    for hd in [64, 128, 256]:
        x = torch.randn(hd, device='cpu')
        recon = vht2(vht2(x.unsqueeze(0))).squeeze(0)
        err = (x - recon).abs().max().item()
        t.check(err < 1e-5, f"CPU VHT2 round-trip hd={hd}: err={err:.2e}")

    # ShadowCache on CPU
    cache = ShadowCache(head_dim=128, n_layers=1, n_heads_kv=1, max_seq_len=32,
                        k_band_bits=[4,3,3,3], v_band_bits=[3,3,3,3])
    k = torch.randn(128)
    v = torch.randn(128)
    cache.write_k(0, 0, 0, k)
    cache.write_v(0, 0, 0, v)
    k_out = cache.read_k(0, 0, 0)
    cpu_corr = correlation(k, k_out)
    t.check(cpu_corr > 0.97, f"CPU ShadowCache K_corr={cpu_corr:.4f}")

    # ========================================================================
    # SECTION 3: CUDA Parity (if available)
    # ========================================================================
    if has_cuda:
        t.set_category("cuda")
        print("\n  -- CUDA Parity --")

        # VHT2 on CUDA should match CPU
        for hd in [64, 128, 256]:
            x = torch.randn(hd, device='cpu')
            cpu_result = vht2(x.unsqueeze(0)).squeeze(0)

            x_cuda = x.cuda()
            cuda_result = vht2(x_cuda.unsqueeze(0)).squeeze(0).cpu()

            err = (cpu_result - cuda_result).abs().max().item()
            t.check(err < 1e-4, f"CUDA vs CPU VHT2 hd={hd}: max_err={err:.2e}")

        # Full pipeline parity
        quant = BandedQuantizer(128, [4, 3, 3, 3])
        x = torch.randn(128)
        X_cpu = vht2(x.unsqueeze(0))
        Xs_cpu, Xq_cpu = quant.quantize(X_cpu)
        Xd_cpu = quant.dequantize(Xs_cpu, Xq_cpu)
        recon_cpu = vht2(Xd_cpu).squeeze(0)
        cpu_corr = correlation(x, recon_cpu)

        x_cuda = x.cuda()
        X_cuda = vht2(x_cuda.unsqueeze(0))
        Xs_cuda, Xq_cuda = quant.quantize(X_cuda)
        Xd_cuda = quant.dequantize(Xs_cuda, Xq_cuda)
        recon_cuda = vht2(Xd_cuda).squeeze(0).cpu()
        cuda_corr = correlation(x, recon_cuda)

        t.check(abs(cpu_corr - cuda_corr) < 0.005,
                f"Pipeline parity: CPU={cpu_corr:.4f} CUDA={cuda_corr:.4f}")

        # CUDA memory tracking
        torch.cuda.reset_peak_memory_stats()
        cache_cuda = ShadowCache(head_dim=128, n_layers=4, n_heads_kv=8, max_seq_len=256,
                                 k_band_bits=[4,3,3,3], v_band_bits=[3,3,3,3])
        for layer in range(4):
            for head in range(8):
                for pos in range(256):
                    k = torch.randn(128, device='cuda')
                    v = torch.randn(128, device='cuda')
                    cache_cuda.write_k(layer, head, pos, k)
                    cache_cuda.write_v(layer, head, pos, v)

        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        baseline_mb = 4 * 8 * 256 * 128 * 2 * 2 / 1e6  # fp16 uncompressed
        t.check(peak_mb < baseline_mb * 1.5,
                f"CUDA memory: {peak_mb:.1f}MB (baseline {baseline_mb:.1f}MB)")

        # ── FP8 Tests (Ada+ only, SM >= 8.9) ──
        if cuda_count > 0:
            props = torch.cuda.get_device_properties(0)
            if props.major > 8 or (props.major == 8 and props.minor >= 9):
                t.set_category("fp8")
                print("\n  -- FP8 (Ada+) --")

                # torch.float8_e4m3fn available?
                has_fp8 = hasattr(torch, 'float8_e4m3fn')
                t.check(has_fp8, "torch.float8_e4m3fn available")
                if has_fp8:
                    x = torch.randn(128, device='cuda')
                    x8 = x.to(torch.float8_e4m3fn)
                    x_back = x8.to(torch.float32)
                    err = (x - x_back).abs().max().item()
                    t.check(err < 0.5, f"FP8 E4M3 round-trip max_err={err:.3f}")

            # ── Blackwell FP4 (SM >= 10.0) ──
            if props.major >= 10:
                t.set_category("blackwell")
                print("\n  -- Blackwell FP4 --")
                t.check(True, f"Blackwell GPU detected: {props.name}")
                # FP4 not yet in PyTorch — placeholder for when it lands
                t.check(True, "FP4 tests pending PyTorch fp4 dtype support")

    else:
        print("\n  [SKIP] No CUDA — skipping GPU parity tests")

    # ========================================================================
    # SECTION 4: C Library (if compiled)
    # ========================================================================
    t.set_category("native")
    print("\n  -- Native C Library --")

    lib_path = _find_sp_lib()
    if lib_path:
        try:
            lib = ctypes.CDLL(lib_path)
            t.check(True, f"Loaded {os.path.basename(lib_path)}")

            # Check sp_config_init exists
            t.check(hasattr(lib, 'sp_config_init'), "sp_config_init symbol found")
            t.check(hasattr(lib, 'sp_vht2_forward'), "sp_vht2_forward symbol found")
            t.check(hasattr(lib, 'sp_prime_pe_freq_factors'), "sp_prime_pe_freq_factors symbol found")
        except OSError as e:
            t.check(False, f"Failed to load: {e}")
    else:
        print("  [SKIP] No compiled shannon_prime library found")
        t.check(True, "Native lib not compiled (build with make first)")

    # ========================================================================
    # SECTION 5: Multi-Device Consistency (if >1 GPU)
    # ========================================================================
    if has_cuda and cuda_count > 1:
        t.set_category("multi-gpu")
        print("\n  -- Multi-GPU Consistency --")

        x = torch.randn(128)
        results = []
        for dev_id in range(min(cuda_count, 4)):
            x_dev = x.to(f'cuda:{dev_id}')
            X = vht2(x_dev.unsqueeze(0)).squeeze(0).cpu()
            results.append(X)

        for i in range(1, len(results)):
            err = (results[0] - results[i]).abs().max().item()
            t.check(err < 1e-4, f"GPU 0 vs GPU {i}: max_err={err:.2e}")

    # ========================================================================
    # SECTION 6: Dtype Promotion
    # ========================================================================
    t.set_category("dtype")
    print("\n  -- Dtype Promotion --")

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        name = str(dtype).split('.')[-1]
        x = torch.randn(128, dtype=torch.float32)
        x_cast = x.to(dtype)
        X = vht2(x_cast.unsqueeze(0)).squeeze(0)
        # Should produce output without NaN/Inf
        t.check(not torch.isnan(X).any().item(), f"{name}: no NaN in VHT2 output")
        t.check(not torch.isinf(X).any().item(), f"{name}: no Inf in VHT2 output")

        if dtype == torch.float32:
            recon = vht2(X.unsqueeze(0)).squeeze(0)
            err = (x - recon).abs().max().item()
            t.check(err < 1e-5, f"{name}: round-trip err={err:.2e}")

    # ========================================================================
    # SECTION 7: Throughput Benchmark (informational)
    # ========================================================================
    t.set_category("benchmark")
    print("\n  -- Throughput Benchmark --")

    import time

    # CPU throughput
    n_iters = 1000
    x_batch = torch.randn(n_iters, 128)
    t0 = time.perf_counter()
    for i in range(n_iters):
        vht2(x_batch[i:i+1])
    cpu_ms = (time.perf_counter() - t0) * 1000
    t.check(True, f"CPU VHT2: {n_iters} iters in {cpu_ms:.1f}ms "
            f"({cpu_ms/n_iters:.3f}ms/iter)")

    if has_cuda:
        x_batch_cuda = x_batch.cuda()
        # Warmup
        for _ in range(10):
            vht2(x_batch_cuda[0:1])
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for i in range(n_iters):
            vht2(x_batch_cuda[i:i+1])
        torch.cuda.synchronize()
        cuda_ms = (time.perf_counter() - t0) * 1000
        t.check(True, f"CUDA VHT2: {n_iters} iters in {cuda_ms:.1f}ms "
                f"({cuda_ms/n_iters:.3f}ms/iter)")

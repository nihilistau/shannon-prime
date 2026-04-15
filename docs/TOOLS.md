# Shannon-Prime: Tools Reference

## Overview

Three command-line tools for working with Shannon-Prime outside of the runtime integration:

| Tool | Purpose | Requires Model? |
|------|---------|-----------------|
| `sp_benchmark.py` | Validate VHT2 math, measure compression quality | No |
| `sp_inject_freqs.py` | Inject lattice RoPE frequencies into GGUF models | Yes (GGUF) |
| `sp_compress_model.py` | Analyze/compress model weights via VHT2 | Yes (safetensors) |

---

## sp_benchmark.py — Compression Benchmark

Demonstrates and measures VHT2 KV cache compression on synthetic vectors. This is the first thing to run — it validates that the compression math works and shows the K/V spectral asymmetry that makes Shannon-Prime possible.

No model required. No GPU required. Runs in seconds.

### Usage

```bash
# Full benchmark across all standard configs
python tools/sp_benchmark.py

# Spectral energy analysis — see WHT concentration
python tools/sp_benchmark.py --spectral

# Mobile head dimension
python tools/sp_benchmark.py --head-dim 64

# Test a specific bit allocation
python tools/sp_benchmark.py --config 5,5,4,3

# More vectors for stable statistics
python tools/sp_benchmark.py --n-vectors 2000
```

### What the Output Shows

**Spectral Analysis** (`--spectral`):

```
  RoPE-like K:
    Band    Energy% Bar
       0       3.6%  █
       1      76.7%  ██████████████████████████████████████
       2       1.9%
       3      17.8%  ████████
  First half: 80.3% (concentrated → banded quant helps)

  Random V:
    Band    Energy% Bar
       0      25.6%  ████████████
       1      38.0%  ███████████████████
       2      19.8%  █████████
       3      16.5%  ████████
  First half: 63.6% (uniform → flat quant OK)
```

This is the foundational observation. K vectors (with RoPE periodicity) concentrate 80%+ of their WHT energy in the first two bands. V vectors (random content) spread energy uniformly. This asymmetry is why K gets banded quantization (more bits where energy is) and V gets flat quantization (all bands equally important).

**Compression Table**:

Shows correlation, compression ratio, bytes per vector, and microseconds per vector for each bit allocation config. Look for:

- Correlation >0.99 on ship config (5/5/4/3)
- Möbius improvement is positive (typically +0.003)
- Flat beats banded for V vectors

**Memory Projection**:

Shows actual GB savings for a real model configuration (32K context, 32 layers, 8 KV heads).

### Dependencies

PyTorch only (for the torch backend). No model files needed.

---

## sp_inject_freqs.py — GGUF Frequency Injection

Blends prime-lattice-aligned RoPE frequencies into an existing GGUF model. This corrects positional frequency mismatch — a structural error in standard geometric RoPE that is independent of weight quantization precision.

**Zero retraining.** The modified model loads and runs normally in llama.cpp. The only difference is which RoPE frequencies are used during inference.

### Paper Results

| Model | Quant | Baseline PPL | Injected PPL | α | Improvement |
|-------|-------|-------------|-------------|---|-------------|
| Dolphin 3.0 Llama 3.2 1B | Q8_0 | 11.6413 | 11.5462 | 0.22 | −0.82% |
| Dolphin 3.0 Llama 3.2 1B | Q6_K | 11.7615 | 11.6843 | 0.17 | −0.66% |
| Dolphin 3.0 Llama 3.2 1B | Q4_K_M | 12.2380 | 12.1630 | 0.17 | −0.61% |
| Qwen2.5-3B | — | — | — | 0.12 | −0.20% |
| Phi-3.1-3.8B | — | — | — | 0.05 | −0.02% |

Optimal α range: 0.15–0.22 (flat optimum, deployment-robust). Higher `rope_freq_base` → wider harmonic gaps → more room for injection.

### Usage

```bash
# Show model's RoPE configuration
python tools/sp_inject_freqs.py model.gguf --info

# Analyze what injection would do (no file modification)
python tools/sp_inject_freqs.py model.gguf --analyze --alpha 0.17

# Inject and produce output
python tools/sp_inject_freqs.py model.gguf model_sp.gguf --alpha 0.17

# Use prime frequencies instead of composite
python tools/sp_inject_freqs.py model.gguf model_sp.gguf --alpha 0.22 --tier-mode prime
```

### Alpha Sweep

To find the optimal α for your specific model:

```bash
# PowerShell
for ($a=0.05; $a -le 0.30; $a+=0.05) {
    python tools/sp_inject_freqs.py model.gguf "model_a$a.gguf" --alpha $a
}
# Then benchmark each with llama.cpp perplexity

# Bash
for a in 0.05 0.10 0.15 0.17 0.20 0.22 0.25 0.30; do
    python tools/sp_inject_freqs.py model.gguf model_a${a}.gguf --alpha $a
done
```

### How It Works

Standard RoPE uses geometric frequencies: θ_j = base^(−2j/d). This creates redundant coverage at some scales and gaps at others. Shannon-Prime replaces a fraction (α) of these frequencies with lattice-aligned values distributed across three tiers:

| Tier | Head Range | Frequency Range | Scale |
|------|-----------|-----------------|-------|
| Local (25%) | First quarter | 2–101 | Word/syntax |
| Mid (33%) | Middle third | 101–1009 | Clause/paragraph |
| Long (42%) | Last 42% | 1009–8209 | Section/document |

The tiered allocation preserves per-head scale diversity (which is load-bearing — uniform replacement caused +370% degradation in the paper).

### Output Files

The tool produces two files:

1. **`output.gguf`** — Copy of input model (currently metadata-identical; full GGUF tensor modification for embedding freq_factors directly is the next integration step)
2. **`output.sp_freq_factors.bin`** — Binary file containing the freq_factors array (n_freqs × float32). This companion file is loaded by the Shannon-Prime llama.cpp integration layer.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--alpha` | 0.17 | Blending ratio. 0 = pure geometric, 1 = pure lattice. |
| `--tier-mode` | composite | `prime` (lattice generators) or `composite` (lattice coordinates). Paper showed identical results. |
| `--info` | — | Print model RoPE info and exit |
| `--analyze` | — | Show per-frequency analysis without modifying |
| `--quiet` | — | Suppress verbose output |

### Dependencies

`gguf` Python package (`pip install gguf`), NumPy.

---

## sp_compress_model.py — Weight Compression Analysis

Analyzes and optionally compresses transformer model weights using VHT2 spectral quantization. This is the research tool that answers the question: do model weights have the same spectral structure as KV cache vectors?

### The Hypothesis

W_K and W_Q weight matrices are trained with RoPE applied. The learned weights implicitly encode the frequency patterns they were optimized for. If the rows of W_K show WHT spectral concentration similar to K vectors at inference time, VHT2 weight quantization will outperform generic quantization on those specific tensors.

W_V, W_O, and FFN weights were NOT trained with RoPE — they shouldn't show this structure, and they shouldn't benefit from VHT2.

### Usage

```bash
# Analyze a HuggingFace model directory (safetensors format)
python tools/sp_compress_model.py --analyze path/to/model/

# Specify head dimension explicitly
python tools/sp_compress_model.py --analyze path/to/model/ --head-dim 64

# Analyze with custom bit allocation
python tools/sp_compress_model.py --analyze path/to/model/ --k-bits 4,4,4,3
```

### What the Analysis Shows

```
Tensor                                        Shape                Conc%  VHT2   Flat      Δ
───────────────────────────────────────────── ──────────────────── ────── ────── ────── ───────
  model.layers.0.self_attn.q_proj.weight      [4096, 4096]         72.3% 0.994  0.991 +0.0030 ◀
  model.layers.0.self_attn.k_proj.weight      [1024, 4096]         74.1% 0.995  0.990 +0.0050 ◀
  model.layers.0.self_attn.v_proj.weight      [1024, 4096]         51.2% 0.991  0.991 +0.0000
  model.layers.0.self_attn.o_proj.weight      [4096, 4096]         53.8% 0.990  0.991 -0.0010
```

For each attention weight tensor, the tool reports:

- **Conc%**: WHT spectral concentration (first half energy). >60% indicates structure VHT2 can exploit.
- **VHT2**: Correlation after VHT2 compression (WHT → Möbius → 5/5/4/3 banded quant → reconstruct).
- **Flat**: Correlation after flat quantization (same total bits, uniform 4/4/4/4 allocation).
- **Δ**: VHT2 advantage. Positive = VHT2 is better. Marked with ◀ when significant.

**What to look for:**

- Q_proj and K_proj should show higher concentration (>65%) and positive Δ
- V_proj and O_proj should show ~50% concentration and near-zero Δ
- If Q/K don't show structure, VHT2 weight compression won't help for this model

### Compression Mode (Research)

Full weight compression (producing a modified model with VHT2-compressed Q/K weights) is implemented in code but requires a custom dequantization path at inference time. The analysis mode is the validated, ship-ready tool. The compression path is the next integration piece.

### Dependencies

PyTorch, safetensors (`pip install torch safetensors`).

---

## Tool Workflow

The three tools form a progression:

```
1. sp_benchmark.py          Does VHT2 work at all?
   (no model needed)        → Confirms spectral asymmetry and compression quality

2. sp_inject_freqs.py       Can we improve this model's RoPE?
   (needs GGUF model)       → Produces freq-injected model for PPL testing

3. sp_compress_model.py     Do this model's weights have spectral structure?
   (needs HF safetensors)   → Identifies which tensors benefit from VHT2
```

For production deployment, `sp_inject_freqs.py` is the immediate tool — it produces a GGUF file that runs in unmodified llama.cpp with measurable PPL improvement. The VHT2 KV cache compression (the core Shannon-Prime system) operates at inference time via the shannon-prime library integration, not via model file modification.

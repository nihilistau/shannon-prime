# Shannon-Prime Model Pack

A model pack is a small record that carries known-good compression defaults for
a specific model architecture family. It lets the bridges (engine, llama.cpp,
comfyui) ship an honest "auto-adapts" story: a Qwen-3 MoE user running the
default path gets a config tuned for Qwen-3 MoE, not the Llama-3 defaults.

This document covers the API, the current registry, and how to promote a
preset from `PROVISIONAL` to `CALIBRATED`.

## API

Header: `core/shannon_prime_modelpack.h`.

```c
const sp_model_preset_t *preset =
    sp_model_preset_resolve(arch_name, head_dim, n_layers, n_heads_kv);

if (preset) {
    sp_config_apply_preset(&cfg, preset);
    char buf[256];
    sp_model_preset_describe(preset, buf, sizeof(buf));
    fprintf(stderr, "[Shannon-Prime] model-pack: %s\n", buf);
}
```

`arch_name` is the GGUF `general.architecture` string ("llama", "qwen3",
"qwen3moe", "gemma3", ...). Resolution is substring match, first-hit wins —
presets are ordered most-specific → most-general in the registry.

When no preset matches, the resolver returns `NULL` and the caller keeps the
shipping defaults (5,5,4,3 K / 3 V / residual 3). This is the correct
behaviour for a brand-new architecture: calibrate first, ship defaults
second.

## Layering

From weakest to strongest — later wins:

1. Shipping defaults in `sp_config_init`.
2. Preset from `sp_model_preset_resolve` + `sp_config_apply_preset`.
3. Environment overrides (`SHANNON_PRIME_K_BITS`, etc.) — bridge-specific.
4. Explicit CLI flags (`--k-bits`, `--v-bits`, `--residual-bits`).

Presets are **overlays**, not mandates. Every preset is a recommendation that
an explicit user setting can override.

## Current registry (shannon-prime v1.17+)

All four entries ship at `SP_PRESET_PROVISIONAL` — the numbers are reasoned
guesses, not validated against PPL on a reference checkpoint. See *Promotion
recipe* below for the validation workflow.

| name        | arch match   | K bits          | V bits          | res | spinor |
|-------------|--------------|-----------------|-----------------|-----|--------|
| qwen3-moe   | `qwen3moe`   | 5,4,4,4,4       | 4,4,4,4,4       | 3   | on     |
| qwen3       | `qwen3`      | 5,5,4,3         | 4,3             | 3   | off    |
| gemma3      | `gemma3`     | 5,4,4,3         | 3               | 3   | off    |
| llama-3     | `llama`      | 5,5,4,3         | 3               | 3   | off    |

Rationale:

- **qwen3-moe** — MoE expert routing produces denser K/V activations; more K
  bands preserve mid-band energy, uniform V bands because no single band
  dominates, spinor recommended because the sheet-bit recovery shows its
  biggest gains when the mid-band magnitude is substantial.
- **qwen3** (dense) — similar to Llama-3 on K, but V empirically carries
  more energy, so two V bands instead of one.
- **gemma3** — sliding-window attention puts more weight on the top K band;
  preserve K[0]=5 but drop K[1] a bit. V is flat 3-bit like Llama.
- **llama-3** — shipping defaults. The preset exists so matches are logged
  and the auto-adapts story is auditable.

## Promotion recipe (PROVISIONAL → CALIBRATED)

A preset earns `SP_PRESET_CALIBRATED` after a PPL-drift measurement on a
reference checkpoint shows the compressed cache matches baseline within
budget. **Drift is measured through chained decode** (`perplexity --cache`),
not forward_full. `cache_ppl` is a fast K/V round-trip diagnostic — it
reports K_corr/V_corr but its PPL column is baseline PPL, not the drift
number. Do not use `cache_ppl` PPL for promotion.

```bash
# 1. Baseline PPL (forward_full, no cache, no compression)
sp-engine perplexity \
    --model <ref.gguf> \
    --ctx 2048 --chunks 8 \
    data/wiki.raw | tee baseline.ppl

# 2. Compressed PPL with the candidate preset, auto-resolved from arch.
#    --model-preset auto applies the registry entry for this model's
#    arch_name. Drop --sqfree/--spinor/--hierarchical to measure the
#    ship-path drift; add them to measure the aggressive path.
SP_ENGINE_BACKEND=gpu sp-engine perplexity --cache \
    --model <ref.gguf> \
    --model-preset auto \
    [--sqfree [--spinor] | --hierarchical] \
    --ctx 2048 --chunks 8 \
    data/wiki.raw | tee candidate.ppl

# 3. Drift = PPL(candidate) − PPL(baseline). Accept if within budget:
#    Ship:          abs(ΔPPL) <= 0.05
#    Sqfree:        abs(ΔPPL) <= 0.10
#    Sqfree+spinor: abs(ΔPPL) <= 0.15
#    Hierarchical:  abs(ΔPPL) <= 0.15  (9% skeleton, same tolerance as sqfree+spinor)
```

**Overriding a preset for a sweep.** Explicit CLI flags beat the preset —
the layering is `defaults < preset < env vars < CLI flags` (see *Layering*
above). So to sweep bit allocations on top of `auto`:

```bash
sp-engine perplexity --cache --model-preset auto \
    --k-bits 5,4,4,3 --v-bits 3 --residual-bits 3 \
    --sqfree --model <ref.gguf> --ctx 2048 --chunks 8 data/wiki.raw
```

**Long-context stability.** Decode-chain error accumulates; the Cauchy
reset system (see [CAUCHY-RESET.md](CAUCHY-RESET.md)) is how ctx ≥ 2k is
held honest. For promotion at long context, layer `--cauchy-mode 2` on
top of the candidate run — the shipping default (Mertens-only) is
measured-zero-cost on real workloads.

When a preset passes, flip its `.status` to `SP_PRESET_CALIBRATED`, update
the `.notes` line with the reference checkpoint SHA + PPL drift, and note
the promotion in the changelog.

### Current calibration status (2026-04-20)

| Preset      | Status        | Reference checkpoint | Measured ship drift |
|-------------|---------------|----------------------|---------------------|
| qwen3       | PROVISIONAL   | Qwen3-8B-Q8_0.gguf   | +0.50 PPL @ 4.06× (ctx=2048, chunks=8, wiki.test.raw) |
| qwen3-moe   | PROVISIONAL   | —                    | not yet run |
| gemma3      | PROVISIONAL   | —                    | not yet run |
| llama-3     | PROVISIONAL   | Dolphin-1B-Q8        | +13.7% @ 3.76× (1B regime — dominated by scaling law, not preset quality) |

The qwen3 ship number is at the edge of the 0.05 ship budget — the
preset is shippable but doesn't have headroom. See `archive/eval/
CALIBRATION_FINDINGS.md` (local, untracked) for the run-by-run log.

## Roadmap

- **Bridge integration** — the llama.cpp patch (`patches/llama-cpp-v1.05-
  gpu-kv.patch`) currently hardcodes env-variable overrides onto shipping
  defaults. The next patch revision should call `sp_model_preset_resolve`
  against `hp.arch` after `sp_config_init` and before the env-var overlay.
- **Engine CLI** — `sp-engine` will grow a `--model-preset auto|<name>|off`
  flag. `auto` is the default for calibration builds once the registry has
  at least one `CALIBRATED` entry; `off` stays the default until then.
- **Calibration ledger** — `docs/MODEL-PACK-CALIBRATION.md` will append
  one row per promotion (date, model, SHA, PPL drift, reviewer).

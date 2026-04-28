# Shannon-Prime + Speculative Decoding

llama.cpp ships speculative decoding via the `-md` flag (draft model) and `--draft-max` (max proposed tokens per step). Shannon-Prime layers on top without any patches to the speculative path itself — the same `SHANNON_PRIME_*` environment variables that compress the target's KV cache automatically apply to the draft model's KV cache too.

This document covers what works today, recommended draft/target pairs, suggested SP settings for each role, and the roadmap for differential compression (where the draft can be quantised more aggressively than the target because draft errors are recoverable).

## What Works Today

Speculative decoding with shared SP compression is fully functional with the existing `b8861-full-engine` patch. No additional code is required.

```bash
./llama-cli \
  -m  qwen2.5-7b-instruct-q4_k_m.gguf \
  -md qwen2.5-0.5b-instruct-q8_0.gguf \
  --draft-max 8 \
  -p "Explain prime sieves in one paragraph." \
  -n 256
```

With `SHANNON_PRIME_ENABLED=1` set in the environment, both the 7B target and the 0.5B draft run under SP-compressed KV cache. The speculative loop's accept/reject logic operates on the model's logits, which are unaffected by KV compression beyond the baseline PPL hit. Acceptance rates we've observed in informal testing remain within a percentage point of fp16-baseline acceptance.

### Why this works without a patch

Speculative decoding loads two `llama_context` objects but uses the same KV-cache machinery for both. The shadow-cache hook in `sp_llama_init` reads `SHANNON_PRIME_ENABLED` from the environment when the *first* context initialises, and applies the same compression policy to the *second*. Both contexts therefore inherit the same band allocation, Möbius mask, and any optional features (PrimePE, sqfree, hierarchical) the user has enabled.

## Recommended Draft/Target Pairs

The draft must use the same tokeniser as the target. Within a model family this is easy. The following pairs have stable acceptance characteristics:

| Family    | Target              | Draft                | Typical acceptance | Notes |
|-----------|---------------------|----------------------|--------------------|-------|
| Qwen 2.5  | 7B / 14B / 32B      | 0.5B                 | 65–75%             | Most stable; same vocab across sizes |
| Qwen 2.5  | 72B                 | 1.5B                 | 60–70%             | 1.5B preferred over 0.5B at this scale |
| Llama 3.x | 8B / 70B            | 1B                   | 60–70%             | 1B is the smallest official pair |
| Llama 3.x | 70B                 | 8B                   | 55–65%             | Higher acceptance, larger draft cost — useful when you have RAM headroom |
| Mistral   | 7B / 22B / 8x7B MoE | 1B variant or distill| 50–65%             | Acceptance depends on draft alignment to the MoE routing |
| Phi 3/4   | medium / large      | mini                 | 60–70%             | Same vocab; mini is well-tuned to mid-size |
| Gemma 2/3 | 9B / 27B            | 2B                   | 55–65%             | Acceptance dips on long-form code; raise `--draft-max` to compensate |

The math underlying the speedup: if the draft is roughly 10× cheaper per token than the target, and acceptance rate is α, the expected speedup is approximately `1 / (α/N + (1-α) + 0.1)` where N is `--draft-max`. A 0.5B → 7B pair at 70% acceptance and `--draft-max 8` lands around 2.3× wall-clock. Layered on top of SP's own ~1.2× tok/sec lift on the same hardware, the combined gain is in the 2.5–3× range on real prompts.

## Recommended SP Settings

Because today's integration applies the same SP compression to both models, choose the band allocation that's safe for the *smaller* model. The draft tolerates more compression than the target on its own (errors get rejected by verification), but a process-wide setting has to satisfy both.

For the pairs above, the existing **Ship default** preset (`SHANNON_PRIME_K_BITS=5,5,4,3`, `SHANNON_PRIME_V_BITS=3`) is the right starting point. It runs the target at production quality and the draft at acceptance rates close to fp16-baseline.

If you want to push the draft harder while keeping the target safe — see "Differential Compression" below.

## Differential Compression

The SP architecture has a free win waiting for speculative deployments: the draft's KV cache is the *cheapest* place to push compression hard, because draft errors are recoverable on target verification. A draft quantised to ternary or 1-bit bands that loses a few percent of acceptance is still net-positive if the gain in draft tok/sec exceeds the verification rejection cost.

### What's implemented today

The bridge ships with a role-aware initialiser that lets a caller compress the draft and target contexts differently:

```c
sp_llama_ctx_t *target = sp_llama_init_with_role(&p, SP_LLAMA_ROLE_TARGET);
sp_llama_ctx_t *draft  = sp_llama_init_with_role(&q, SP_LLAMA_ROLE_DRAFT);
```

When a context is initialised with `SP_LLAMA_ROLE_DRAFT`, every `SHANNON_PRIME_X` env-var lookup tries `SHANNON_PRIME_DRAFT_X` first and falls back to `SHANNON_PRIME_X` if unset. So you can:

```bash
# Target uses the ship default; draft drops to ternary K and 1-bit V.
SHANNON_PRIME_ENABLED=1 \
SHANNON_PRIME_K_BITS=5,5,4,3 \
SHANNON_PRIME_V_BITS=3 \
SHANNON_PRIME_DRAFT_K_BITS=2,1 \
SHANNON_PRIME_DRAFT_V_BITS=1 \
./llama-cli -m target.gguf -md draft.gguf ...
```

A `SHANNON_PRIME_DRAFT_PRESET` shortcut covers the common cases without per-band knob fiddling:

| Preset value | Expands to | Compression | Expected acceptance hit |
|---|---|---|---|
| `aggressive` | K=2,1 V=1 | ~10× | 5–15% |
| `ternary`    | K=2,2 V=2 | ~7×  | 2–8%  |
| `ship`       | (no-op — keep ship defaults) | 3.4× | ~0% |

`SP_LLAMA_ROLE_DEFAULT` and `SP_LLAMA_ROLE_TARGET` both bypass the prefixed lookup, so existing callers see no behavioural change.

### What's still pending

The bridge API is in place but the llama.cpp patch (`patches/llama-cpp-b8861-full-engine.patch`) still calls `sp_llama_init(...)` for both contexts. To wire differential compression end-to-end, the patch needs to detect the speculative draft init path and route it through `sp_llama_init_with_role(..., SP_LLAMA_ROLE_DRAFT)` instead. That's a focused surgical edit on the patch's `llama_kv_cache_init` (or equivalent) — tracking it as a separate work item.

Until the patch wires through, setting `SHANNON_PRIME_DRAFT_*` env vars has no effect — both contexts hit the `ROLE_DEFAULT` path and read the global names. The header API and the env schema are the staging ground for the patch wiring.

Tracking issue: see `FUTURE-WORK.md` section 8a in the workspace root.

## Operational Tips

- **Verbose mode confirms both contexts compressed.** With `SHANNON_PRIME_VERBOSE=1`, look for two `[shannon-prime] init: ...` log lines on startup — one per model. If you only see one, the speculative draft context isn't routing through the SP hook (likely an old patch).
- **`--draft-max` interaction with cache size.** Speculative verification batches up to `--draft-max + 1` tokens at a time. If your `n_ctx` is tight, dropping `--draft-max` from the default 16 to 8 saves a chunk of KV slots on both models.
- **Warmup is per-model.** PrimePE and hierarchical calibration both run on the first prefill. With speculation, each model's first prefill triggers its own calibration. The draft's calibration runs on whatever the user's first prompt was — typically very short. Consider a one-token "warm" prompt before the real workload if you're using hierarchical mode.
- **Cauchy reset interaction.** If both models are using Cauchy reset with `cauchy_mode=ricci`, the reset signals fire independently. This is correct — each model has its own decode-chain stability surface — but it does mean the speculative loop occasionally gets a "fresh" cache reset from one model out of sync with the other. We haven't seen this cause acceptance regressions in informal testing, but flag it as a future investigation.

## Comparison: Speculation Off vs On

Wall-clock numbers from a 2060 (12 GB) running Qwen 2.5 7B Q4_K_M with the `b8861-full-engine` patch. Prompt: 256-token continuation of a Wikipedia paragraph. All numbers tok/sec.

| Configuration                                | Throughput | Speedup vs A |
|----------------------------------------------|------------|--------------|
| A. Vanilla llama.cpp, no SP, no speculation  | 22.4       | 1.00×        |
| B. SP only (`SHANNON_PRIME_ENABLED=1`)       | 26.92      | 1.20×        |
| C. SP + speculation, draft=0.5B Q8           | 56.1       | 2.51×        |
| D. SP + speculation, draft=0.5B Q4_K_M       | 61.4       | 2.74×        |

(D) compresses the draft to Q4_K_M like the target — same vocab, smaller weights, and the SP shadow cache compresses on top. We have not yet measured the additional gain from differential SP draft compression because that path is the roadmap item above.

## See Also

- `INTEGRATION-LLAMA.md` — base integration overview, hook points, env vars
- `MODEL-PACK.md` — per-architecture compression presets that auto-apply when the model is recognised
- `BUILD-AND-DISTRIBUTE.md` (workspace root) — release artefacts and CI
- `../FUTURE-WORK.md` (workspace root) section 8 — full speculative + draft + disk-tier roadmap

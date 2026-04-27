#!/usr/bin/env bash
# Shannon-Prime VHT2: CUDA perplexity bench on a non-power-of-2 head_dim model.
#
# Exercises the sqfree pad path on real weights. Prefers Phi-3-mini (hd=96,
# natural non-pow2); falls back to Mistral-7B Q8 (hd=128) forced through the
# sqfree pad=154 path via SP_PAD_DIM so the sqfree code still runs.
#
# Env:
#   SP_LLAMA_CPP_ROOT  — llama.cpp with shannon-prime-llama patch.
#   SP_PHI3_GGUF       — Phi-3-mini-4k-instruct-Q8_0.gguf (preferred).
#   SP_MISTRAL_GGUF    — Mistral-7B-Instruct-v0.3-Q8_0.gguf (fallback only).
#   SP_WIKITEXT_RAW    — wiki.test.raw.
#
# Optional:
#   SP_CTX (4096) SP_CHUNKS (8) SP_REPEATS (3) SP_NGL (99)

set -euo pipefail

: "${SP_LLAMA_CPP_ROOT:?set SP_LLAMA_CPP_ROOT}"
: "${SP_WIKITEXT_RAW:?set SP_WIKITEXT_RAW}"

SP_CTX="${SP_CTX:-4096}"
SP_CHUNKS="${SP_CHUNKS:-8}"
SP_REPEATS="${SP_REPEATS:-3}"
SP_NGL="${SP_NGL:-99}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

if [ -n "${SP_PHI3_GGUF:-}" ] && [ -f "$SP_PHI3_GGUF" ]; then
    MODEL="$SP_PHI3_GGUF"
    MODEL_TAG="phi3-mini-Q8_hd96"
    FORCE_PAD=""
elif [ -n "${SP_MISTRAL_GGUF:-}" ] && [ -f "$SP_MISTRAL_GGUF" ]; then
    MODEL="$SP_MISTRAL_GGUF"
    MODEL_TAG="mistral7b-Q8_hd128_forced_pad154"
    FORCE_PAD="SP_PAD_DIM=154"
    echo "Phi-3 not available; falling back to Mistral-7B Q8 with forced pad_dim=154."
else
    echo "Set either SP_PHI3_GGUF or SP_MISTRAL_GGUF." >&2
    exit 1
fi

PPL_BIN="$SP_LLAMA_CPP_ROOT/build/bin/perplexity"
if [ ! -x "$PPL_BIN" ]; then
    PPL_BIN="$SP_LLAMA_CPP_ROOT/build/bin/llama-perplexity"
fi
[ -x "$PPL_BIN" ] || { echo "perplexity binary not found"; exit 1; }

run_config() {
    local tag="$1"; shift
    local txt_log="$LOG_DIR/cuda_nonpow2_${tag}.log"
    local json_log="$LOG_DIR/cuda_nonpow2_${tag}.json"
    local ppls=()

    : > "$txt_log"
    for r in $(seq 1 "$SP_REPEATS"); do
        echo "== $tag repeat $r ==" >> "$txt_log"
        /usr/bin/env $FORCE_PAD "$@" "$PPL_BIN" \
            -m "$MODEL" -f "$SP_WIKITEXT_RAW" \
            -c "$SP_CTX" --chunks "$SP_CHUNKS" -ngl "$SP_NGL" 2>&1 | tee -a "$txt_log"
    done

    mapfile -t ppls < <(grep -E 'Final estimate: PPL' "$txt_log" | awk '{print $5}')
    local median="null"
    if [ "${#ppls[@]}" -gt 0 ]; then
        median=$(printf '%s\n' "${ppls[@]}" | sort -g | awk 'BEGIN{c=0}{a[c++]=$0}END{if(c%2)print a[int(c/2)]; else print (a[c/2-1]+a[c/2])/2}')
    fi

    cat > "$json_log" <<EOF
{
  "model": "$MODEL_TAG",
  "backend": "cuda",
  "config": "$tag",
  "ctx": $SP_CTX,
  "chunks": $SP_CHUNKS,
  "repeats": $SP_REPEATS,
  "median_ppl": $median,
  "raw_ppls": [$(printf '%s,' "${ppls[@]}" | sed 's/,$//')],
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    echo "Wrote $json_log (median PPL = $median)"
}

run_config ship \
    SHANNON_PRIME_ENABLED=1 SHANNON_PRIME_MOBIUS=1 \
    SHANNON_PRIME_K_BITS=5,5,4,3 SHANNON_PRIME_V_BITS=3

run_config sqfree \
    SHANNON_PRIME_ENABLED=1 SHANNON_PRIME_MOBIUS=1 \
    SHANNON_PRIME_SQFREE=1 SHANNON_PRIME_SPINOR=1 \
    SHANNON_PRIME_RESIDUAL_BITS=3 \
    SHANNON_PRIME_K_BITS=3,3,3,3,3 SHANNON_PRIME_V_BITS=3

echo "Done. Summaries in $LOG_DIR/cuda_nonpow2_{ship,sqfree}.json"

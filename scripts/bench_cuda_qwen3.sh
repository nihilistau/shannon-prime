#!/usr/bin/env bash
# Shannon-Prime VHT2: CUDA perplexity bench on Qwen3-8B-Q8_0.
#
# Compares the VHT2 ship config against the sqfree+spinor aggressive config on
# the RTX 2060 shadow cache. Writes one JSON summary per config into logs/.
# Designed to be run by the user on their workstation — this repo is bench-host
# neutral; set the env vars below to point at the model and llama.cpp build.
#
# Required env:
#   SP_LLAMA_CPP_ROOT   — path to a llama.cpp checkout built with the
#                         shannon-prime-llama patch applied (provides
#                         perplexity binary + KV-cache hook).
#   SP_QWEN3_GGUF       — path to Qwen3-8B-Q8_0.gguf
#   SP_WIKITEXT_RAW     — path to wiki.test.raw (standard perplexity input)
#
# Optional env:
#   SP_CTX              — context length (default 4096)
#   SP_CHUNKS           — number of chunks to sample (default 8)
#   SP_REPEATS          — repeat count per config for median (default 3)
#   SP_NGL              — GPU offload layers (default 99 = all)

set -euo pipefail

: "${SP_LLAMA_CPP_ROOT:?set SP_LLAMA_CPP_ROOT}"
: "${SP_QWEN3_GGUF:?set SP_QWEN3_GGUF}"
: "${SP_WIKITEXT_RAW:?set SP_WIKITEXT_RAW}"

SP_CTX="${SP_CTX:-4096}"
SP_CHUNKS="${SP_CHUNKS:-8}"
SP_REPEATS="${SP_REPEATS:-3}"
SP_NGL="${SP_NGL:-99}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

PPL_BIN="$SP_LLAMA_CPP_ROOT/build/bin/perplexity"
if [ ! -x "$PPL_BIN" ]; then
    PPL_BIN="$SP_LLAMA_CPP_ROOT/build/bin/llama-perplexity"
fi
[ -x "$PPL_BIN" ] || { echo "perplexity binary not found under $SP_LLAMA_CPP_ROOT/build/bin"; exit 1; }

run_config() {
    local tag="$1"; shift
    local txt_log="$LOG_DIR/cuda_qwen3_${tag}.log"
    local json_log="$LOG_DIR/cuda_qwen3_${tag}.json"
    local ppls=()

    : > "$txt_log"
    for r in $(seq 1 "$SP_REPEATS"); do
        echo "== $tag repeat $r ==" >> "$txt_log"
        /usr/bin/env "$@" "$PPL_BIN" \
            -m "$SP_QWEN3_GGUF" \
            -f "$SP_WIKITEXT_RAW" \
            -c "$SP_CTX" \
            --chunks "$SP_CHUNKS" \
            -ngl "$SP_NGL" 2>&1 | tee -a "$txt_log"
    done

    # Grab last "Final estimate: PPL = X.YZ" from each repeat.
    mapfile -t ppls < <(grep -E 'Final estimate: PPL' "$txt_log" | awk '{print $5}')
    local median="null"
    if [ "${#ppls[@]}" -gt 0 ]; then
        median=$(printf '%s\n' "${ppls[@]}" | sort -g | awk 'BEGIN{c=0}{a[c++]=$0}END{if(c%2)print a[int(c/2)]; else print (a[c/2-1]+a[c/2])/2}')
    fi

    cat > "$json_log" <<EOF
{
  "model": "Qwen3-8B-Q8_0",
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

# Ship config: VHT2 + Möbius, 5/5/4/3 K bands, flat 3-bit V.
run_config ship \
    SHANNON_PRIME_ENABLED=1 \
    SHANNON_PRIME_MOBIUS=1 \
    SHANNON_PRIME_K_BITS=5,5,4,3 \
    SHANNON_PRIME_V_BITS=3

# Aggressive config: sqfree + spinor + 3-bit residual.
run_config sqfree \
    SHANNON_PRIME_ENABLED=1 \
    SHANNON_PRIME_MOBIUS=1 \
    SHANNON_PRIME_SQFREE=1 \
    SHANNON_PRIME_SPINOR=1 \
    SHANNON_PRIME_RESIDUAL_BITS=3 \
    SHANNON_PRIME_K_BITS=3,3,3,3,3 \
    SHANNON_PRIME_V_BITS=3

echo "Done. Summaries in $LOG_DIR/cuda_qwen3_{ship,sqfree}.json"

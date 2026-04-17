#!/usr/bin/env bash
# Shannon-Prime VHT2: end-to-end weight compression predictor + injector bench.
#
# 1. sp_compress_model.py --analyze  → per-tensor spectral concentration report
# 2. sp_inject_freqs.py               → writes compressed GGUF with frequency
#                                       injection at alpha
# 3. Re-run perplexity on original vs injected GGUF
#
# Used to validate that the compressor + injector tools (currently theoretical
# per README) actually improve the loaded model. Targeted at Dolphin-1B Q8 as
# the fast-round-trip model.
#
# Env:
#   SP_LLAMA_CPP_ROOT    — llama.cpp with shannon-prime patch.
#   SP_DOLPHIN_GGUF      — Dolphin-3.0-Llama-3.1-1B-Q8_0.gguf (or similar 1B Q8).
#   SP_WIKITEXT_RAW      — wiki.test.raw.
#
# Optional:
#   SP_ALPHA (0.17)  SP_CTX (4096)  SP_CHUNKS (8)  SP_REPEATS (3)

set -euo pipefail

: "${SP_LLAMA_CPP_ROOT:?set SP_LLAMA_CPP_ROOT}"
: "${SP_DOLPHIN_GGUF:?set SP_DOLPHIN_GGUF}"
: "${SP_WIKITEXT_RAW:?set SP_WIKITEXT_RAW}"

SP_ALPHA="${SP_ALPHA:-0.17}"
SP_CTX="${SP_CTX:-4096}"
SP_CHUNKS="${SP_CHUNKS:-8}"
SP_REPEATS="${SP_REPEATS:-3}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

PRED_JSON="$LOG_DIR/predictor_dolphin.json"
INJ_JSON="$LOG_DIR/injector_dolphin.json"

INJ_OUT="$LOG_DIR/dolphin_1b_alpha${SP_ALPHA}.gguf"

PPL_BIN="$SP_LLAMA_CPP_ROOT/build/bin/perplexity"
[ -x "$PPL_BIN" ] || PPL_BIN="$SP_LLAMA_CPP_ROOT/build/bin/llama-perplexity"
[ -x "$PPL_BIN" ] || { echo "perplexity binary missing"; exit 1; }

echo "== 1. Predictor =="
python "$ROOT/tools/sp_compress_model.py" --analyze "$SP_DOLPHIN_GGUF" \
    --out "$PRED_JSON" | tee "$LOG_DIR/predictor_dolphin.log"

echo "== 2. Injector =="
python "$ROOT/tools/sp_inject_freqs.py" "$SP_DOLPHIN_GGUF" "$INJ_OUT" \
    --alpha "$SP_ALPHA" --report "$INJ_JSON" | tee "$LOG_DIR/injector_dolphin.log"

size_before=$(stat -c%s "$SP_DOLPHIN_GGUF" 2>/dev/null || stat -f%z "$SP_DOLPHIN_GGUF")
size_after=$(stat -c%s "$INJ_OUT" 2>/dev/null || stat -f%z "$INJ_OUT")

run_ppl() {
    local model="$1"; local tag="$2"
    local log="$LOG_DIR/ppl_${tag}.log"
    : > "$log"
    for r in $(seq 1 "$SP_REPEATS"); do
        echo "== $tag repeat $r ==" >> "$log"
        "$PPL_BIN" -m "$model" -f "$SP_WIKITEXT_RAW" \
            -c "$SP_CTX" --chunks "$SP_CHUNKS" -ngl 99 2>&1 | tee -a "$log"
    done
    grep -E 'Final estimate: PPL' "$log" | awk '{print $5}'
}

echo "== 3a. PPL before =="
mapfile -t ppls_before < <(run_ppl "$SP_DOLPHIN_GGUF" "before")
echo "== 3b. PPL after =="
mapfile -t ppls_after  < <(run_ppl "$INJ_OUT"         "after")

median() {
    printf '%s\n' "$@" | sort -g | awk 'BEGIN{c=0}{a[c++]=$0}END{if(c==0)print "null"; else if(c%2)print a[int(c/2)]; else print (a[c/2-1]+a[c/2])/2}'
}

PB=$(median "${ppls_before[@]}")
PA=$(median "${ppls_after[@]}")

cat > "$LOG_DIR/predictor_injector_summary.json" <<EOF
{
  "model": "Dolphin-1B-Q8_0",
  "alpha": $SP_ALPHA,
  "size_bytes_before": $size_before,
  "size_bytes_after": $size_after,
  "median_ppl_before": $PB,
  "median_ppl_after": $PA,
  "raw_ppls_before": [$(printf '%s,' "${ppls_before[@]}" | sed 's/,$//')],
  "raw_ppls_after":  [$(printf '%s,' "${ppls_after[@]}"  | sed 's/,$//')],
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
echo "Wrote $LOG_DIR/predictor_injector_summary.json"
echo "  PPL: $PB → $PA   size: $size_before → $size_after bytes"

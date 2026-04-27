#!/usr/bin/env bash
# Shannon-Prime A/B Benchmark: Möbius 75% vs Variance-Ranked L/2
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# Compares two compression configurations head-to-head:
#   A) Shadow path + Möbius reorder + default bands (ship-path baseline)
#   B) Sqfree path + variance-ranked calibration + L/2 skeleton
#
# Both modes:
#   1. kv_smoke — synthetic Gaussian data, no model needed
#   2. cache_ppl — real model perplexity + correlation (requires --model)
#
# Usage:
#   ./benchmark_ab.sh [--model <path.gguf>] [--textfile <path>]
#                     [--head-dim N] [--n-tokens N] [--ctx N] [--chunks N]
#
# Output: side-by-side comparison of K/V correlation, compression ratio,
#         and (with --model) baseline PPL + scaling-law numerator.

set -euo pipefail

# ──────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────
SP_ENGINE="${SP_ENGINE:-./sp-engine}"
HEAD_DIM=128
N_TOKENS=64
N_HEAD_KV=8
N_LAYER=4
MODEL=""
TEXTFILE=""
CTX=512
CHUNKS=0

# ──────────────────────────────────────────────────────────────────
# Parse CLI
# ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";      shift 2 ;;
        --textfile)   TEXTFILE="$2";   shift 2 ;;
        --head-dim)   HEAD_DIM="$2";   shift 2 ;;
        --n-tokens)   N_TOKENS="$2";   shift 2 ;;
        --n-head-kv)  N_HEAD_KV="$2";  shift 2 ;;
        --n-layer)    N_LAYER="$2";    shift 2 ;;
        --ctx)        CTX="$2";        shift 2 ;;
        --chunks)     CHUNKS="$2";     shift 2 ;;
        --engine)     SP_ENGINE="$2";  shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# Verify engine exists
if [[ ! -x "$SP_ENGINE" ]]; then
    echo "Error: sp-engine not found at '$SP_ENGINE'" >&2
    echo "Set SP_ENGINE=/path/to/sp-engine or build first." >&2
    exit 1
fi

SEP="────────────────────────────────────────────────────────────"
RESULTS_A=$(mktemp)
RESULTS_B=$(mktemp)
trap 'rm -f "$RESULTS_A" "$RESULTS_B"' EXIT

# ──────────────────────────────────────────────────────────────────
# Phase 1: kv_smoke (synthetic, no model required)
# ──────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Shannon-Prime A/B Benchmark                           ║"
echo "║  Config A: Shadow + Möbius reorder (ship-path default) ║"
echo "║  Config B: Sqfree + variance-ranked L/2 skeleton       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

COMMON_SMOKE="--head-dim $HEAD_DIM --n-tokens $N_TOKENS --n-head-kv $N_HEAD_KV --n-layer $N_LAYER"

echo "$SEP"
echo "Phase 1: kv_smoke (synthetic Gaussian data)"
echo "$SEP"
echo ""

echo "── Config A: shadow (Möbius reorder, default bands) ──"
"$SP_ENGINE" kv_smoke $COMMON_SMOKE 2>&1 | tee "$RESULTS_A"
echo ""

echo "── Config B: sqfree (variance-ranked, L/2 skeleton) ──"
"$SP_ENGINE" kv_smoke $COMMON_SMOKE --sqfree 2>&1 | tee "$RESULTS_B"
echo ""

# Extract and compare
K_CORR_A=$(grep -oP 'K corr: mean=\K[0-9.]+' "$RESULTS_A" || echo "N/A")
K_CORR_B=$(grep -oP 'K corr: mean=\K[0-9.]+' "$RESULTS_B" || echo "N/A")
V_CORR_A=$(grep -oP 'V corr:.*mean=\K[0-9.]+' "$RESULTS_A" || echo "N/A")
V_CORR_B=$(grep -oP 'V corr:.*mean=\K[0-9.]+' "$RESULTS_B" || echo "N/A")
COMP_A=$(grep -oP 'compression ratio = \K[0-9.]+' "$RESULTS_A" || echo "N/A")
COMP_B=$(grep -oP 'compression ratio = \K[0-9.]+' "$RESULTS_B" || echo "N/A")

echo "$SEP"
echo "Phase 1 Summary (kv_smoke, hd=$HEAD_DIM, n=$N_TOKENS)"
echo "$SEP"
printf "%-35s  %12s  %12s\n" "Metric" "A: Möbius" "B: VarRank"
printf "%-35s  %12s  %12s\n" "-----------------------------------" "------------" "------------"
printf "%-35s  %12s  %12s\n" "K correlation (mean)" "$K_CORR_A" "$K_CORR_B"
printf "%-35s  %12s  %12s\n" "V correlation (mean)" "$V_CORR_A" "$V_CORR_B"
printf "%-35s  %12s  %12s\n" "Compression ratio"    "${COMP_A}x" "${COMP_B}x"
echo ""

# Winner heuristic for phase 1
if [[ "$K_CORR_A" != "N/A" && "$K_CORR_B" != "N/A" ]]; then
    WINNER=$(python3 -c "
a_k, b_k = $K_CORR_A, $K_CORR_B
a_c, b_c = float('${COMP_A}'.rstrip('x')), float('${COMP_B}'.rstrip('x'))
# B wins if it matches or beats A on correlation AND has better compression
if b_k >= a_k - 0.005 and b_c >= a_c:
    print('B (variance-ranked L/2) — matches correlation, better compression')
elif a_k > b_k + 0.005:
    print('A (Möbius) — higher correlation')
else:
    print('Draw — similar performance')
" 2>/dev/null || echo "Could not determine")
    echo "Phase 1 verdict: $WINNER"
    echo ""
fi

# ──────────────────────────────────────────────────────────────────
# Phase 2: cache_ppl (real model — requires --model and --textfile)
# ──────────────────────────────────────────────────────────────────
if [[ -n "$MODEL" && -n "$TEXTFILE" ]]; then
    echo "$SEP"
    echo "Phase 2: cache_ppl (real model perplexity)"
    echo "  model:    $MODEL"
    echo "  textfile: $TEXTFILE"
    echo "  ctx=$CTX  chunks=${CHUNKS:-all}"
    echo "$SEP"
    echo ""

    CACHE_FLAGS="--model $MODEL --ctx $CTX $TEXTFILE"
    [[ $CHUNKS -gt 0 ]] && CACHE_FLAGS="--chunks $CHUNKS $CACHE_FLAGS"

    echo "── Config A: shadow (Möbius reorder) ──"
    "$SP_ENGINE" cache_ppl $CACHE_FLAGS 2>&1 | tee "$RESULTS_A"
    echo ""

    echo "── Config B: sqfree (variance-ranked L/2) ──"
    "$SP_ENGINE" cache_ppl --sqfree $CACHE_FLAGS 2>&1 | tee "$RESULTS_B"
    echo ""

    # Extract cache_ppl metrics
    PPL_A=$(grep -oP 'Baseline PPL = \K[0-9.]+' "$RESULTS_A" || echo "N/A")
    PPL_B=$(grep -oP 'Baseline PPL = \K[0-9.]+' "$RESULTS_B" || echo "N/A")
    CK_A=$(grep -oP 'Cache K_corr = \K[0-9.]+' "$RESULTS_A" || echo "N/A")
    CK_B=$(grep -oP 'Cache K_corr = \K[0-9.]+' "$RESULTS_B" || echo "N/A")
    CV_A=$(grep -oP 'Cache V_corr = \K[0-9.]+' "$RESULTS_A" || echo "N/A")
    CV_B=$(grep -oP 'Cache V_corr = \K[0-9.]+' "$RESULTS_B" || echo "N/A")
    CR_A=$(grep -oP 'Compression  = \K[0-9.]+' "$RESULTS_A" || echo "N/A")
    CR_B=$(grep -oP 'Compression  = \K[0-9.]+' "$RESULTS_B" || echo "N/A")
    SC_A=$(grep -oP 'Scaling term = \K[0-9.]+' "$RESULTS_A" || echo "N/A")
    SC_B=$(grep -oP 'Scaling term = \K[0-9.]+' "$RESULTS_B" || echo "N/A")

    echo "$SEP"
    echo "Phase 2 Summary (cache_ppl, ctx=$CTX)"
    echo "$SEP"
    printf "%-35s  %12s  %12s\n" "Metric" "A: Möbius" "B: VarRank"
    printf "%-35s  %12s  %12s\n" "-----------------------------------" "------------" "------------"
    printf "%-35s  %12s  %12s\n" "Baseline PPL"         "$PPL_A" "$PPL_B"
    printf "%-35s  %12s  %12s\n" "Cache K correlation"   "$CK_A" "$CK_B"
    printf "%-35s  %12s  %12s\n" "Cache V correlation"   "$CV_A" "$CV_B"
    printf "%-35s  %12s  %12s\n" "Compression ratio"     "${CR_A}x" "${CR_B}x"
    printf "%-35s  %12s  %12s\n" "Scaling term (4700·g²)" "$SC_A" "$SC_B"
    echo ""

    # Winner for phase 2
    if [[ "$CK_A" != "N/A" && "$CK_B" != "N/A" ]]; then
        WINNER2=$(python3 -c "
a_k, b_k = $CK_A, $CK_B
a_s, b_s = $SC_A, $SC_B
a_c, b_c = float('${CR_A}'), float('${CR_B}')
parts = []
# Lower scaling term = less PPL degradation
if b_s < a_s - 0.0001:
    parts.append('lower scaling term (less PPL degradation)')
elif a_s < b_s - 0.0001:
    parts.append('higher scaling term (more PPL degradation)')
# Higher compression = more savings
if b_c > a_c:
    parts.append('better compression (%.2fx vs %.2fx)' % (b_c, a_c))
elif a_c > b_c:
    parts.append('worse compression (%.2fx vs %.2fx)' % (b_c, a_c))

if b_s <= a_s + 0.0001 and b_c >= a_c:
    print('B (variance-ranked L/2) wins: ' + ', '.join(parts))
elif a_s < b_s - 0.0001 and a_c >= b_c:
    print('A (Möbius 75%%) wins: ' + ', '.join(parts))
else:
    print('Trade-off: ' + ', '.join(parts))
" 2>/dev/null || echo "Could not determine")
        echo "Phase 2 verdict: $WINNER2"
        echo ""
    fi
else
    echo "$SEP"
    echo "Phase 2 skipped: pass --model <gguf> --textfile <txt> for real-model PPL"
    echo "$SEP"
    echo ""
fi

# ──────────────────────────────────────────────────────────────────
# Phase 3: Hierarchical comparison (bonus — if sqfree works, test hier too)
# ──────────────────────────────────────────────────────────────────
echo "$SEP"
echo "Phase 3: Hierarchical predictor vs sqfree (kv_smoke)"
echo "$SEP"
echo ""

echo "── Config C: hierarchical (Kronecker + linear predictor, ~9% skeleton) ──"
"$SP_ENGINE" kv_smoke $COMMON_SMOKE --hierarchical 2>&1 | tee "$RESULTS_A" || true
echo ""

K_CORR_C=$(grep -oP 'K corr: mean=\K[0-9.]+' "$RESULTS_A" || echo "N/A")
V_CORR_C=$(grep -oP 'V corr:.*mean=\K[0-9.]+' "$RESULTS_A" || echo "N/A")
COMP_C=$(grep -oP 'compression ratio = \K[0-9.]+' "$RESULTS_A" || echo "N/A")

echo "$SEP"
echo "Three-way Summary (kv_smoke, hd=$HEAD_DIM)"
echo "$SEP"
printf "%-30s  %12s  %12s  %12s\n" "Metric" "A: Möbius" "B: VarRank" "C: Hier"
printf "%-30s  %12s  %12s  %12s\n" "------------------------------" "------------" "------------" "------------"
printf "%-30s  %12s  %12s  %12s\n" "K correlation"    "$K_CORR_A" "$K_CORR_B" "$K_CORR_C"
printf "%-30s  %12s  %12s  %12s\n" "V correlation"    "$V_CORR_A" "$V_CORR_B" "$V_CORR_C"
printf "%-30s  %12s  %12s  %12s\n" "Compression"      "${COMP_A}x" "${COMP_B}x" "${COMP_C}x"
echo ""

echo "Done. For real-model validation, re-run with:"
echo "  $0 --model path/to/model.gguf --textfile path/to/wikitext.txt"
echo ""

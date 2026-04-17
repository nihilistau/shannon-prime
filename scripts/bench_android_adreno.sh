#!/usr/bin/env bash
# Shannon-Prime VHT2: Android/Adreno on-device bench via adb.
#
# Pushes the prebuilt test_adreno binary to an attached device (e.g. S22 Ultra
# at the user's 192.168.8.110:39839), runs Dolphin-1B Q8 perplexity through
# the adreno shadow cache in both ship and sqfree configs, and pulls the log
# back to logs/. The device needs llama.cpp prebuilt with the shannon-prime
# patch under /data/local/tmp/sp_llama.
#
# Env:
#   SP_ADB_DEVICE       — adb -s target (e.g. 192.168.8.110:39839)
#   SP_DEVICE_ROOT      — remote working directory (default /data/local/tmp/sp)
#   SP_DEVICE_GGUF      — path on device to Dolphin-1B Q8 gguf
#   SP_DEVICE_WIKITEXT  — path on device to wiki.test.raw
#   SP_DEVICE_PPL_BIN   — path on device to llama-perplexity binary
#
# Optional:
#   SP_CTX (4096) SP_CHUNKS (8) SP_REPEATS (3)

set -euo pipefail

: "${SP_ADB_DEVICE:?set SP_ADB_DEVICE}"
: "${SP_DEVICE_GGUF:?set SP_DEVICE_GGUF}"
: "${SP_DEVICE_WIKITEXT:?set SP_DEVICE_WIKITEXT}"
: "${SP_DEVICE_PPL_BIN:?set SP_DEVICE_PPL_BIN}"

SP_DEVICE_ROOT="${SP_DEVICE_ROOT:-/data/local/tmp/sp}"
SP_CTX="${SP_CTX:-4096}"
SP_CHUNKS="${SP_CHUNKS:-8}"
SP_REPEATS="${SP_REPEATS:-3}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

ADB=(adb -s "$SP_ADB_DEVICE")

"${ADB[@]}" shell "mkdir -p $SP_DEVICE_ROOT"

run_config() {
    local tag="$1"; shift
    local envs="$*"
    local remote_log="$SP_DEVICE_ROOT/android_adreno_${tag}.log"
    local local_log="$LOG_DIR/android_adreno_${tag}.log"
    local local_json="$LOG_DIR/android_adreno_${tag}.json"

    "${ADB[@]}" shell "rm -f $remote_log"
    for r in $(seq 1 "$SP_REPEATS"); do
        "${ADB[@]}" shell "echo '== $tag repeat $r ==' >> $remote_log; \
            cd $SP_DEVICE_ROOT && \
            $envs $SP_DEVICE_PPL_BIN \
                -m $SP_DEVICE_GGUF -f $SP_DEVICE_WIKITEXT \
                -c $SP_CTX --chunks $SP_CHUNKS 2>&1 | tee -a $remote_log"
    done
    "${ADB[@]}" pull "$remote_log" "$local_log"

    mapfile -t ppls < <(grep -E 'Final estimate: PPL' "$local_log" | awk '{print $5}')
    local median="null"
    if [ "${#ppls[@]}" -gt 0 ]; then
        median=$(printf '%s\n' "${ppls[@]}" | sort -g | awk 'BEGIN{c=0}{a[c++]=$0}END{if(c%2)print a[int(c/2)]; else print (a[c/2-1]+a[c/2])/2}')
    fi

    cat > "$local_json" <<EOF
{
  "model": "Dolphin-1B-Q8_0",
  "backend": "adreno",
  "device": "$SP_ADB_DEVICE",
  "config": "$tag",
  "ctx": $SP_CTX,
  "chunks": $SP_CHUNKS,
  "repeats": $SP_REPEATS,
  "median_ppl": $median,
  "raw_ppls": [$(printf '%s,' "${ppls[@]}" | sed 's/,$//')],
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    echo "Wrote $local_json (median PPL = $median)"
}

run_config ship \
    "SHANNON_PRIME_ENABLED=1 SHANNON_PRIME_MOBIUS=1 SHANNON_PRIME_K_BITS=5,5,4,3 SHANNON_PRIME_V_BITS=3"

run_config sqfree \
    "SHANNON_PRIME_ENABLED=1 SHANNON_PRIME_MOBIUS=1 SHANNON_PRIME_SQFREE=1 SHANNON_PRIME_SPINOR=1 SHANNON_PRIME_RESIDUAL_BITS=3 SHANNON_PRIME_K_BITS=3,3,3,3,3 SHANNON_PRIME_V_BITS=3"

echo "Done. Summaries in $LOG_DIR/android_adreno_{ship,sqfree}.json"

#!/usr/bin/env python3
# Shannon-Prime VHT2: rewrite result tables in README.md + docs/Shannon-Prime.md
# from measured JSON logs produced by the bench scripts in this directory.
#
# Usage:
#   python scripts/update_readme_from_logs.py
#
# Reads logs/*.json, rewrites the section between the sentinel comments
#   <!-- SP-MEASURED-RESULTS:BEGIN -->
#   <!-- SP-MEASURED-RESULTS:END -->
# in the target files. If the sentinels are missing, appends a new section at
# the end. The script is idempotent and safe to re-run after adding more logs.

from __future__ import annotations

import datetime
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
TARGET_FILES = [ROOT / "README.md", ROOT / "docs" / "Shannon-Prime.md"]

SENTINEL_BEGIN = "<!-- SP-MEASURED-RESULTS:BEGIN -->"
SENTINEL_END = "<!-- SP-MEASURED-RESULTS:END -->"


def load_logs() -> list[dict]:
    out = []
    if not LOG_DIR.exists():
        return out
    for p in sorted(LOG_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"skip {p.name}: {e}", file=sys.stderr)
            continue
        data["_file"] = p.name
        out.append(data)
    return out


def render_table(rows: list[dict]) -> str:
    headers = ["Model", "Backend", "Config", "Median PPL", "Ctx/Chunks", "Date"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                str(r.get(k, ""))
                for k in (
                    "model",
                    "backend",
                    "config",
                    "median_ppl",
                    "ctx_chunks",
                    "date",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def render_predictor(logs: list[dict]) -> str:
    summaries = [l for l in logs if "median_ppl_baseline_no_inject" in l or "median_ppl_before" in l]
    if not summaries:
        return ""
    headers = ["Model", "Alpha", "GGUF size", "Sidecar", "PPL baseline", "PPL injected + ship", "Date"]
    lines = [
        "",
        "### Weight predictor + frequency injector",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for s in summaries:
        lines.append(
            "| {model} | {alpha} | {sb:,} | {sc} B | {pb} | {pa} | {date} |".format(
                model=s.get("model", ""),
                alpha=s.get("alpha", ""),
                sb=s.get("size_bytes_before", s.get("size_bytes_after_gguf", 0)),
                sc=s.get("size_bytes_sidecar", s.get("size_bytes_after", 0) - s.get("size_bytes_before", 0)),
                pb=s.get("median_ppl_baseline_no_inject", s.get("median_ppl_before", "")),
                pa=s.get("median_ppl_injected_sp_ship", s.get("median_ppl_after", "")),
                date=s.get("date", ""),
            )
        )
    note = next((s.get("note") for s in summaries if s.get("note")), None)
    if note:
        lines += ["", f"_{note}_"]
    return "\n".join(lines)


def build_section(logs: list[dict]) -> str:
    ppl_rows = []
    for l in logs:
        if "median_ppl" not in l:
            continue
        if not l.get("config") or not l.get("model"):
            continue
        l2 = dict(l)
        l2["ctx_chunks"] = f"{l.get('ctx', '')}/{l.get('chunks', '')}"
        ppl_rows.append(l2)
    ppl_rows.sort(key=lambda r: (r.get("backend", ""), r.get("model", ""), r.get("config", "")))

    parts = [
        SENTINEL_BEGIN,
        f"",
        f"_Auto-generated from `logs/*.json` on "
        f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "### KV cache perplexity (VHT2 ship vs sqfree aggressive)",
        "",
    ]
    if ppl_rows:
        parts.append(render_table(ppl_rows))
    else:
        parts.append("_No perplexity logs present yet — run `scripts/bench_cuda_qwen3.sh` etc._")
    pred = render_predictor(logs)
    if pred:
        parts.append(pred)
    parts.append("")
    parts.append(SENTINEL_END)
    return "\n".join(parts)


def splice(text: str, section: str) -> str:
    if SENTINEL_BEGIN in text and SENTINEL_END in text:
        pre = text.split(SENTINEL_BEGIN, 1)[0]
        post = text.split(SENTINEL_END, 1)[1]
        return pre.rstrip() + "\n\n" + section + "\n" + post.lstrip("\n")
    return text.rstrip() + "\n\n" + section + "\n"


def main() -> int:
    logs = load_logs()
    section = build_section(logs)
    for target in TARGET_FILES:
        if not target.exists():
            print(f"skip {target} (missing)", file=sys.stderr)
            continue
        old = target.read_text(encoding="utf-8")
        new = splice(old, section)
        if old != new:
            target.write_text(new, encoding="utf-8")
            print(f"updated {target}")
        else:
            print(f"{target} unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())

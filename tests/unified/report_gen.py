#!/usr/bin/env python3
# Shannon-Prime VHT2: Test Report Generator
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
# Licensed under AGPLv3.
#
# Reads JSON test logs and produces Markdown tables + standalone HTML dashboard.

import json
import os
from datetime import datetime


def generate_report(data, output_path, fmt='md'):
    """Generate a report from test JSON data."""
    if fmt == 'md':
        _generate_markdown(data, output_path)
    elif fmt == 'html':
        _generate_html(data, output_path)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def _generate_markdown(data, path):
    """Generate Markdown report with tables."""
    s = data['summary']
    p = data['platform']
    hw = data.get('hardware', {})

    lines = [
        f"# Shannon-Prime Test Report",
        f"",
        f"**Date:** {data['timestamp']}  ",
        f"**Platform:** {p['os']} {p['arch']} | Python {p['python']} | {p['hostname']}  ",
    ]

    if hw.get('cuda_devices'):
        gpus = ', '.join(d['name'] for d in hw['cuda_devices'])
        lines.append(f"**GPU:** {gpus}  ")

    lines += [
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total tests | {s['total']} |",
        f"| Passed | {s['passed']} |",
        f"| Failed | {s['failed']} |",
        f"| Duration | {s['elapsed']:.1f}s |",
        f"| Pass rate | {s['passed']/max(s['total'],1)*100:.1f}% |",
        f"",
    ]

    # Per-suite table
    suites = {}
    for r in data['results']:
        suite = r['suite']
        if suite not in suites:
            suites[suite] = {'passed': 0, 'failed': 0, 'total': 0}
        suites[suite]['total'] += 1
        if r['passed']:
            suites[suite]['passed'] += 1
        else:
            suites[suite]['failed'] += 1

    lines += [
        f"## Per-Suite Breakdown",
        f"",
        f"| Suite | Passed | Failed | Total | Rate |",
        f"|-------|--------|--------|-------|------|",
    ]
    for suite, st in sorted(suites.items()):
        rate = st['passed'] / max(st['total'], 1) * 100
        lines.append(f"| {suite} | {st['passed']} | {st['failed']} | {st['total']} | {rate:.1f}% |")

    # Category breakdown
    categories = {}
    for r in data['results']:
        cat = r.get('category', 'uncategorized')
        if cat not in categories:
            categories[cat] = {'passed': 0, 'failed': 0}
        if r['passed']:
            categories[cat]['passed'] += 1
        else:
            categories[cat]['failed'] += 1

    lines += [
        f"",
        f"## Per-Category Breakdown",
        f"",
        f"| Category | Passed | Failed |",
        f"|----------|--------|--------|",
    ]
    for cat, ct in sorted(categories.items()):
        lines.append(f"| {cat} | {ct['passed']} | {ct['failed']} |")

    # Failures
    failures = [r for r in data['results'] if not r['passed']]
    if failures:
        lines += [
            f"",
            f"## Failures",
            f"",
            f"| Suite | Test | Detail |",
            f"|-------|------|--------|",
        ]
        for r in failures:
            detail = r.get('detail', '').replace('|', '\\|')[:80]
            lines.append(f"| {r['suite']} | {r['name']} | {detail} |")

    lines.append("")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _generate_html(data, path):
    """Generate standalone HTML dashboard."""
    s = data['summary']
    p = data['platform']
    hw = data.get('hardware', {})

    # Build suite data for chart
    suites = {}
    for r in data['results']:
        suite = r['suite']
        if suite not in suites:
            suites[suite] = {'passed': 0, 'failed': 0}
        if r['passed']:
            suites[suite]['passed'] += 1
        else:
            suites[suite]['failed'] += 1

    suite_labels = json.dumps(list(suites.keys()))
    suite_passed = json.dumps([v['passed'] for v in suites.values()])
    suite_failed = json.dumps([v['failed'] for v in suites.values()])

    # Build category data
    categories = {}
    for r in data['results']:
        cat = r.get('category', 'uncategorized')
        if cat not in categories:
            categories[cat] = {'passed': 0, 'failed': 0}
        if r['passed']:
            categories[cat]['passed'] += 1
        else:
            categories[cat]['failed'] += 1

    # Failures list
    failures = [r for r in data['results'] if not r['passed']]
    failures_html = ""
    if failures:
        failures_html = "<h2>Failures</h2><table><tr><th>Suite</th><th>Test</th><th>Detail</th></tr>"
        for r in failures:
            d = r.get('detail', '')[:100].replace('<', '&lt;').replace('>', '&gt;')
            failures_html += f"<tr><td>{r['suite']}</td><td>{r['name']}</td><td>{d}</td></tr>"
        failures_html += "</table>"

    gpu_info = ""
    if hw.get('cuda_devices'):
        gpu_info = " | ".join(f"{d['name']} ({d['vram_gb']}GB)" for d in hw['cuda_devices'])

    pass_rate = s['passed'] / max(s['total'], 1) * 100
    status_color = '#22c55e' if s['failed'] == 0 else '#ef4444' if pass_rate < 80 else '#f59e0b'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Shannon-Prime Test Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 24px; }}
  h1 {{ font-size: 24px; margin-bottom: 8px; }}
  h2 {{ font-size: 18px; margin: 24px 0 12px; color: #94a3b8; }}
  .meta {{ color: #64748b; font-size: 13px; margin-bottom: 24px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 20px; text-align: center; }}
  .card .value {{ font-size: 32px; font-weight: 700; }}
  .card .label {{ font-size: 12px; color: #64748b; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
  th {{ background: #1e293b; color: #94a3b8; text-align: left; padding: 10px 12px; font-size: 12px; text-transform: uppercase; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #1e293b; font-size: 14px; }}
  tr:hover {{ background: #1e293b; }}
  .pass {{ color: #22c55e; }} .fail {{ color: #ef4444; }}
  .bar {{ display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 6px; }}
  .bar-pass {{ background: #22c55e; }} .bar-fail {{ background: #ef4444; }}
  .chart {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; }}
  .chart-item {{ background: #1e293b; border-radius: 8px; padding: 12px 16px; min-width: 180px; flex: 1; }}
  .chart-item .name {{ font-size: 14px; font-weight: 600; margin-bottom: 6px; }}
  .chart-item .stats {{ font-size: 12px; color: #94a3b8; }}
</style>
</head>
<body>
<h1>Shannon-Prime Test Report</h1>
<div class="meta">
  {data['timestamp']} | {p['os']} {p['arch']} | Python {p['python']} | {p['hostname']}
  {f' | GPU: {gpu_info}' if gpu_info else ''}
</div>

<div class="cards">
  <div class="card"><div class="value" style="color:{status_color}">{pass_rate:.0f}%</div><div class="label">Pass Rate</div></div>
  <div class="card"><div class="value">{s['total']}</div><div class="label">Total Tests</div></div>
  <div class="card"><div class="value pass">{s['passed']}</div><div class="label">Passed</div></div>
  <div class="card"><div class="value fail">{s['failed']}</div><div class="label">Failed</div></div>
  <div class="card"><div class="value">{s['elapsed']:.1f}s</div><div class="label">Duration</div></div>
</div>

<h2>Per-Suite</h2>
<div class="chart">"""

    for suite, st in sorted(suites.items()):
        total = st['passed'] + st['failed']
        pct = st['passed'] / max(total, 1) * 100
        pw = st['passed'] / max(total, 1) * 100
        fw = st['failed'] / max(total, 1) * 100
        html += f"""
  <div class="chart-item">
    <div class="name">{suite}</div>
    <div class="stats"><span class="pass">{st['passed']}</span> / {total} ({pct:.0f}%)</div>
    <div class="bar"><div class="bar-pass" style="width:{pw}%"></div><div class="bar-fail" style="width:{fw}%"></div></div>
  </div>"""

    html += f"""
</div>

<h2>Per-Category</h2>
<table>
<tr><th>Category</th><th>Passed</th><th>Failed</th><th>Rate</th></tr>"""

    for cat, ct in sorted(categories.items()):
        total = ct['passed'] + ct['failed']
        rate = ct['passed'] / max(total, 1) * 100
        html += f"<tr><td>{cat}</td><td class='pass'>{ct['passed']}</td><td class='fail'>{ct['failed']}</td><td>{rate:.0f}%</td></tr>"

    html += f"""
</table>

{failures_html}

<h2>All Results</h2>
<table>
<tr><th>Suite</th><th>Category</th><th>Test</th><th>Status</th><th>Time</th><th>Detail</th></tr>"""

    for r in data['results']:
        status = "<span class='pass'>PASS</span>" if r['passed'] else "<span class='fail'>FAIL</span>"
        detail = r.get('detail', '')[:60].replace('<', '&lt;')
        dur = f"{r['duration']*1000:.1f}ms" if r['duration'] > 0 else ""
        html += f"<tr><td>{r['suite']}</td><td>{r.get('category','')}</td><td>{r['name']}</td><td>{status}</td><td>{dur}</td><td>{detail}</td></tr>"

    html += """
</table>
</body>
</html>"""

    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python report_gen.py <results.json> [--html|--md|--both]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    fmt = 'both' if '--both' in sys.argv else 'html' if '--html' in sys.argv else 'md'
    base = os.path.splitext(sys.argv[1])[0]

    if fmt in ('md', 'both'):
        generate_report(data, base + '.md', 'md')
        print(f"Wrote {base}.md")
    if fmt in ('html', 'both'):
        generate_report(data, base + '.html', 'html')
        print(f"Wrote {base}.html")

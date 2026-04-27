#!/usr/bin/env python3
# Shannon-Prime VHT2: Unified Test Suite
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
# Licensed under AGPLv3. Commercial license available.
#
# Usage:
#   python run_tests.py                    # Run all available suites
#   python run_tests.py --suite math       # Run specific suite(s)
#   python run_tests.py --suite math,comp  # Comma-separated
#   python run_tests.py --report html      # Generate HTML dashboard
#   python run_tests.py --colab            # Colab-friendly output
#   python run_tests.py --json out.json    # Raw JSON log
#
# Environment:
#   SP_TEST_DEVICE=cpu|cuda|vulkan   Override device detection
#   SP_TEST_VERBOSE=1                Extra diagnostics
#   SP_TEST_MODEL=<path.gguf>        Enable live model tests

import argparse
import json
import os
import sys
import time
import traceback
import platform
from datetime import datetime
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
TESTS_DIR = Path(__file__).resolve().parent
# Support both layouts: workspace/tests/ or shannon-prime/tests/unified/
_parent = TESTS_DIR.parent
if _parent.name == 'tests' and (_parent.parent / 'backends').exists():
    # We're inside shannon-prime/tests/unified/ — go up 3 levels
    WORKSPACE = _parent.parent.parent
else:
    # We're at workspace/tests/ — go up 1 level
    WORKSPACE = _parent
SP_CORE = WORKSPACE / "shannon-prime"
SP_ENGINE = WORKSPACE / "shannon-prime-engine"
SP_LLAMA = WORKSPACE / "shannon-prime-llama"
SP_COMFYUI = WORKSPACE / "shannon-prime-comfyui"

# Add import paths — TESTS_DIR itself must be on sys.path for test_*.py discovery
for p in [
    TESTS_DIR,
    SP_CORE / "backends" / "torch",
    SP_CORE / "tools",
    SP_COMFYUI / "nodes",
    SP_COMFYUI / "lib" / "shannon-prime" / "tools",
]:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Ensure `from run_tests import ...` in test_*.py resolves to THIS module
# (avoids double-import when running as `python run_tests.py` → __main__)
import importlib as _il
if __name__ == '__main__':
    sys.modules.setdefault('run_tests', sys.modules[__name__])


# ── Test result accumulator ──────────────────────────────────────��──────────

class TestResult:
    """Single test outcome."""
    __slots__ = ('name', 'suite', 'passed', 'duration', 'detail', 'category')

    def __init__(self, name, suite, passed, duration=0.0, detail="", category=""):
        self.name = name
        self.suite = suite
        self.passed = passed
        self.duration = duration
        self.detail = detail
        self.category = category  # 'invariant', 'falsification', 'quality', etc.

    def to_dict(self):
        return {
            'name': self.name, 'suite': self.suite,
            'passed': self.passed, 'duration': round(self.duration, 4),
            'detail': self.detail, 'category': self.category,
        }


class TestRunner:
    """Accumulates results across all suites."""

    def __init__(self, verbose=False):
        self.results: list[TestResult] = []
        self.verbose = verbose
        self._suite = ""
        self._category = ""
        self._start = time.time()

    def set_suite(self, name):
        self._suite = name

    def set_category(self, cat):
        self._category = cat

    def check(self, cond, name, detail=""):
        """Record a single test check."""
        r = TestResult(
            name=name, suite=self._suite, passed=bool(cond),
            detail=detail, category=self._category,
        )
        self.results.append(r)
        status = "\033[32mPASS\033[0m" if r.passed else "\033[31mFAIL\033[0m"
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail and self.verbose else ""))
        return r.passed

    def check_raises(self, exc_type, fn, name, detail=""):
        """Falsification: verify that fn() raises the expected exception."""
        try:
            fn()
            self.check(False, name, f"Expected {exc_type.__name__}, got no exception")
        except exc_type:
            self.check(True, name, detail or f"Correctly raised {exc_type.__name__}")
        except Exception as e:
            self.check(False, name, f"Expected {exc_type.__name__}, got {type(e).__name__}: {e}")

    def check_must_fail(self, cond, name, detail=""):
        """Falsification: a condition that MUST be False for the test to pass."""
        r = TestResult(
            name=f"[FALSIFY] {name}", suite=self._suite, passed=not cond,
            detail=detail or "Correctly failed as expected",
            category="falsification",
        )
        self.results.append(r)
        if not cond:
            print(f"  [\033[32mPASS\033[0m] [FALSIFY] {name}")
        else:
            print(f"  [\033[31mFAIL\033[0m] [FALSIFY] {name} — should have failed but didn't")
        return not cond

    def timed(self, fn, name, detail=""):
        """Run fn(), check it returns truthy, record timing."""
        t0 = time.perf_counter()
        try:
            result = fn()
            dt = time.perf_counter() - t0
            r = TestResult(
                name=name, suite=self._suite, passed=bool(result),
                duration=dt, detail=detail, category=self._category,
            )
            self.results.append(r)
            status = "\033[32mPASS\033[0m" if r.passed else "\033[31mFAIL\033[0m"
            print(f"  [{status}] {name} [{dt*1000:.1f}ms]")
            return r.passed
        except Exception as e:
            dt = time.perf_counter() - t0
            r = TestResult(
                name=name, suite=self._suite, passed=False,
                duration=dt, detail=f"Exception: {e}", category=self._category,
            )
            self.results.append(r)
            print(f"  [\033[31mFAIL\033[0m] {name} — {e}")
            if self.verbose:
                traceback.print_exc()
            return False

    # ── Summary ─────────────────────────────────────────────────────────────

    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        elapsed = time.time() - self._start

        print(f"\n{'='*70}")
        print(f"  TOTAL: {passed}/{total} passed ({failed} failed) in {elapsed:.1f}s")
        print(f"{'='*70}")

        # Per-suite breakdown
        suites = {}
        for r in self.results:
            s = suites.setdefault(r.suite, {'p': 0, 't': 0})
            s['t'] += 1
            if r.passed:
                s['p'] += 1

        print(f"\n  {'Suite':<35} {'Pass':>6} {'Total':>6} {'Rate':>8}")
        print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*8}")
        for suite, s in sorted(suites.items()):
            rate = s['p'] / s['t'] * 100 if s['t'] else 0
            color = "\033[32m" if s['p'] == s['t'] else "\033[33m" if rate > 80 else "\033[31m"
            print(f"  {suite:<35} {s['p']:>6} {s['t']:>6} {color}{rate:>7.1f}%\033[0m")

        # List failures
        failures = [r for r in self.results if not r.passed]
        if failures:
            print(f"\n  FAILURES:")
            for r in failures:
                print(f"    [{r.suite}] {r.name}" + (f" — {r.detail}" if r.detail else ""))

        return failed == 0

    def to_json(self):
        return {
            'timestamp': datetime.now().isoformat(),
            'platform': {
                'os': platform.system(),
                'arch': platform.machine(),
                'python': platform.python_version(),
                'hostname': platform.node(),
            },
            'hardware': self._detect_hardware(),
            'summary': {
                'total': len(self.results),
                'passed': sum(1 for r in self.results if r.passed),
                'failed': sum(1 for r in self.results if not r.passed),
                'elapsed': round(time.time() - self._start, 2),
            },
            'results': [r.to_dict() for r in self.results],
        }

    @staticmethod
    def _detect_hardware():
        hw = {'cuda': False, 'cuda_devices': [], 'vulkan': False}
        try:
            import torch
            hw['cuda'] = torch.cuda.is_available()
            if hw['cuda']:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    hw['cuda_devices'].append({
                        'name': props.name,
                        'vram_gb': round(props.total_mem / 1e9, 1),
                        'compute': f"{props.major}.{props.minor}",
                    })
        except ImportError:
            pass
        return hw


# ── Suite registry ──────────────────────────────────────────────────────────

SUITE_REGISTRY = {}

def register_suite(name, description=""):
    """Decorator to register a test suite function."""
    def deco(fn):
        SUITE_REGISTRY[name] = {'fn': fn, 'desc': description}
        return fn
    return deco


# ── Import all test modules (they register themselves) ──────────────────────

def _import_suites():
    """Import all test_*.py in this directory."""
    for f in sorted(TESTS_DIR.glob("test_*.py")):
        module_name = f.stem
        try:
            __import__(module_name)
        except ImportError as e:
            print(f"  [SKIP] {module_name}: {e}")
        except Exception as e:
            print(f"  [ERROR] loading {module_name}: {e}")


# ── Main ──���─────────────────────────────────────────────────────────────────

def main():
    # Force UTF-8 + line-buffered stdout for Windows/CI/notebook environments
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Shannon-Prime Unified Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                      Run all suites
  python run_tests.py --suite math,comp    Run specific suites
  python run_tests.py --report html        Generate HTML dashboard
  python run_tests.py --list               List available suites
  python run_tests.py --json results.json  Save raw JSON log
  python run_tests.py --colab              Colab-friendly mode
        """)
    parser.add_argument('--suite', '-s', type=str, default='all',
                        help='Comma-separated suite names, or "all"')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available suites and exit')
    parser.add_argument('--json', type=str, default=None,
                        help='Write JSON results to file')
    parser.add_argument('--report', type=str, default=None,
                        choices=['md', 'html', 'both'],
                        help='Generate report (md/html/both)')
    parser.add_argument('--colab', action='store_true',
                        help='Colab-friendly output (no ANSI, inline tables)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Extra diagnostics')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  Shannon-Prime Unified Test Suite")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {platform.node()}")
    print(f"{'='*70}\n")

    # Import all test modules
    _import_suites()

    if args.list:
        print(f"Available suites ({len(SUITE_REGISTRY)}):\n")
        for name, info in sorted(SUITE_REGISTRY.items()):
            print(f"  {name:<30} {info['desc']}")
        return 0

    runner = TestRunner(verbose=args.verbose)

    # Select suites
    if args.suite == 'all':
        suites = list(SUITE_REGISTRY.keys())
    else:
        suites = [s.strip() for s in args.suite.split(',')]
        for s in suites:
            if s not in SUITE_REGISTRY:
                print(f"Unknown suite: {s}")
                print(f"Available: {', '.join(sorted(SUITE_REGISTRY.keys()))}")
                return 1

    # Run suites
    for name in sorted(suites):
        info = SUITE_REGISTRY[name]
        print(f"\n{'─'*70}")
        print(f"  Suite: {name} — {info['desc']}")
        print(f"{'─'*70}")
        runner.set_suite(name)
        try:
            info['fn'](runner)
        except Exception as e:
            print(f"\n  [\033[31mSUITE ERROR\033[0m] {name}: {e}")
            if args.verbose:
                traceback.print_exc()
            runner.check(False, f"{name} suite execution", str(e))

    # Summary
    all_passed = runner.summary()

    # JSON output
    json_data = runner.to_json()
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        print(f"\n  JSON results: {args.json}")

    # Default: always write JSON to tests/results/
    results_dir = TESTS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_json = results_dir / f"run_{ts}.json"
    with open(default_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

    # Report generation
    if args.report:
        try:
            from report_gen import generate_report
            if args.report in ('md', 'both'):
                md_path = results_dir / f"report_{ts}.md"
                generate_report(json_data, str(md_path), fmt='md')
                print(f"  Markdown report: {md_path}")
            if args.report in ('html', 'both'):
                html_path = results_dir / f"report_{ts}.html"
                generate_report(json_data, str(html_path), fmt='html')
                print(f"  HTML dashboard: {html_path}")
        except ImportError:
            print("  [WARN] report_gen.py not found -- skipping report generation")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

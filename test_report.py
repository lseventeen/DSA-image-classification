"""
Test Report Generator — Run all tests and produce an analysis report.

Executes pytest with JUnit XML output, parses results, and generates:
  1. Console summary with pass/fail counts per module
  2. A Markdown report  (outputs/test_report.md)
  3. A bar-chart PNG     (outputs/test_results.png)

Usage:
    python test_report.py
"""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
JUNIT_XML = OUTPUT_DIR / "junit_results.xml"
REPORT_MD = OUTPUT_DIR / "test_report.md"
CHART_PNG = OUTPUT_DIR / "test_results.png"


def run_tests():
    """Run pytest and capture JUnit XML results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pytest",
        str(PROJECT_ROOT / "tests"),
        f"--junitxml={JUNIT_XML}",
        "-v",
        "--tb=short",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def parse_junit_xml(xml_path):
    """Parse JUnit XML into a structured dict of module → test results."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    modules = defaultdict(lambda: {"passed": [], "failed": [], "errors": [],
                                    "skipped": [], "duration": 0.0})

    for suite in root.iter("testsuite"):
        for tc in suite.iter("testcase"):
            classname = tc.attrib.get("classname", "unknown")
            name = tc.attrib.get("name", "unknown")
            time_s = float(tc.attrib.get("time", 0))

            # Extract module name (e.g. tests.test_model)
            module = classname.rsplit(".", 1)[0] if "." in classname else classname

            full_name = f"{classname}::{name}"

            failure = tc.find("failure")
            error = tc.find("error")
            skipped = tc.find("skipped")

            if failure is not None:
                modules[module]["failed"].append({
                    "name": full_name,
                    "message": failure.attrib.get("message", ""),
                    "duration": time_s,
                })
            elif error is not None:
                modules[module]["errors"].append({
                    "name": full_name,
                    "message": error.attrib.get("message", ""),
                    "duration": time_s,
                })
            elif skipped is not None:
                modules[module]["skipped"].append({
                    "name": full_name,
                    "message": skipped.attrib.get("message", ""),
                    "duration": time_s,
                })
            else:
                modules[module]["passed"].append({
                    "name": full_name,
                    "duration": time_s,
                })
            modules[module]["duration"] += time_s

    return dict(modules)


def generate_chart(modules, save_path):
    """Generate a grouped bar chart of test results per module."""
    mod_names = sorted(modules.keys())
    # Shorten module names for display
    short_names = [m.replace("tests.test_", "") for m in mod_names]

    passed = [len(modules[m]["passed"]) for m in mod_names]
    failed = [len(modules[m]["failed"]) + len(modules[m]["errors"]) for m in mod_names]
    skipped = [len(modules[m]["skipped"]) for m in mod_names]

    x = np.arange(len(mod_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_p = ax.bar(x - width, passed, width, label="Passed", color="#4CAF50")
    bars_f = ax.bar(x, failed, width, label="Failed", color="#F44336")
    bars_s = ax.bar(x + width, skipped, width, label="Skipped", color="#FFC107")

    # Add count labels on bars
    for bars in [bars_p, bars_f, bars_s]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Test Module")
    ax.set_ylabel("Number of Tests")
    ax.set_title("Test Results by Module")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {save_path}")


def generate_markdown_report(modules, save_path):
    """Generate a Markdown report summarising all test results."""
    total_passed = sum(len(m["passed"]) for m in modules.values())
    total_failed = sum(len(m["failed"]) + len(m["errors"]) for m in modules.values())
    total_skipped = sum(len(m["skipped"]) for m in modules.values())
    total = total_passed + total_failed + total_skipped
    total_time = sum(m["duration"] for m in modules.values())

    status = "✅ ALL TESTS PASSED" if total_failed == 0 else "❌ SOME TESTS FAILED"

    lines = [
        "# 🧪 Test Report — DSA Image Classification",
        "",
        f"**Status:** {status}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total tests | {total} |",
        f"| Passed | {total_passed} |",
        f"| Failed | {total_failed} |",
        f"| Skipped | {total_skipped} |",
        f"| Pass rate | {total_passed / max(total, 1) * 100:.1f}% |",
        f"| Total time | {total_time:.2f}s |",
        "",
        "## Results by Module",
        "",
        "| Module | Passed | Failed | Skipped | Time (s) |",
        "|--------|--------|--------|---------|----------|",
    ]

    for mod in sorted(modules.keys()):
        m = modules[mod]
        short = mod.replace("tests.test_", "")
        p = len(m["passed"])
        f = len(m["failed"]) + len(m["errors"])
        s = len(m["skipped"])
        icon = "✅" if f == 0 else "❌"
        lines.append(f"| {icon} {short} | {p} | {f} | {s} | {m['duration']:.2f} |")

    # Detail section for failures
    any_failures = any(
        m["failed"] or m["errors"] for m in modules.values()
    )
    if any_failures:
        lines += ["", "## ❌ Failure Details", ""]
        for mod in sorted(modules.keys()):
            m = modules[mod]
            for item in m["failed"] + m["errors"]:
                lines.append(f"### `{item['name']}`")
                lines.append(f"```\n{item['message']}\n```")
                lines.append("")

    # Chart reference
    lines += [
        "",
        "## 📊 Results Chart",
        "",
        "![Test Results](test_results.png)",
        "",
        "---",
        "*Report generated automatically by `test_report.py`*",
    ]

    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to {save_path}")


def print_console_summary(modules):
    """Print a coloured console summary."""
    print("\n" + "=" * 70)
    print("  TEST RESULTS SUMMARY")
    print("=" * 70)

    total_p = total_f = total_s = 0
    for mod in sorted(modules.keys()):
        m = modules[mod]
        p = len(m["passed"])
        f = len(m["failed"]) + len(m["errors"])
        s = len(m["skipped"])
        total_p += p
        total_f += f
        total_s += s
        short = mod.replace("tests.test_", "")
        icon = "✅" if f == 0 else "❌"
        print(f"  {icon} {short:20s}  passed={p}  failed={f}  "
              f"skipped={s}  time={m['duration']:.2f}s")

    total = total_p + total_f + total_s
    print("-" * 70)
    rate = total_p / max(total, 1) * 100
    print(f"  TOTAL: {total}  |  Passed: {total_p}  |  Failed: {total_f}  "
          f"|  Skipped: {total_s}  |  Pass rate: {rate:.1f}%")
    print("=" * 70 + "\n")


def main():
    print("Running tests …\n")
    exit_code = run_tests()

    if not JUNIT_XML.exists():
        print("ERROR: JUnit XML not generated. Tests may have crashed.")
        sys.exit(1)

    modules = parse_junit_xml(JUNIT_XML)

    print_console_summary(modules)
    generate_chart(modules, CHART_PNG)
    generate_markdown_report(modules, REPORT_MD)

    print(f"\n📄 Markdown report: {REPORT_MD}")
    print(f"📊 Chart:           {CHART_PNG}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

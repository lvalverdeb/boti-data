#!/usr/bin/env python3
"""
Validate that test counts in the fit/gap analysis match actual pytest collection.

Exits non-zero on mismatch, so it can be used as a CI gate.
Expected counts are read from `specs/boti-data-etl/fit-gap-analysis.md`.

Usage:
    python scripts/validate_spec_counts.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "specs" / "boti-data-etl" / "fit-gap-analysis.md"


def _run_pytest_collect(pattern: str = "", marker: str = "") -> int:
    cmd = ["uv", "run", "pytest", "--collect-only", "-q", "--no-header"]
    if marker:
        cmd.extend(["-m", marker])
    if pattern:
        cmd.append(pattern)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print("pytest --collect-only failed:", result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    last_line = result.stdout.strip().rsplit("\n", 1)[-1]
    match = re.search(r"(\d+)\s+tests?\s+collected", last_line)
    if not match:
        print(f"Could not parse test count from: {last_line!r}", file=sys.stderr)
        sys.exit(1)
    return int(match.group(1))


def _extract_counts_from_spec() -> tuple[int, int]:
    """Return (expected_total, expected_security) parsed from fit-gap-analysis.md."""
    text = SPEC_PATH.read_text()

    m_total = re.search(r"collect-only -q[`]* returns (\d+) tests", text)
    if not m_total:
        print(f"Could not find total test count in {SPEC_PATH}", file=sys.stderr)
        sys.exit(1)

    m_sec = re.search(r"tests/security/.*?returns (\d+) tests", text, re.DOTALL)
    if not m_sec:
        print(f"Could not find security test count in {SPEC_PATH}", file=sys.stderr)
        sys.exit(1)

    return int(m_total.group(1)), int(m_sec.group(1))


def main() -> None:
    errors = 0
    expected_total, expected_security = _extract_counts_from_spec()

    total_collected = _run_pytest_collect()
    if total_collected != expected_total:
        print(
            f"FAIL: Total tests collected = {total_collected}, "
            f"expected {expected_total} (from SC-005 in fit-gap-analysis.md)",
            file=sys.stderr,
        )
        errors += 1
    else:
        print(f"OK: Total tests = {total_collected}")

    security_collected = _run_pytest_collect(pattern="tests/security/")
    if security_collected != expected_security:
        print(
            f"FAIL: Security tests collected = {security_collected}, "
            f"expected {expected_security} (from SC-006 in fit-gap-analysis.md)",
            file=sys.stderr,
        )
        errors += 1
    else:
        print(f"OK: Security tests = {security_collected}")

    if errors:
        print(
            f"\n{errors} count(s) drifted. Bump expected counts in:\n"
            f"  1. {SPEC_PATH.relative_to(REPO_ROOT)}\n"
            f"  2. {Path(__file__).relative_to(REPO_ROOT)} (if hardcoded defaults change)"
        )
        sys.exit(1)

    print("\nAll spec test counts match collection.")


if __name__ == "__main__":
    main()

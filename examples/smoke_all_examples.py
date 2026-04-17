"""
Run all example scripts as a lightweight smoke test.

Usage:
    uv run python examples/smoke_all_examples.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Keep heavy examples quick unless the caller already set explicit values.
_FAST_DEFAULTS: dict[str, str] = {
    "BOTI_EXAMPLE_LEFT_ROWS": "1000",
    "BOTI_EXAMPLE_RIGHT_ROWS": "800",
    "BOTI_EXAMPLE_BATCH_SIZE": "200",
    "BOTI_EXAMPLE_DIAGNOSTIC_LEFT_ROWS": "1000",
    "BOTI_EXAMPLE_DIAGNOSTIC_RIGHT_ROWS": "750",
    "BOTI_EXAMPLE_DIAGNOSTIC_BATCH_SIZE": "250",
    "BOTI_EXAMPLE_DIAGNOSTIC_WORKERS": "1",
}


def discover_examples(examples_dir: Path) -> list[Path]:
    scripts = sorted(
        p for p in examples_dir.glob("*.py") if p.name != "smoke_all_examples.py"
    )
    return scripts


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    for key, value in _FAST_DEFAULTS.items():
        env.setdefault(key, value)
    return env


def run_script(repo_root: Path, script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"
    scripts = discover_examples(examples_dir)

    if not scripts:
        print("No example scripts found.")
        return 0

    env = build_env()
    failures: list[tuple[Path, subprocess.CompletedProcess[str]]] = []

    print(f"Running {len(scripts)} example scripts...")
    for script in scripts:
        print(f"=== RUN {script.relative_to(repo_root)} ===")
        result = run_script(repo_root, script, env)
        if result.returncode == 0:
            print(f"PASS {script.relative_to(repo_root)}")
            continue

        print(f"FAIL {script.relative_to(repo_root)} (exit={result.returncode})")
        failures.append((script, result))

    print()
    print(f"Completed: {len(scripts) - len(failures)} passed, {len(failures)} failed")

    if not failures:
        return 0

    print("\nFailure details:")
    for script, result in failures:
        rel = script.relative_to(repo_root)
        print(f"\n--- {rel} ---")
        if result.stdout.strip():
            print("[stdout]")
            print(result.stdout.rstrip())
        if result.stderr.strip():
            print("[stderr]")
            print(result.stderr.rstrip())

    return 1


if __name__ == "__main__":
    raise SystemExit(main())


"""
Interactive menu for browsing and running boti-data example scripts.

Usage:
    uv run python examples/menu.py                # interactive menu
    uv run python examples/menu.py --list          # print the menu and exit
    uv run python examples/menu.py sql_settings    # run one example directly
    uv run python examples/menu.py 7               # run by menu number
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

# Mirrors the grouping documented in examples/README.md. New example files are
# still picked up automatically (see _build_entries) and listed under "Other"
# until they're filed into a category here.
CATEGORIES: list[tuple[str, list[str]]] = [
    (
        "SQL & connections",
        [
            "connection_catalog.py",
            "sql_settings.py",
            "sql_sync_query_only.py",
            "sql_sync_writable.py",
            "sql_model_registry.py",
            "sql_model_builder.py",
            "sql_async_resource.py",
            "sql_partitioned_loader.py",
            "sql_pgvector.py",
        ],
    ),
    ("Parquet", ["parquet_resource.py", "data_parquet_pipeline.py", "data_incremental_loading.py"]),
    (
        "DataGateway (facade)",
        [
            "data_facade_db.py",
            "data_facade_parquet.py",
            "data_facade_diagnostics.py",
            "data_facade_distributed.py",
            "data_facade_distributed_parquet.py",
            "data_facade_datacube.py",
            "data_facade_datacube_contract_rejection.py",
        ],
    ),
    (
        "DataHelper",
        [
            "data_helper_legacy.py",
            "data_helper_distributed.py",
            "data_helper_bootstrap.py",
            "data_helper_semi_join.py",
            "data_helper_worker_config.py",
        ],
    ),
    (
        "Hybrid dataset",
        ["data_hybrid_dataset.py", "data_hybrid_dataset_sql_parquet.py", "data_hybrid_dataset_distributed.py"],
    ),
    ("Sink pipelines", ["data_csv_sink_pipeline.py", "data_jsonl_sink_pipeline.py"]),
    ("Enrichment", ["data_enrichment_v1.py"]),
    ("Utilities", ["smoke_all_examples.py"]),
]

Entry = tuple[int, str, str, Path, str]

EXIT_UNKNOWN_EXAMPLE = 2


def _docstring_summary(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return ""
    doc = ast.get_docstring(tree)
    return doc.strip().splitlines()[0] if doc else ""


def _build_entries(examples_dir: Path) -> list[Entry]:
    entries: list[Entry] = []
    listed: set[str] = set()
    number = 1
    for category, names in CATEGORIES:
        for name in names:
            path = examples_dir / name
            if not path.is_file():
                continue
            entries.append((number, category, name, path, _docstring_summary(path)))
            listed.add(name)
            number += 1

    extras = sorted(
        p.name
        for p in examples_dir.glob("*.py")
        if p.name not in listed and p.name != "menu.py" and not p.name.startswith("_")
    )
    for name in extras:
        path = examples_dir / name
        entries.append((number, "Other", name, path, _docstring_summary(path)))
        number += 1

    return entries


def _print_menu(entries: list[Entry]) -> None:
    print("boti-data examples\n")
    last_category = None
    for number, category, name, _path, summary in entries:
        if category != last_category:
            print(f"{category}:")
            last_category = category
        print(f"  {number:2d}) {name:<40} {summary}")
    print("\n  a) run all (smoke test)\n  q) quit\n")


def _resolve_selection(selection: str, entries: list[Entry]) -> Path | None:
    selection = selection.strip()
    if selection.isdigit():
        target = int(selection)
        matches = (path for number, _cat, _name, path, _summary in entries if number == target)
    else:
        matches = (
            path for _number, _cat, name, path, _summary in entries if name in (selection, f"{selection}.py")
        )
    return next(matches, None)


def _run_example(repo_root: Path, path: Path) -> int:
    print(f"\n=== RUN {path.relative_to(repo_root)} ===\n", flush=True)
    result = subprocess.run([sys.executable, str(path)], cwd=repo_root, check=False)
    return result.returncode


def _interactive_loop(repo_root: Path, examples_dir: Path, entries: list[Entry]) -> int:
    while True:
        _print_menu(entries)
        choice = input("Select an example (number, q to quit): ").strip().lower()
        if choice in ("q", "quit", "exit"):
            return 0
        if choice in ("a", "all"):
            return _run_example(repo_root, examples_dir / "smoke_all_examples.py")

        path = _resolve_selection(choice, entries)
        if path is None:
            print(f"Unknown selection: {choice!r}\n")
            continue

        _run_example(repo_root, path)
        input("\nPress Enter to return to the menu...")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"
    entries = _build_entries(examples_dir)

    if args and args[0] in ("-l", "--list"):
        _print_menu(entries)
        exit_code = 0
    elif not args:
        exit_code = _interactive_loop(repo_root, examples_dir, entries)
    else:
        path = _resolve_selection(args[0], entries)
        if path is None:
            print(f"Unknown example: {args[0]!r}", file=sys.stderr)
            exit_code = EXIT_UNKNOWN_EXAMPLE
        else:
            exit_code = _run_example(repo_root, path)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

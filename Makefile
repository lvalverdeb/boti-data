# boti-data is its own git repo nested inside the uv workspace root, so
# `git rev-parse --show-toplevel` would return boti-data's own root, not
# the workspace root where `uv build` actually writes dist/ — derive it
# from the Makefile's own location instead (always one level up).
REPO_ROOT := $(shell realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST))))..)

.PHONY: help clean build verify check publish publish-test upload upload-test install-dev test test-security lint validate-spec bench bench-compare bench-baseline docs docs-serve

LOAD_ENV = if [ -f .env ]; then set -a; . ./.env; set +a; fi
REQUIRE_PUBLISH_TOKEN = test -n "$$UV_PUBLISH_TOKEN" || { echo "UV_PUBLISH_TOKEN is required (set it in .env or the environment)."; exit 1; }

help:
	@echo "Available targets:"
	@echo "  clean          - Remove build and distribution artifacts"
	@echo "  build          - Build sdist and wheel"
	@echo "  verify         - Run the exact checks CI runs: lint + tests + security tests + spec-count validation"
	@echo "  check          - verify + dry-run publish"
	@echo "  publish        - Publish to PyPI (loads UV_PUBLISH_TOKEN from .env if present)"
	@echo "  publish-test   - Publish to TestPyPI (loads UV_PUBLISH_TOKEN from .env if present)"
	@echo "  upload         - Alias for publish (kept for backward compatibility)"
	@echo "  upload-test    - Alias for publish-test (kept for backward compatibility)"
	@echo "  install-dev    - Install package with dev dependencies"
	@echo "  test           - Run full test suite (excluding security regressions)"
	@echo "  test-security  - Run security regression tests only"
	@echo "  lint           - Run ruff linter on src/ and tests/"
	@echo "  validate-spec  - Validate spec test counts match actual collection"
	@echo "  bench          - Run the benchmark suite and save results to benchmarks/results/"
	@echo "  bench-compare  - Run the benchmarks and fail on a >20% mean regression against the baseline"
	@echo "  bench-baseline - Regenerate the committed baseline (do this on each release)"
	@echo "  docs           - Build the documentation site into site/ (strict)"
	@echo "  docs-serve     - Serve the documentation site locally with live reload"

clean:
	rm -rf $(REPO_ROOT)/dist/ $(REPO_ROOT)/build/ *.egg-info src/*.egg-info

build: clean
	uv build

# Mirrors .github/workflows/ci.yml's test job exactly, so a local publish
# can't happen without the same checks CI enforces.
verify: lint test test-security validate-spec

check: verify build
	@$(LOAD_ENV); $(REQUIRE_PUBLISH_TOKEN); uv publish --dry-run --token "$$UV_PUBLISH_TOKEN" $(REPO_ROOT)/dist/boti_data-*

publish: verify build
	@$(LOAD_ENV); $(REQUIRE_PUBLISH_TOKEN); uv publish --token "$$UV_PUBLISH_TOKEN" $(REPO_ROOT)/dist/boti_data-*

publish-test: verify build
	@$(LOAD_ENV); $(REQUIRE_PUBLISH_TOKEN); uv publish --publish-url https://test.pypi.org/legacy/ --token "$$UV_PUBLISH_TOKEN" $(REPO_ROOT)/dist/boti_data-*

# Backward-compatible aliases -- publish/publish-test are canonical (match
# spaghetti/fenceline's naming); upload/upload-test kept so nothing already
# calling them by the old name breaks.
upload: publish
upload-test: publish-test

install-dev:
	uv sync --group dev

test:
	uv run pytest tests/ -m "not security_regression" --tb=short -q

test-security:
	uv run pytest tests/ -m security_regression --tb=short -q

lint:
	uv run ruff check src/ tests/

validate-spec:
	uv run python scripts/validate_spec_counts.py


# Benchmarks live outside tests/ and are marked perf, so the default suite
# never runs them. The baseline is per platform (pytest-benchmark keys it on
# OS/interpreter) and is regenerated on each release.
BENCH_FLAGS = -m perf --benchmark-only --benchmark-min-rounds=5 -p no:cacheprovider
BENCH_STORAGE = --benchmark-storage=file://benchmarks/baseline

bench:
	mkdir -p benchmarks/results
	uv run pytest benchmarks/ $(BENCH_FLAGS) --benchmark-json=benchmarks/results/$$(date +%Y%m%dT%H%M%S).json

bench-compare:
	uv run pytest benchmarks/ $(BENCH_FLAGS) $(BENCH_STORAGE) --benchmark-compare='*/0001_baseline' --benchmark-compare-fail=mean:20%

bench-baseline:
	rm -rf benchmarks/baseline
	uv run pytest benchmarks/ $(BENCH_FLAGS) $(BENCH_STORAGE) --benchmark-save=baseline

docs:
	uv run --group docs mkdocs build --strict

docs-serve:
	uv run --group docs mkdocs serve

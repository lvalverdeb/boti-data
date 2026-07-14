# boti-data is its own git repo nested inside the uv workspace root, so
# `git rev-parse --show-toplevel` would return boti-data's own root, not
# the workspace root where `uv build` actually writes dist/ — derive it
# from the Makefile's own location instead (always one level up).
REPO_ROOT := $(shell realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST))))..)

.PHONY: help clean build check upload upload-test install-dev test test-security lint

LOAD_ENV = if [ -f .env ]; then set -a; . ./.env; set +a; fi
REQUIRE_PUBLISH_TOKEN = test -n "$$UV_PUBLISH_TOKEN" || { echo "UV_PUBLISH_TOKEN is required (set it in .env or the environment)."; exit 1; }

help:
	@echo "Available targets:"
	@echo "  clean          - Remove build and distribution artifacts"
	@echo "  build          - Build sdist and wheel"
	@echo "  check          - Validate the publish command with uv publish --dry-run"
	@echo "  upload         - Upload to PyPI (loads UV_PUBLISH_TOKEN from .env if present)"
	@echo "  upload-test    - Upload to TestPyPI (loads UV_PUBLISH_TOKEN from .env if present)"
	@echo "  install-dev    - Install package with dev dependencies"
	@echo "  test           - Run full test suite (excluding security regressions)"
	@echo "  test-security  - Run security regression tests only"
	@echo "  lint           - Run ruff linter on src/ and tests/"

clean:
	rm -rf $(REPO_ROOT)/dist/ $(REPO_ROOT)/build/ *.egg-info src/*.egg-info

build: clean
	uv build

check: build
	@$(LOAD_ENV); $(REQUIRE_PUBLISH_TOKEN); uv publish --dry-run --token "$$UV_PUBLISH_TOKEN" $(REPO_ROOT)/dist/boti_data-*

upload: build
	@$(LOAD_ENV); $(REQUIRE_PUBLISH_TOKEN); uv publish --token "$$UV_PUBLISH_TOKEN" $(REPO_ROOT)/dist/boti_data-*

upload-test: build
	@$(LOAD_ENV); $(REQUIRE_PUBLISH_TOKEN); uv publish --publish-url https://test.pypi.org/legacy/ --token "$$UV_PUBLISH_TOKEN" $(REPO_ROOT)/dist/boti_data-*

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


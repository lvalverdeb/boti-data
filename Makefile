# boti-data is its own git repo nested inside the uv workspace root, so
# `git rev-parse --show-toplevel` would return boti-data's own root, not
# the workspace root where `uv build` actually writes dist/ — derive it
# from the Makefile's own location instead (always one level up).
REPO_ROOT := $(shell realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST))))..)

.PHONY: help clean build verify check publish publish-test upload upload-test install-dev test test-security lint validate-spec

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


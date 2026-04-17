.PHONY: help clean build check upload upload-test install-dev

LOAD_ENV = if [ -f .env ]; then set -a; . ./.env; set +a; fi
REQUIRE_PUBLISH_TOKEN = test -n "$$UV_PUBLISH_TOKEN" || { echo "UV_PUBLISH_TOKEN is required (set it in .env or the environment)."; exit 1; }

help:
	@echo "Available targets:"
	@echo "  clean        - Remove build and distribution artifacts"
	@echo "  build        - Build sdist and wheel"
	@echo "  check        - Validate the publish command with uv publish --dry-run"
	@echo "  upload       - Upload to PyPI (loads UV_PUBLISH_TOKEN from .env if present)"
	@echo "  upload-test  - Upload to TestPyPI (loads UV_PUBLISH_TOKEN from .env if present)"
	@echo "  install-dev  - Install package with dev dependencies"

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info

build: clean
	uv build

check: build
	@$(LOAD_ENV); $(REQUIRE_PUBLISH_TOKEN); uv publish --dry-run --token "$$UV_PUBLISH_TOKEN" dist/*

upload: build
	@$(LOAD_ENV); $(REQUIRE_PUBLISH_TOKEN); uv publish --token "$$UV_PUBLISH_TOKEN" dist/*

upload-test: build
	@$(LOAD_ENV); $(REQUIRE_PUBLISH_TOKEN); uv publish --publish-url https://test.pypi.org/legacy/ --token "$$UV_PUBLISH_TOKEN" dist/*

install-dev:
	uv pip install -e ".[dev]"

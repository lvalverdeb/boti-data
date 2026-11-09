# Deployment

`boti-data` publishes to PyPI with `uv publish`.

> **Breaking change reminder:** Dask runtime/session/resilience helpers are no longer re-exported from `boti_data`.
> Update imports like `from boti_data import DaskSession, safe_compute` to `from boti_dask import DaskSession, safe_compute`.

## Prerequisites

1. Create a PyPI API token.
2. Store it in a local `.env` file at the repository root:

```bash
UV_PUBLISH_TOKEN=your-pypi-token
```

`.env` is already gitignored and should not be committed.

## Release commands

Validate the built artifacts and publish command without uploading:

```bash
make check
```

Publish to PyPI:

```bash
make upload
```

Publish to TestPyPI:

```bash
make upload-test
```

The Makefile loads `.env` automatically when present and passes `UV_PUBLISH_TOKEN` to `uv publish`.

# Changelog

All notable changes to boti-data are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning policy

- Every name exported from `boti_data` is listed in exactly one of
  `boti_data.STABLE` and `boti_data.EXPERIMENTAL`.
- **Stable** names keep their signature and behaviour across minor releases.
  A stable name is removed or changed incompatibly only in a major release,
  and is first deprecated for at least one minor release: accessing it emits
  a `DeprecationWarning` naming the removal version and the replacement.
- **Experimental** names may change in any minor release. They are marked as
  such in the reference documentation and in this file when they change.
- Patch releases contain fixes only. Minor releases add features and
  deprecations. Major releases (2.0 onwards) remove what was deprecated and
  may change stable behaviour, with a migration guide.
- Floors on `boti` and `boti-dask` are raised in minor releases and recorded
  under "Changed", together with the compatibility matrix in the README.

## [Unreleased]

Targeting 1.4.0.

### Added

- Cluster-wide partition fetch gate: on a Dask cluster the gate is
  `boti_dask.cluster_semaphore`, a scheduler-side semaphore shared by every
  worker, so `max_concurrent_fetches` bounds the whole cluster rather than
  each worker process. The gate is keyed on a hash of the database identity
  alone, never on the limit, so two loads with different limits share one gate
  (first limit wins, warning logged). Without a running client, or with
  boti-dask below 1.1.0, the process-wide gate applies as before.
- Compatibility-matrix workflow testing the floor, locked and latest releases
  of `boti` and `boti-dask` on every push, weekly, and on an
  `upstream-release` repository dispatch; matrix published in the README.
- `s3` extra (`boti-data[s3]`) delegating to `boti[s3]` for `s3fs`, which
  boti 1.1.0 and later no longer install by default.
- Public API contract: `boti_data.STABLE` and `boti_data.EXPERIMENTAL`
  stability tuples covering every exported name, `boti_data.DEPRECATED` with
  the `boti_data._contract.deprecated` helper wired into the lazy module
  `__getattr__`, and this changelog.
- Benchmark suite under `benchmarks/` (eight pytest-benchmark cases runnable
  against SQLite and local Parquet), a committed baseline, `make bench` and
  `make bench-compare`, and a workflow that fails a release when a median
  regresses by more than 20 percent.
- Reference documentation site built with mkdocs-material and mkdocstrings,
  versioned with mike on release tags; README cut to installation, three
  examples and links.

### Changed

- `boti` floor raised to 1.2.0: `ParquetDataResource` resolves a relative
  `parquet_storage_path` against the project root, which boti anchors there
  only from 1.2.0 (1.0.x anchored to the working directory).
- README "Sizing connections and fetch concurrency" rewritten for the
  cluster-wide gate.

### Fixed

- `DataGateway` is pickleable again on boti 1.0.11 and later: the boti
  `Logger` now holds a `threading.Event`, so the gateway's post-processor
  drops its logger on pickle and `DataGateway.__setstate__` re-attaches the
  restored resource's logger.
- Parquet discovery tests no longer patch the removed
  `boti.core.secure_io.is_secure_path`.

## [1.3.11] - 2026-09-22

Last release before this changelog was kept. Sinks honour an injected
filesystem on remote paths and reject NUL bytes for every scheme; the
`dataframes` extra is named in the error when its dependencies are missing.

[Unreleased]: https://github.com/lvalverdeb/boti-data/compare/v1.3.11...HEAD
[1.3.11]: https://github.com/lvalverdeb/boti-data/releases/tag/v1.3.11

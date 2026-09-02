# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-02

### Added
- `comparison` check family: recompute the delta between two entities (`abs`, `pct`, `ratio`,
  `pp`) and diagnose the convention the claim actually matches when it mismatches.
- `higher_is_better` and `on_missing` keywords on `superlative` and `ranking_prefix`;
  `SparseColumnWarning` emitted under `on_missing="skip"`.
- `require_number` cell validation: `None`, strings, `bool`, NaN and inf are rejected with a
  message naming the fact, column and row.
- Release plumbing: dynamic version, `twine check`, tag-matches-version and wheel smoke steps in
  `release.yml`; `ruff format --check`, coverage and a real-adapters job in CI.
- `examples/campaign_summary.py`, `CONTRIBUTING.md`, false-positive issue template, badges.

### Changed (breaking)
- `Table.to_ranking` and table-backed `superlative` / `ranking_prefix` raise `ValueError` when any
  row lacks the ranked column. Previously those rows were dropped silently.
- `aggregate_reconciles` with `min` / `max` / `median` over a partial column raises
  `UnsupportedClaim`, matching `sum` / `mean`.
- `record_ranking` rejects duplicate names, non-numeric scores and non-finite scores.
- `entities_recorded` keeps native key types: an int-keyed table accepts an int claim, and
  `UnsupportedClaim.supported` may contain non-string members.
- `row_integrity` raises `ValueError` when the recorded cell is `None` or non-finite and the model
  claims a number.

### Fixed
- Table-backed checks always ranked descending; a "lowest wins" metric validated the worst row.

## [0.0.1] - 2026-09-02

Initial scaffold, never published to PyPI. Work between 2026-08-27 and 2026-09-02.

### Added
- `FactRegistry` with write-once facts, provenance (`tool` + `args`), and typed shapes:
  `Ranking` (`record_ranking`), `Table` (`record_table`, with `exhaustive=False` for truncated
  result sets), and `Domain` (`record_domain`). `FactRegistry.tool` decorator registers a tool
  function's return value.
- `boundary(...)`: context manager and decorator that declares which facts a block of output may
  reference, runs checks on `submit()`, and raises `FactBoundaryError` if the block exits
  unchecked. A boundary declaration is reusable across `with` blocks.
- Five check families, each with a `check_*` function and a `*(...)` builder for `boundary`:
  `superlative`, `ranking_prefix`, `aggregate_reconciles`, `entities_recorded`, `row_integrity`.
  `field_matches_fact` for simple field-to-fact equality.
- Errors: `FactBoundaryError`, `UnregisteredFact`, `UnsupportedClaim` (carries field, claimed
  value, supported value, fact name, provenance, and reason).
- Adapters for LangGraph nodes and MCP tool results (`groundplane.adapters`); the adapters import
  neither framework, so they are testable without them installed.
- Adversarial test suite named for real-world failures (swapped rows, tied argmax, blended entity
  names, truncated aggregates, interloper in a top-k list).
- CI on Python 3.10-3.14 with ruff and mypy; trusted-publishing release workflow; `py.typed`.

[Unreleased]: https://github.com/ong6/groundplane/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ong6/groundplane/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/ong6/groundplane/releases/tag/v0.0.1

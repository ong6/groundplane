# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed (breaking)
- `Ranking`, `Table` and `Domain` validate their invariants on construction, not only in the
  `record_*` methods. A hand-built `Ranking` passed to `record()` that is not ordered best-first
  (or is empty, repeats a name, or holds a non-finite or non-numeric score), a `Table` whose key
  is missing or repeated, and a `Domain` that is empty or holds non-strings now raise. Previously
  an out-of-order `Ranking` let `superlative` accept the runner-up as the winner.
- `Boundary.submit()` and `Boundary.report()` raise `FactBoundaryError` when the boundary has no
  checks. Previously any output passed and was returned as if verified.
- `aggregate_reconciles(where=...)` matches filter values with the library's equality rule, so
  `where={"flag": True}` no longer selects rows holding `1` (numbers still match across `int` and
  `float`).

### Fixed
- `entities_recorded` no longer accepts a composite name such as `(True, "x")` for the recorded
  `(1, "x")`.
- `row_integrity`'s "belongs to row" diagnosis uses the check's own matching rule and tolerance, so
  it no longer blames a row holding `[1]` for a claimed `[True]`.
- `field_matches_fact` compares mappings and sets in linear time; a 20,000-key fact took tens of
  seconds before.
- The sdist no longer ships the local `.hypothesis/` cache.
- README: install instructions no longer lead with `pip install groundplane`, which fails until the
  first PyPI release; the PyPI badge is gone; `record_state` documents that adopted state is only
  as trustworthy as the node that wrote it.

### Removed
- The unused internal `TOLERANCE_NOTE` constant.

### Hardening since 0.1.0 (2026-09-26, previously unlisted)
- Boundary checks see a read-only registry restricted to `facts=`; registration and reads deep-copy
  values and provenance arguments.
- Exact intermediate arithmetic for aggregates and comparisons; integers keep full precision;
  tolerances must be finite and non-negative.
- `FactRegistry.tool` captures positional arguments and defaults before the call and supports async
  callables; `guarded_node` supports async nodes; MCP text payloads reject non-finite numbers.

## [0.1.0] - 2026-09-02

### Added
- Named checks and report mode: every builder returns a `NamedCheck` (`Check` protocol with a
  `name`); `Boundary.report(output)` runs every check and returns a `Report` (`violations`,
  `passed`, `failed`, `ok`, `to_dict()`, `to_text()`); `submit(output, mode="all")` and
  `boundary(..., mode="all")` raise `ClaimsUnsupported` carrying that report;
  `UnsupportedClaim.to_dict()` is JSON-safe. Plain callables are still accepted as checks.
- `boundary(...)` is typed: it returns `BoundaryDeclaration`, so `with boundary(...) as b` gives a
  `Boundary` under mypy and the decorator form preserves the wrapped signature.
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

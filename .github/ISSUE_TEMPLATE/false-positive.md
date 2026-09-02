---
name: False positive
about: A check raised UnsupportedClaim on output that the recorded facts do support
title: "False positive: <check family> on <one-line description>"
labels: false-positive
---

## Recorded facts (with provenance)

The `record*` calls, verbatim. Include `tool=` and `args=` — provenance is part of the verdict.

```python
reg = FactRegistry()
reg.record_ranking(...)
```

## Submitted output

The exact mapping passed to `submit()` (or the checker), not a paraphrase.

```python
{"winner": ...}
```

## Check config

The `boundary(...)` / check builder call, including every keyword you set (tolerances,
`allow_tie_pick`, `column`, `exhaustive`, ...).

```python
boundary(reg, facts=[...], checks=[...])
```

## Expected verdict

What you believe should have happened (pass, or a different `UnsupportedClaim`) and why the
recorded facts support it.

## Actual error

Paste the full `UnsupportedClaim` message.

```
```

## Environment

- groundplane version:
- Python version:

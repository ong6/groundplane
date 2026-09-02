# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before you push

All four must pass; CI runs the same commands.

```bash
ruff check .
ruff format --check groundplane tests examples
mypy
pytest -q
```

## The one rule

**A new check family ships with adversarial tests named for the real-world failure it catches.**
Not "test_superlative_wrong_winner" but the actual thing that went wrong in production: the
second-best campaign reported as the winner, the attribute swap between neighbouring rows, the
top-k list with the right names in an invented order. If you cannot name the failure, the check
is not ready. See `tests/test_superlative_adversarial.py` for the pattern.

## Scope

Bounded, declared facts only. No model-based grading, no free-prose claim extraction. If a
proposal needs an LLM to decide whether the output is right, it belongs in a different library.

## Reporting a false positive

Use the "False positive" issue template. Include the recorded facts with provenance, the exact
submitted output, the check config, and the verdict you expected.

# groundplane

A hard line between the parts of an agent's output the model may generate and the parts that must
come from code — and a loud failure when the model crosses it.

**Status: v0, in development.** API unstable. Not yet on PyPI.

## The problem in 60 seconds

An agent calls tools, gets ground truth, then writes prose. The prose usually happens to match. When
it doesn't, the failure is silent and confident: the summary names the second-best campaign as the
winner, quotes a plausible number nobody computed, or invents an entity that was not in the result
set. Nothing throws. Nothing logs. The customer reads it.

`groundplane` makes the ground truth *declared*. Tool results are registered as typed facts
with provenance; a boundary names which facts a block of output may reference; a deterministic
checker validates the model's structured output against those facts. An unsupported claim raises.

Scope is deliberately narrow: **bounded, declared facts only**. Not a general hallucination detector,
and not model-based grading — an LLM judge would reintroduce the failure mode being removed.

## What it checks

Five families. Each one asks how the recorded facts relate to *each other*, which is what a
per-field validator cannot see. Every value in a swapped row is a real value, and a wrong
argmax is spelled the same as the right one.

| Check | Failure it catches |
|---|---|
| `superlative` | "Best X" names something other than the computed argmax, or quotes a score that is not the computed one. |
| `ranking_prefix` | A top-k list with the right names in an invented order, an interloper in the last slot, a repeated name, or a cut through a block of tied scores. |
| `aggregate_reconciles` | A total that does not sum, an unweighted mean presented as a weighted one, a filtered aggregate taken over the wrong subset, and any aggregate over a truncated table. |
| `entities_recorded` | A name blended out of two real ones, an id leaked from an earlier tool call, case drift, or an empty list smuggling in a "none qualified" claim. |
| `row_integrity` | Attribute swap: the right row named, with a neighbouring row's value in another field. |

## Flow

```mermaid
flowchart LR
    T["Tools<br/>(SQL, metrics, API)"] --> R["FactRegistry<br/>value + provenance"]
    R --> B["boundary(...)<br/>declares permitted facts"]
    M["Model<br/>structured output"] --> B
    B --> C{"Checker<br/>superlative · ordering<br/>aggregates · entities · rows"}
    C -->|claim resolves| P["pass — output returned"]
    C -->|claim unsupported| X["raise UnsupportedClaim"]
```

## Before / after

`superlative` is the narrowest of the five and the easiest to show.

Before — the winner is whatever the model wrote:

```python
rows = warehouse.query("select campaign, ctr from campaign_daily")
summary = llm(f"Which campaign performed best?\n{rows}")   # says "north". It was "harbour".
```

After — the argmax is computed in code, and the model can only phrase it:

```python
from groundplane import FactRegistry, boundary, superlative

reg = FactRegistry()
reg.record_ranking(
    "campaign_ctr",
    {"north": 0.0412, "harbour": 0.0455, "delta": 0.0301},
    key="ctr", tool="warehouse.query", args={"table": "campaign_daily", "window": "7d"},
)

with boundary(reg, facts=["campaign_ctr"], checks=[superlative(fact="campaign_ctr")]) as b:
    b.submit(llm_structured(facts=b.facts()))   # {"winner": "north", ...}
```

```
UnsupportedClaim: unsupported claim in field 'winner': model said 'north',
registered facts support 'harbour' | fact='campaign_ctr' |
provenance=warehouse.query(table='campaign_daily', window='7d') |
computed argmax on 'ctr' is 'harbour' (0.0455); 'north' ranked #2 at 0.0412
```

## Install

```bash
pip install -e ".[dev]"   # not yet published to PyPI
pytest
```

Requires Python >= 3.10. The core has no runtime dependencies.

## API

| Symbol | Purpose |
|---|---|
| `FactRegistry.record(name, value, tool=, args=)` | Register a typed fact with provenance. Write-once. |
| `FactRegistry.record_ranking(name, scores, key=, tool=, higher_is_better=)` | Compute an argmax ordering in code and register it. |
| `FactRegistry.record_table(name, rows, key=, tool=, columns=, exhaustive=)` | Register retrieved rows as a `Table`, keyed by one column. `exhaustive=False` marks a truncated result set. |
| `FactRegistry.record_domain(name, members, tool=, label=)` | Register a closed set of names as a `Domain`. |
| `FactRegistry.tool(name, fact=)` | Decorator that registers a tool function's return value. |
| `boundary(reg, facts=, checks=)` | Context manager **and** decorator. Verifies declared facts exist; runs checks on `submit()`; raises if a block exits unchecked. |
| `superlative(fact=, winner_field=, score_field=, rank_field=, tolerance=, rel_tolerance=, allow_tie_pick=, column=)` | The argmax guard. Reads a `Ranking`, or a `Table` with `column=`. |
| `field_matches_fact(field=, fact=)` | Assert an output field equals a registered value. |
| `ranking_prefix(fact=, field=, k=, ordered=, allow_tie_pick=, column=)` | Top-k guard: a claimed prefix must match the computed ordering, and the cut at k must not fall inside a tie. Reads a `Ranking` or a `Table` column. |
| `aggregate_reconciles(fact=, field=, column=, op=, where=, weight_column=, tolerance=, rel_tolerance=)` | Recompute a stated `sum`/`count`/`mean`/`min`/`max`/`median` over a recorded `Table`. Refuses a non-exhaustive table or a partial column. |
| `entities_recorded(fields=, fact=, path=, allow_empty=, case_sensitive=)` | Every name the model uses must come from a recorded `Domain`, `Ranking`, or `Table` key column. |
| `row_integrity(fact=, key_field=, fields=, tolerance=, rel_tolerance=)` | Resolve the named row first, then read every other field off that one row — catches attribute swap. |
| `UnsupportedClaim`, `UnregisteredFact` | Failures, both subclasses of `FactBoundaryError`. |

Each `X(...)` builder has an imperative twin, `check_X(registry, output, ...)`, callable outside a
boundary. Facts are write-once; one recorded `Table` can back the superlative, ranking, aggregate,
entity, and row checks at the same time.

Every numeric check takes `tolerance` (absolute) and `rel_tolerance` (relative). Both default to
exact: `rel_tolerance=None` means `rel_tol=0.0`, not `math.isclose`'s usual `1e-9`. A checker built
to catch a wrong number shouldn't forgive nine digits by default. Pass `rel_tolerance=1e-9` for the
stdlib behaviour.

Row keys keep their own type. An `int` id recorded by `record_table` stays an `int` all the way
through `to_ranking`, so a model answering `101` is compared against `101` and never `"101"`.

## Claim extraction: structured-output-first

v0 validates **declared fields**, not English. The model emits `{"winner": ..., "ctr": ...}` and the
checker resolves each field against the registry. Parsing prose would need either a brittle parser or
an LLM judge; constraining the model to fields makes the check total and deterministic. Prose
extraction, if it lands, sits on top of this layer — never instead of it. `Boundary.submit()` rejects
a plain string for that reason.

## Adversarial fixtures

`tests/test_superlative_adversarial.py` holds cases where the plausible model answer is provably
wrong: wrong winner, right winner with an invented score, unranked entity, a genuine tie flattened
into one winner, a min/max direction flip, a missing field, a numeric-looking string, and an
inconsistent rank. `test_ordering.py`, `test_aggregates.py`, `test_entities.py`, and `test_rows.py`
do the same for the checks above.

## Prior art

Checked 2026-08-27. Star counts as of that date.

| Project | What it validates | Where it stops |
|---|---|---|
| [Guardrails AI](https://github.com/guardrails-ai/guardrails) (7.3k★) | Validators over LLM output, with reask on violation; its provenance validators ask whether generated text is supported by a source document. | Support is decided by embedding similarity or an LLM judge, so the verdict is probabilistic and the input is prose. Nothing is compared against a value computed in code. |
| [Instructor](https://github.com/567-labs/instructor) (13.8k★) | Structured output with Pydantic validation and reask. | The substrate, not a competitor — `groundplane` checks the object Instructor gets back. A field validator sees one field at a time and cannot see a relation that spans several. |
| [agent-citation](https://dev.to/mukundakatta/agent-citation-track-where-every-agent-claim-came-from-3hbd) | That every claim requiring a source has one attached, kept in Python objects rather than re-asked of the model. | Presence, not agreement. A citation can sit on a claim the cited fact contradicts. |
| [GroundProbe](https://github.com/aureliocpr-ctrl/groundprobe) | Tags each claim grounded / inferred / speculative from n-gram, LCS and tf-idf overlap with the source, with an optional LLM tie-break near the thresholds. | Lexical overlap, and it reports a score rather than raising. A wrong argmax is spelled almost exactly like the right one. |

All four ask a version of "does this string appear in, or resemble, a source?". None validates a
**relational** property of the recorded facts — that the named winner is the argmax, that the cut at
k misses a tie, that a stated total reconciles, that the number printed against a row came off that
row. The argmax case on its own is about forty lines of Instructor validator. The reason to build a
library is the other four.

## License

MIT © 2026 Ong Jun Xiong

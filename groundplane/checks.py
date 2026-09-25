"""Checks that run at a boundary.

v0 is **structured-output-first**: the model emits declared fields (``{"winner": "b",
"score": 0.41}``), and the checker validates those fields against facts computed in
code. It does not parse English.

Rationale (spec open question, "how are claims extracted from prose"): parsing free
text needs either a parser that is wrong at the margins or an LLM judge, and an LLM
judge reintroduces exactly the failure mode this library removes. Constraining the
model to fields makes the check total and deterministic. Prose extraction, if it ever
lands, belongs on top of this layer and never instead of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._internal import is_number, require_field, resolve_ranking, values_close, values_equal
from .boundary import NamedCheck
from .errors import UnsupportedClaim
from .registry import FactRegistry, OnMissing, same_key

__all__ = ["superlative", "field_matches_fact", "check_superlative"]


def check_superlative(
    registry: FactRegistry,
    output: Mapping[str, Any],
    *,
    fact: str,
    winner_field: str = "winner",
    score_field: str | None = None,
    rank_field: str | None = None,
    tolerance: float = 0.0,
    rel_tolerance: float | None = None,
    allow_tie_pick: bool = False,
    column: str | None = None,
    higher_is_better: bool | None = None,
    on_missing: OnMissing = "raise",
) -> None:
    """Verify a claimed "best X" against the argmax the code computed.

    Raises :class:`UnsupportedClaim` when the model names a different winner, an
    entity that was never ranked, a score that does not match the computed score,
    or a sole winner where the data has a tie.

    ``fact`` may be a ``Ranking`` or, with ``column=``, a recorded ``Table`` — the
    same two shapes :func:`~groundplane.ordering.check_ranking_prefix` accepts.
    ``higher_is_better=False`` says the lowest value wins (latency, cost); it is
    required for a ``Table`` whose "best" is a minimum, and must agree with the
    recorded direction of a ``Ranking``. ``on_missing`` decides what a ``Table``
    row without ``column`` means: ``"raise"`` (default) refuses to rank a subset,
    ``"skip"`` ranks the rows that carry it and warns.

    ``tolerance`` is an absolute tolerance and ``rel_tolerance`` a relative one, using
    the :func:`math.isclose` rule without converting integers to floats.

    **Footgun, and deliberate:** ``rel_tolerance=None`` (the default) means ``rel_tol=0.0``,
    *not* :func:`math.isclose`'s usual ``1e-9``. A checker whose job is to catch a wrong
    number must default to exact comparison; silently forgiving the last nine digits is a
    policy, and a policy belongs in the caller's hands. Pass ``rel_tolerance=1e-9``
    explicitly to get the stdlib behaviour.
    """
    ranking = resolve_ranking(
        registry, fact, column, higher_is_better=higher_is_better, on_missing=on_missing
    )
    provenance = str(registry.get(fact).provenance)
    claimed = require_field(output, winner_field)

    if not any(same_key(claimed, name) for name in ranking.names):
        raise UnsupportedClaim(
            winner_field,
            claimed,
            ranking.winner,
            fact=fact,
            provenance=provenance,
            reason=f"{claimed!r} was never ranked; ranked entities: {list(ranking.names)}",
        )

    ties = ranking.ties
    if len(ties) > 1 and not allow_tie_pick:
        raise UnsupportedClaim(
            winner_field,
            claimed,
            list(ties),
            fact=fact,
            provenance=provenance,
            reason=(
                f"{len(ties)} entities tie on {ranking.key!r} at {ranking.winning_score!r}; "
                "no single winner is supported (pass allow_tie_pick=True to permit any of them)"
            ),
        )

    if not any(same_key(claimed, name) for name in ties):
        raise UnsupportedClaim(
            winner_field,
            claimed,
            ranking.winner,
            fact=fact,
            provenance=provenance,
            reason=(
                f"computed argmax on {ranking.key!r} is {ranking.winner!r} "
                f"({ranking.winning_score!r}); {claimed!r} ranked "
                f"#{ranking.rank_of(claimed)} at {ranking.score_of(claimed)!r}"
            ),
        )

    if score_field is not None:
        claimed_score = require_field(output, score_field)
        true_score = ranking.score_of(claimed)
        # claimed was established to be in ranking.names above, so it has a score.
        assert true_score is not None
        if not is_number(claimed_score):
            raise UnsupportedClaim(
                score_field,
                claimed_score,
                true_score,
                fact=fact,
                provenance=provenance,
                reason="claimed score is not a number",
            )
        if not values_close(
            claimed_score, true_score, tolerance=tolerance, rel_tolerance=rel_tolerance
        ):
            raise UnsupportedClaim(
                score_field,
                claimed_score,
                true_score,
                fact=fact,
                provenance=provenance,
                reason=(
                    f"score for {claimed!r} on {ranking.key!r} does not match "
                    f"(tol={tolerance}, rel_tol={rel_tolerance})"
                ),
            )

    if rank_field is not None:
        claimed_rank = require_field(output, rank_field)
        if type(claimed_rank) is not int or claimed_rank != 1:
            raise UnsupportedClaim(
                rank_field,
                claimed_rank,
                1,
                fact=fact,
                provenance=provenance,
                reason=f"{claimed!r} is the argmax, so its rank must be 1",
            )


def superlative(
    *,
    fact: str,
    winner_field: str = "winner",
    score_field: str | None = None,
    rank_field: str | None = None,
    tolerance: float = 0.0,
    rel_tolerance: float | None = None,
    allow_tie_pick: bool = False,
    column: str | None = None,
    higher_is_better: bool | None = None,
    on_missing: OnMissing = "raise",
) -> NamedCheck:
    """Build a boundary check from :func:`check_superlative`."""

    def check(registry: FactRegistry, output: Mapping[str, Any]) -> None:
        check_superlative(
            registry,
            output,
            fact=fact,
            winner_field=winner_field,
            score_field=score_field,
            rank_field=rank_field,
            tolerance=tolerance,
            rel_tolerance=rel_tolerance,
            allow_tie_pick=allow_tie_pick,
            column=column,
            higher_is_better=higher_is_better,
            on_missing=on_missing,
        )

    return NamedCheck(f"superlative(fact={fact!r}, field={winner_field!r})", check)


def field_matches_fact(*, field: str, fact: str, path: Sequence[str] = ()) -> NamedCheck:
    """Build a check asserting ``output[field]`` equals a registered fact's value."""

    def check(registry: FactRegistry, output: Mapping[str, Any]) -> None:
        value: Any = registry.value(fact)
        for step in path:
            value = value[step]
        claimed = require_field(output, field)
        if not values_equal(claimed, value):
            raise UnsupportedClaim(
                field,
                claimed,
                value,
                fact=fact,
                provenance=str(registry.get(fact).provenance),
            )

    return NamedCheck(f"field_matches_fact(fact={fact!r}, field={field!r})", check)

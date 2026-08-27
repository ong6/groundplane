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

from ._internal import is_number, require_field, values_close
from .boundary import Check
from .errors import UnsupportedClaim
from .registry import FactRegistry

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
) -> None:
    """Verify a claimed "best X" against the argmax the code computed.

    Raises :class:`UnsupportedClaim` when the model names a different winner, an
    entity that was never ranked, a score that does not match the computed score,
    or a sole winner where the data has a tie.

    ``tolerance`` is an absolute tolerance and ``rel_tolerance`` a relative one, passed
    straight to :func:`math.isclose` as ``abs_tol`` and ``rel_tol``.

    **Footgun, and deliberate:** ``rel_tolerance=None`` (the default) means ``rel_tol=0.0``,
    *not* :func:`math.isclose`'s usual ``1e-9``. A checker whose job is to catch a wrong
    number must default to exact comparison; silently forgiving the last nine digits is a
    policy, and a policy belongs in the caller's hands. Pass ``rel_tolerance=1e-9``
    explicitly to get the stdlib behaviour.
    """
    ranking = registry.ranking(fact)
    provenance = str(registry.get(fact).provenance)
    claimed = require_field(output, winner_field)

    if claimed not in ranking.names:
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

    if claimed not in ties:
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
        if claimed_rank != 1:
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
) -> Check:
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
        )

    return check


def field_matches_fact(*, field: str, fact: str, path: Sequence[str] = ()) -> Check:
    """Build a check asserting ``output[field]`` equals a registered fact's value."""

    def check(registry: FactRegistry, output: Mapping[str, Any]) -> None:
        value: Any = registry.value(fact)
        for step in path:
            value = value[step]
        claimed = require_field(output, field)
        if claimed != value:
            raise UnsupportedClaim(
                field,
                claimed,
                value,
                fact=fact,
                provenance=str(registry.get(fact).provenance),
            )

    return check

"""Top-k guard: a claimed ranking prefix must match the ordering computed in code.

The generalisation of the superlative guard. ``superlative`` answers "is the named
winner the argmax"; :func:`check_ranking_prefix` answers the same question for a list
of k names, including the part a per-field validator can never see — whether the cut
at k lands inside a block of tied scores, in which case "top 3" is not a fact.
"""

from __future__ import annotations

from typing import Any, Mapping, NoReturn, Sequence

from .boundary import Check
from .checks import _require_field
from .errors import UnsupportedClaim
from .registry import FactRegistry, Ranking, Table

__all__ = ["ranking_prefix", "check_ranking_prefix"]


def _resolve_ranking(registry: FactRegistry, fact: str, column: str | None) -> Ranking:
    """Return the ranking a check should run against, from a Ranking or Table fact."""
    value = registry.get(fact).value
    if isinstance(value, Ranking):
        if column is not None:
            raise ValueError(
                f"fact {fact!r} is already a Ranking; column={column!r} applies only to a Table fact"
            )
        return value
    if isinstance(value, Table):
        if column is None:
            raise ValueError(
                f"fact {fact!r} is a Table; pass column= to say which column to rank it by"
            )
        return value.to_ranking(column)
    raise TypeError(
        f"fact {fact!r} is a {registry.get(fact).type_name}, not a Ranking or Table; "
        "use record_ranking() or record_table()"
    )


def _blocks_for_positions(ranking: Ranking, n: int) -> tuple[tuple[str, ...], ...]:
    """The tie block covering each of the first ``n`` positions, position-aligned."""
    covering: list[tuple[str, ...]] = []
    for block in ranking.tie_blocks:
        for _ in block:
            if len(covering) == n:
                return tuple(covering)
            covering.append(block)
    return tuple(covering)


def check_ranking_prefix(
    registry: FactRegistry,
    output: Mapping[str, Any],
    *,
    fact: str,
    field: str,
    k: int | None = None,
    ordered: bool = True,
    allow_tie_pick: bool = False,
    column: str | None = None,
) -> None:
    """Verify a claimed top-k list against the ranking the code computed.

    Raises :class:`UnsupportedClaim` when the claim is not a list, has the wrong
    length for ``k``, repeats a name, names an entity that outranks nothing it was
    placed above, or cuts through a block of tied scores (unless ``allow_tie_pick``).
    """
    ranking = _resolve_ranking(registry, fact, column)
    provenance = str(registry.get(fact).provenance)
    claimed = _require_field(output, field)

    def fail(reason: str, supported: object) -> NoReturn:
        raise UnsupportedClaim(
            field, claimed, supported, fact=fact, provenance=provenance, reason=reason
        )

    if isinstance(claimed, (str, bytes)) or not isinstance(claimed, Sequence):
        fail(
            "claimed top-k is not a list of names; a bare string is not a ranking",
            list(ranking.names[: k or 1]),
        )
    items = list(claimed)

    if k is not None and len(items) != k:
        fail(
            f"claimed {len(items)} names under a top-{k} claim",
            list(ranking.names[:k]),
        )

    n = k if k is not None else len(items)
    expected = list(ranking.names[:n])

    if n == 0:
        fail("an empty list makes no verifiable ranking claim", expected)
    if n > len(ranking.names):
        fail(
            f"only {len(ranking.names)} entities were ranked on {ranking.key!r}",
            list(ranking.names),
        )

    seen: set[str] = set()
    for name in items:
        if not isinstance(name, str):
            fail(f"{name!r} is not an entity name", expected)
        if name in seen:
            fail(f"{name!r} appears twice; a ranking prefix cannot repeat a name", expected)
        seen.add(name)
        if name not in ranking.names:
            fail(f"{name!r} was never ranked; ranked entities: {list(ranking.names)}", expected)

    covering = _blocks_for_positions(ranking, n)
    last_block = covering[-1]
    covered = sum(1 for block in covering if block is last_block)
    cut_inside = covered < len(last_block)
    if cut_inside and not allow_tie_pick:
        fail(
            f"the cut at {n} falls inside a tie on {ranking.key!r} "
            f"({list(last_block)} share {ranking.score_of(last_block[0])!r}); "
            "top-k is undefined there (pass allow_tie_pick=True to permit any of them)",
            expected,
        )

    if ordered:
        for i, name in enumerate(items):
            if name not in covering[i]:
                fail(
                    f"{name!r} is #{ranking.rank_of(name)} on {ranking.key!r} "
                    f"({ranking.score_of(name)!r}); the computed top {n} is {expected}",
                    expected,
                )
        return

    allowed = {name for block in covering for name in block}
    required = {
        name
        for block in covering
        if not (block is last_block and cut_inside)
        for name in block
    }
    for name in items:
        if name not in allowed:
            fail(
                f"{name!r} is #{ranking.rank_of(name)} on {ranking.key!r} "
                f"({ranking.score_of(name)!r}); the computed top {n} is {expected}",
                expected,
            )
    missing = sorted(required - seen)
    if missing:
        fail(f"the computed top {n} on {ranking.key!r} also contains {missing}", expected)


def ranking_prefix(
    *,
    fact: str,
    field: str,
    k: int | None = None,
    ordered: bool = True,
    allow_tie_pick: bool = False,
    column: str | None = None,
) -> Check:
    """Build a boundary check from :func:`check_ranking_prefix`."""
    if k is not None and k < 1:
        raise ValueError(f"k must be at least 1, got {k!r}")

    def check(registry: FactRegistry, output: Mapping[str, Any]) -> None:
        check_ranking_prefix(
            registry,
            output,
            fact=fact,
            field=field,
            k=k,
            ordered=ordered,
            allow_tie_pick=allow_tie_pick,
            column=column,
        )

    return check

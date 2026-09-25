"""Aggregate reconciliation: a stated total must survive being recomputed.

The failure this catches is confident arithmetic over rows the model can see but
cannot add: totals that do not sum, unweighted means presented as weighted ones,
and filtered aggregates computed over the wrong subset. Every figure is recomputed
here from the recorded :class:`~groundplane.registry.Table`, never trusted.

A truncated table is never a valid base for an aggregate, so a fact recorded with
``exhaustive=False`` fails for every op — a sum over a paginated result set is a
silent lie no matter how well it is arithmetically formed.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from fractions import Fraction
from typing import Any, Literal

from ._internal import (
    is_number,
    numeric_result,
    offset_text,
    require_field,
    require_number,
    values_close,
)
from .boundary import NamedCheck
from .errors import UnsupportedClaim
from .registry import FactRegistry

__all__ = ["aggregate_reconciles", "check_aggregate_reconciles"]

Op = Literal["sum", "count", "mean", "min", "max", "median"]

_OPS: tuple[str, ...] = ("sum", "count", "mean", "min", "max", "median")


def _total(values: list[float]) -> Fraction:
    return sum((Fraction(value) for value in values), Fraction())


def _validate_config(op: Op, column: str | None, weight_column: str | None) -> None:
    """Reject wiring mistakes at builder time, not at model time."""
    if op not in _OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {list(_OPS)}")
    if op != "count" and column is None:
        raise ValueError(f"op {op!r} needs a column= to aggregate over")
    if weight_column is not None and op != "mean":
        raise ValueError(f"weight_column is only meaningful with op='mean', not {op!r}")


def _select(
    rows: tuple[Mapping[str, Any], ...],
    columns: tuple[str, ...],
    where: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    if not where:
        return rows
    for key in where:
        if key not in columns:
            raise KeyError(f"where column {key!r} not in table; columns: {list(columns)}")
    return tuple(row for row in rows if all(k in row and row[k] == v for k, v in where.items()))


def check_aggregate_reconciles(
    registry: FactRegistry,
    output: Mapping[str, Any],
    *,
    fact: str,
    field: str,
    column: str | None = None,
    op: Op = "sum",
    where: Mapping[str, Any] | None = None,
    weight_column: str | None = None,
    tolerance: float = 0.0,
    rel_tolerance: float | None = None,
) -> None:
    """Verify a claimed total, count, mean, or extremum against the recorded rows.

    Raises :class:`UnsupportedClaim` when the arithmetic does not reconcile, when
    the filter selects no rows, when the aggregated column is incomplete, or when
    the recorded table was not exhaustive. A recorded cell that is not a finite
    number where one is needed (``None``, a string, NaN) is a data problem rather
    than a model claim and raises ``ValueError`` naming the fact, column and row.

    ``tolerance`` is an absolute tolerance and ``rel_tolerance`` a relative one, using
    the :func:`math.isclose` rule without converting integers to floats.

    **Footgun, and deliberate:** ``rel_tolerance=None`` (the default) means ``rel_tol=0.0``,
    *not* :func:`math.isclose`'s usual ``1e-9``. A checker whose job is to catch a wrong
    number must default to exact comparison; silently forgiving the last nine digits is a
    policy, and a policy belongs in the caller's hands. Pass ``rel_tolerance=1e-9``
    explicitly to get the stdlib behaviour.
    """
    _validate_config(op, column, weight_column)
    table = registry.table(fact)
    provenance = str(registry.get(fact).provenance)
    tool = registry.get(fact).provenance.tool
    claimed = require_field(output, field)

    if not table.exhaustive:
        raise UnsupportedClaim(
            field,
            claimed,
            "no verifiable aggregate",
            fact=fact,
            provenance=provenance,
            reason=(
                f"the recorded table was not exhaustive (from {tool}); an aggregate over a "
                "paginated, capped, or filtered result set cannot be verified"
            ),
        )

    rows = _select(table.rows, table.columns, where)
    if not rows:
        raise UnsupportedClaim(
            field,
            claimed,
            "no rows",
            fact=fact,
            provenance=provenance,
            reason=(
                f"filter {dict(where or {})!r} selected no rows out of {len(table.rows)}; "
                "a claim about an empty set is not verifiable"
            ),
        )

    if op == "count":
        computed_count = len(rows)
        if not isinstance(claimed, int) or isinstance(claimed, bool):
            raise UnsupportedClaim(
                field,
                claimed,
                computed_count,
                fact=fact,
                provenance=provenance,
                reason=f"a count is an integer claim; {claimed!r} is a {type(claimed).__name__}",
            )
        if claimed != computed_count:
            raise UnsupportedClaim(
                field,
                claimed,
                computed_count,
                fact=fact,
                provenance=provenance,
                reason=(
                    f"count over the selected rows is {computed_count}; "
                    f"model said {claimed} (off by {claimed - computed_count:+d})"
                ),
            )
        return

    assert column is not None  # guaranteed by _validate_config
    if column not in table.columns:
        raise KeyError(f"column {column!r} not in table; columns: {list(table.columns)}")

    # No op survives a hole in its column. A sum or mean under-counts, obviously;
    # but a min, max or median is just as exposed, because the row without a value
    # is exactly the one that might hold the extremum or shift the midpoint. The
    # honest answer over a partial column is "not verifiable", for every op.
    present = [row for row in rows if column in row]
    missing = len(rows) - len(present)
    if missing:
        raise UnsupportedClaim(
            field,
            claimed,
            "no verifiable aggregate",
            fact=fact,
            provenance=provenance,
            reason=(
                f"{missing} of {len(rows)} selected rows have no {column!r}; a {op} over a "
                "partial column would be a silent lie (the missing row could be the one "
                "that changes it)"
            ),
        )

    values = [
        require_number(row[column], column=column, key=row[table.key], fact=fact) for row in present
    ]

    naive_mean: float | None = None
    if op == "sum":
        computed = numeric_result(_total(values))
    elif op == "min":
        computed = min(values)
    elif op == "max":
        computed = max(values)
    elif op == "median":
        ordered = sorted(values)
        middle = len(ordered) // 2
        computed = (
            ordered[middle]
            if len(ordered) % 2
            else numeric_result((Fraction(ordered[middle - 1]) + Fraction(ordered[middle])) / 2)
        )
    else:  # mean
        mean_fraction = _total(values) / len(values)
        if weight_column is None:
            computed = numeric_result(mean_fraction)
        else:
            # This figure is only a diagnostic. Failure to represent it must
            # not veto a representable weighted result.
            with suppress(ValueError):
                naive_mean = numeric_result(mean_fraction)
            if weight_column not in table.columns:
                raise KeyError(
                    f"column {weight_column!r} not in table; columns: {list(table.columns)}"
                )
            weights = [
                require_number(
                    row[weight_column], column=weight_column, key=row[table.key], fact=fact
                )
                for row in present
                if weight_column in row
            ]
            if len(weights) != len(present):
                raise UnsupportedClaim(
                    field,
                    claimed,
                    "no verifiable aggregate",
                    fact=fact,
                    provenance=provenance,
                    reason=f"some selected rows have no weight column {weight_column!r}",
                )
            total_weight = _total(weights)
            if total_weight == 0:
                raise UnsupportedClaim(
                    field,
                    claimed,
                    "no verifiable aggregate",
                    fact=fact,
                    provenance=provenance,
                    reason=(
                        f"total weight on {weight_column!r} is zero; a weighted mean is undefined"
                    ),
                )
            weighted_total = sum(
                (Fraction(v) * Fraction(w) for v, w in zip(values, weights, strict=True)),
                Fraction(),
            )
            computed = numeric_result(weighted_total / total_weight)

    if not is_number(claimed):
        raise UnsupportedClaim(
            field,
            claimed,
            computed,
            fact=fact,
            provenance=provenance,
            reason=f"claimed {op} is not a number",
        )

    if values_close(claimed, computed, tolerance=tolerance, rel_tolerance=rel_tolerance):
        return

    reason = (
        f"{op} of {column!r} over {len(present)} rows is {computed!r}; "
        f"model said {claimed!r} (off by {offset_text(claimed, computed)})"
    )
    if weight_column is not None and naive_mean is not None:
        weighted_note = f" weighted by {weight_column!r} = {computed!r}"
        if values_close(claimed, naive_mean, tolerance=tolerance, rel_tolerance=rel_tolerance):
            reason += (
                f"; the claim matches the unweighted mean {naive_mean!r}, so the model averaged "
                f"the per-row figures instead of{weighted_note}"
            )
        else:
            reason += f"; unweighted mean is {naive_mean!r},{weighted_note}"
    raise UnsupportedClaim(
        field,
        claimed,
        computed,
        fact=fact,
        provenance=provenance,
        reason=reason,
    )


def aggregate_reconciles(
    *,
    fact: str,
    field: str,
    column: str | None = None,
    op: Op = "sum",
    where: Mapping[str, Any] | None = None,
    weight_column: str | None = None,
    tolerance: float = 0.0,
    rel_tolerance: float | None = None,
) -> NamedCheck:
    """Build a boundary check from :func:`check_aggregate_reconciles`."""
    _validate_config(op, column, weight_column)

    def check(registry: FactRegistry, output: Mapping[str, Any]) -> None:
        check_aggregate_reconciles(
            registry,
            output,
            fact=fact,
            field=field,
            column=column,
            op=op,
            where=where,
            weight_column=weight_column,
            tolerance=tolerance,
            rel_tolerance=rel_tolerance,
        )

    return NamedCheck(f"aggregate_reconciles(fact={fact!r}, field={field!r})", check)

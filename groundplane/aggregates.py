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

import math
import statistics
from typing import Any, Literal, Mapping

from .boundary import Check
from .checks import _require_field
from .errors import UnsupportedClaim
from .registry import FactRegistry

__all__ = ["aggregate_reconciles", "check_aggregate_reconciles"]

Op = Literal["sum", "count", "mean", "min", "max", "median"]

_OPS: tuple[str, ...] = ("sum", "count", "mean", "min", "max", "median")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    return tuple(row for row in rows if all(row.get(k) == v for k, v in where.items()))


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
    the recorded table was not exhaustive.
    """
    _validate_config(op, column, weight_column)
    table = registry.table(fact)
    provenance = str(registry.get(fact).provenance)
    tool = registry.get(fact).provenance.tool
    claimed = _require_field(output, field)

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

    present = [row for row in rows if column in row]
    missing = len(rows) - len(present)
    if missing and op in ("sum", "mean"):
        raise UnsupportedClaim(
            field,
            claimed,
            "no verifiable aggregate",
            fact=fact,
            provenance=provenance,
            reason=(
                f"{missing} of {len(rows)} selected rows have no {column!r}; a {op} over a "
                "partial column would be a silent lie"
            ),
        )

    values = [float(row[column]) for row in present]
    if not values:
        raise UnsupportedClaim(
            field,
            claimed,
            "no rows",
            fact=fact,
            provenance=provenance,
            reason=f"no selected row carries column {column!r}",
        )

    naive_mean: float | None = None
    if op == "sum":
        computed = math.fsum(values)
    elif op == "min":
        computed = min(values)
    elif op == "max":
        computed = max(values)
    elif op == "median":
        computed = statistics.median(values)
    else:  # mean
        naive_mean = math.fsum(values) / len(values)
        if weight_column is None:
            computed = naive_mean
        else:
            if weight_column not in table.columns:
                raise KeyError(
                    f"column {weight_column!r} not in table; columns: {list(table.columns)}"
                )
            weights = [float(row[weight_column]) for row in present if weight_column in row]
            if len(weights) != len(present):
                raise UnsupportedClaim(
                    field,
                    claimed,
                    "no verifiable aggregate",
                    fact=fact,
                    provenance=provenance,
                    reason=f"some selected rows have no weight column {weight_column!r}",
                )
            total_weight = math.fsum(weights)
            if total_weight == 0:
                raise UnsupportedClaim(
                    field,
                    claimed,
                    "no verifiable aggregate",
                    fact=fact,
                    provenance=provenance,
                    reason=(
                        f"total weight on {weight_column!r} is zero; "
                        "a weighted mean is undefined"
                    ),
                )
            computed = math.fsum(v * w for v, w in zip(values, weights)) / total_weight

    if not _is_number(claimed):
        raise UnsupportedClaim(
            field,
            claimed,
            computed,
            fact=fact,
            provenance=provenance,
            reason=f"claimed {op} is not a number",
        )

    if math.isclose(float(claimed), computed, abs_tol=tolerance, rel_tol=rel_tolerance or 0.0):
        return

    delta = float(claimed) - computed
    reason = (
        f"{op} of {column!r} over {len(present)} rows is {computed:g}; "
        f"model said {claimed:g} (off by {delta:+g})"
    )
    if weight_column is not None and naive_mean is not None:
        weighted_note = f" weighted by {weight_column!r} = {computed:g}"
        rel = rel_tolerance or 0.0
        if math.isclose(float(claimed), naive_mean, abs_tol=tolerance, rel_tol=rel):
            reason += (
                f"; the claim matches the unweighted mean {naive_mean:g}, so the model averaged "
                f"the per-row figures instead of{weighted_note}"
            )
        else:
            reason += f"; unweighted mean is {naive_mean:g},{weighted_note}"
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
) -> Check:
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

    return check

"""Row integrity: every reported field must come from the *same* recorded row.

The failure this catches is attribute swap. The model names one row and then fills the
remaining fields with values lifted from a neighbouring one. Every field individually
exists in the data, so per-field validation passes and a human reader sees nothing
wrong. Only a check that resolves the key first, then reads the other columns off that
one row, can see it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._internal import is_number, require_field, values_close
from .boundary import Check
from .errors import UnsupportedClaim
from .registry import FactRegistry, Table

__all__ = ["row_integrity", "check_row_integrity"]


def _values_match(
    claimed: Any,
    recorded: Any,
    *,
    tolerance: float,
    rel_tolerance: float | None,
) -> bool:
    if is_number(claimed) and is_number(recorded):
        return values_close(claimed, recorded, tolerance=tolerance, rel_tolerance=rel_tolerance)
    if is_number(claimed) != is_number(recorded):
        return False
    return bool(claimed == recorded)


def _other_row_with(table: Table, column: str, value: Any, *, exclude: Any) -> Any | None:
    """The key of another row carrying ``value`` in ``column``, if one exists."""
    for row in table.rows:
        if row[table.key] == exclude or column not in row:
            continue
        recorded = row[column]
        if is_number(value) != is_number(recorded):
            continue
        if recorded == value:
            return row[table.key]
    return None


def check_row_integrity(
    registry: FactRegistry,
    output: Mapping[str, Any],
    *,
    fact: str,
    key_field: str,
    fields: Mapping[str, str],
    tolerance: float = 0.0,
    rel_tolerance: float | None = None,
) -> None:
    """Verify that every mapped output field matches the row the model named.

    ``key_field`` is the output field naming the row; ``fields`` maps output field to
    table column. Raises :class:`UnsupportedClaim` when the named row is not in the
    recorded table, when a declared field is absent from the output, or when any value
    disagrees with the resolved row — naming the row the value actually came from when
    it belongs to a different one.

    ``tolerance`` is an absolute tolerance and ``rel_tolerance`` a relative one, passed
    straight to :func:`math.isclose` as ``abs_tol`` and ``rel_tol``.

    **Footgun, and deliberate:** ``rel_tolerance=None`` (the default) means ``rel_tol=0.0``,
    *not* :func:`math.isclose`'s usual ``1e-9``. A checker whose job is to catch a wrong
    number must default to exact comparison; silently forgiving the last nine digits is a
    policy, and a policy belongs in the caller's hands. Pass ``rel_tolerance=1e-9``
    explicitly to get the stdlib behaviour.
    """
    table = registry.table(fact)
    provenance = str(registry.get(fact).provenance)
    claimed_key = require_field(output, key_field)

    row = table.row(claimed_key)
    if row is None:
        raise UnsupportedClaim(
            key_field,
            claimed_key,
            list(table.keys),
            fact=fact,
            provenance=provenance,
            reason=(
                f"{claimed_key!r} is not a row in the recorded table; "
                f"recorded keys: {list(table.keys)}"
            ),
        )

    for out_field, column in fields.items():
        if column not in table.columns:
            raise KeyError(f"column {column!r} not in table; columns: {list(table.columns)}")
        claimed = require_field(output, out_field)
        recorded = row[column]
        if _values_match(claimed, recorded, tolerance=tolerance, rel_tolerance=rel_tolerance):
            continue
        source = _other_row_with(table, column, claimed, exclude=claimed_key)
        detail = (
            f"; {claimed!r} belongs to row {source!r}"
            if source is not None
            else "; that value is in no recorded row"
        )
        raise UnsupportedClaim(
            out_field,
            claimed,
            recorded,
            fact=fact,
            provenance=provenance,
            reason=(
                f"row {claimed_key!r} has {column}={recorded!r}{detail} "
                f"(tol={tolerance}, rel_tol={rel_tolerance})"
            ),
        )


def row_integrity(
    *,
    fact: str,
    key_field: str,
    fields: Mapping[str, str],
    tolerance: float = 0.0,
    rel_tolerance: float | None = None,
) -> Check:
    """Build a boundary check from :func:`check_row_integrity`."""

    def check(registry: FactRegistry, output: Mapping[str, Any]) -> None:
        check_row_integrity(
            registry,
            output,
            fact=fact,
            key_field=key_field,
            fields=fields,
            tolerance=tolerance,
            rel_tolerance=rel_tolerance,
        )

    return check

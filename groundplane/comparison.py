"""Comparative claims: "A beat B by X" must survive being recomputed.

The failure this catches is a delta that is arithmetically *a* difference between the
two entities, just not the one the field declares. The four conventions — absolute
difference, percent change, ratio, and percentage points — all describe the same pair
of numbers, and a model slides between them freely: "up 12%" when the data says a ratio
of 1.12, "up 0.4%" when the gap is 0.4 percentage points, or the right magnitude with
the sign flipped because it subtracted in the other order. Each is a plausible number
that a reader cannot distinguish from the true one, so the check recomputes the delta
under the declared convention and, on a miss, says which other convention the claim
does match.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from ._internal import is_number, require_field, values_close
from .boundary import NamedCheck
from .errors import UnsupportedClaim
from .registry import FactRegistry, Ranking, Table

__all__ = ["comparison", "check_comparison"]

Kind = Literal["abs", "pct", "ratio", "pp"]
Direction = Literal["a_minus_b", "b_minus_a"]

_KINDS: tuple[str, ...] = ("abs", "pct", "ratio", "pp")
_DIRECTIONS: tuple[str, ...] = ("a_minus_b", "b_minus_a")


def _validate_config(kind: Kind, direction: Direction) -> None:
    """Reject wiring mistakes at builder time, not at model time."""
    if kind not in _KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {list(_KINDS)}")
    if direction not in _DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}; expected one of {list(_DIRECTIONS)}")


def _delta(kind: str, a: float, b: float) -> float | None:
    """The delta of ``a`` against base ``b`` under ``kind``; ``None`` when undefined.

    ``pct`` and ``ratio`` divide by the base, so a zero base has no delta at all —
    not infinity, not zero — and the caller must refuse the claim rather than pick
    a value to compare against.
    """
    if kind == "abs":
        return a - b
    if kind == "pp":
        return (a - b) * 100
    if b == 0:
        return None
    if kind == "pct":
        return (a - b) / b * 100
    return a / b


def _lookup(registry: FactRegistry, fact: str, column: str | None) -> tuple[dict[Any, float], str]:
    """The entity -> value map a comparison runs over, plus the metric it names.

    A ``Ranking`` already carries scores; a ``Table`` needs ``column=`` to say which
    figure is being compared. Passing ``column`` for a ``Ranking``, or omitting it
    for a ``Table``, is a wiring mistake and raises ``ValueError`` — never
    ``UnsupportedClaim``, because the model did nothing wrong.
    """
    value = registry.as_type(fact, Ranking, Table)
    if isinstance(value, Ranking):
        if column is not None:
            raise ValueError(
                f"fact {fact!r} is already a Ranking; column={column!r} applies only "
                "to a Table fact"
            )
        return dict(value.items), value.key
    if column is None:
        raise ValueError(f"fact {fact!r} is a Table; pass column= to say which column to compare")
    if column not in value.columns:
        raise KeyError(f"column {column!r} not in table; columns: {list(value.columns)}")
    return {row[value.key]: row[column] for row in value.rows if column in row}, column


def check_comparison(
    registry: FactRegistry,
    output: Mapping[str, Any],
    *,
    fact: str,
    a_field: str,
    b_field: str,
    delta_field: str,
    column: str | None = None,
    kind: Kind = "abs",
    direction: Direction = "a_minus_b",
    tolerance: float = 0.0,
    rel_tolerance: float | None = None,
) -> None:
    """Verify a claimed delta between two named entities against the recorded values.

    ``a_field`` and ``b_field`` name the two entities in the model output; the delta
    in ``delta_field`` is recomputed from the recorded values under ``kind``:

    - ``abs``: ``a - b``
    - ``pct``: ``(a - b) / b * 100`` — percent change relative to ``b``
    - ``ratio``: ``a / b``
    - ``pp``: ``(a - b) * 100`` — percentage points, for values recorded as fractions

    ``direction="b_minus_a"`` swaps the roles of the two entities (``b`` becomes the
    numerator or minuend). Raises :class:`UnsupportedClaim` when either entity is not
    in the fact, when the two fields name the same entity, when the delta is undefined
    (zero base for ``pct``/``ratio``), when the claimed delta is not a number, or when
    it does not reconcile — in which case the reason names any other convention or
    direction the claim *does* match, since that is almost always what went wrong.

    ``fact`` may be a ``Ranking`` or, with ``column=``, a recorded ``Table``.

    ``tolerance`` is an absolute tolerance and ``rel_tolerance`` a relative one, passed
    straight to :func:`math.isclose` as ``abs_tol`` and ``rel_tol``.

    **Footgun, and deliberate:** ``rel_tolerance=None`` (the default) means ``rel_tol=0.0``,
    *not* :func:`math.isclose`'s usual ``1e-9``. A checker whose job is to catch a wrong
    number must default to exact comparison; silently forgiving the last nine digits is a
    policy, and a policy belongs in the caller's hands. Pass ``rel_tolerance=1e-9``
    explicitly to get the stdlib behaviour.
    """
    _validate_config(kind, direction)
    values, metric = _lookup(registry, fact, column)
    provenance = str(registry.get(fact).provenance)
    a_name = require_field(output, a_field)
    b_name = require_field(output, b_field)
    claimed = require_field(output, delta_field)

    for field, name in ((a_field, a_name), (b_field, b_name)):
        if name not in values:
            raise UnsupportedClaim(
                field,
                name,
                list(values),
                fact=fact,
                provenance=provenance,
                reason=f"{name!r} is not in the fact; known entities: {list(values)}",
            )
        if not is_number(values[name]):
            raise UnsupportedClaim(
                delta_field,
                claimed,
                "no verifiable comparison",
                fact=fact,
                provenance=provenance,
                reason=f"{metric!r} for {name!r} is {values[name]!r}, not a number",
            )

    if a_name == b_name:
        raise UnsupportedClaim(
            delta_field,
            claimed,
            "no verifiable comparison",
            fact=fact,
            provenance=provenance,
            reason=(
                f"{a_field!r} and {b_field!r} both name {a_name!r}; an entity compared "
                "with itself is not a claim about the data"
            ),
        )

    a = float(values[a_name])
    b = float(values[b_name])
    lhs, rhs = (a, b) if direction == "a_minus_b" else (b, a)
    computed = _delta(kind, lhs, rhs)
    if computed is None:
        base = b_name if direction == "a_minus_b" else a_name
        raise UnsupportedClaim(
            delta_field,
            claimed,
            "no verifiable comparison",
            fact=fact,
            provenance=provenance,
            reason=(
                f"{metric!r} for {base!r} is 0, so a {kind} comparison against it is "
                "undefined whatever the claim"
            ),
        )

    if not is_number(claimed):
        raise UnsupportedClaim(
            delta_field,
            claimed,
            computed,
            fact=fact,
            provenance=provenance,
            reason="claimed delta is not a number",
        )

    if values_close(claimed, computed, tolerance=tolerance, rel_tolerance=rel_tolerance):
        return

    reason = (
        f"{kind} delta of {metric!r}, {a_name!r} ({a:g}) vs {b_name!r} ({b:g}) "
        f"{direction}, is {computed:g}; model said {claimed:g} "
        f"(off by {float(claimed) - computed:+g})"
    )
    reason += _diagnose(
        claimed,
        a,
        b,
        kind=kind,
        direction=direction,
        tolerance=tolerance,
        rel_tolerance=rel_tolerance,
    )
    raise UnsupportedClaim(
        delta_field,
        claimed,
        computed,
        fact=fact,
        provenance=provenance,
        reason=reason,
    )


def _diagnose(
    claimed: float,
    a: float,
    b: float,
    *,
    kind: str,
    direction: str,
    tolerance: float,
    rel_tolerance: float | None,
) -> str:
    """Name the convention the claim *does* fit, if any, so the reask is precise.

    Same kind with the direction reversed is checked first: a sign flip is the most
    common slip and the least ambiguous. Then every other kind in the declared
    direction, then the reversed direction. The first hit wins; listing every
    coincidence would bury the likely cause under the unlikely ones.
    """
    reversed_direction = "b_minus_a" if direction == "a_minus_b" else "a_minus_b"
    candidates: list[tuple[str, str]] = [(kind, reversed_direction)]
    candidates += [(k, direction) for k in _KINDS if k != kind]
    candidates += [(k, reversed_direction) for k in _KINDS if k != kind]
    for other_kind, other_direction in candidates:
        lhs, rhs = (a, b) if other_direction == "a_minus_b" else (b, a)
        alt = _delta(other_kind, lhs, rhs)
        if alt is None or not values_close(
            claimed, alt, tolerance=tolerance, rel_tolerance=rel_tolerance
        ):
            continue
        if other_kind == kind:
            return f"; claim matches {other_direction}; declared {direction}"
        if other_direction == direction:
            return (
                f"; claim {claimed:g} equals the {other_kind} delta; field declared kind={kind!r}"
            )
        return (
            f"; claim {claimed:g} equals the {other_kind} delta {other_direction}; "
            f"field declared kind={kind!r}, {direction}"
        )
    return ""


def comparison(
    *,
    fact: str,
    a_field: str,
    b_field: str,
    delta_field: str,
    column: str | None = None,
    kind: Kind = "abs",
    direction: Direction = "a_minus_b",
    tolerance: float = 0.0,
    rel_tolerance: float | None = None,
) -> NamedCheck:
    """Build a boundary check from :func:`check_comparison`."""
    _validate_config(kind, direction)

    def check(registry: FactRegistry, output: Mapping[str, Any]) -> None:
        check_comparison(
            registry,
            output,
            fact=fact,
            a_field=a_field,
            b_field=b_field,
            delta_field=delta_field,
            column=column,
            kind=kind,
            direction=direction,
            tolerance=tolerance,
            rel_tolerance=rel_tolerance,
        )

    return NamedCheck(f"comparison(fact={fact!r}, delta_field={delta_field!r})", check)

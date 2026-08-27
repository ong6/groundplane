"""Shared internals for the check modules. Not part of the public API.

Everything here exists because more than one check needs it and two copies would
drift. Nothing in this module is exported from ``groundplane``; import it from a
check module, never from user code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .errors import UnsupportedClaim
from .registry import FactRegistry, Ranking, Table

__all__ = [
    "TOLERANCE_NOTE",
    "is_number",
    "require_field",
    "resolve_ranking",
    "values_close",
]

TOLERANCE_NOTE = """\
``tolerance`` is an absolute tolerance and ``rel_tolerance`` a relative one, passed
straight to :func:`math.isclose` as ``abs_tol`` and ``rel_tol``.

**Footgun, and deliberate:** ``rel_tolerance=None`` (the default) means ``rel_tol=0.0``,
*not* :func:`math.isclose`'s usual ``1e-9``. A checker whose job is to catch a wrong
number must default to exact comparison; silently forgiving the last nine digits is a
policy, and a policy belongs in the caller's hands. Pass ``rel_tolerance=1e-9``
explicitly to get the stdlib behaviour."""


def resolve_ranking(registry: FactRegistry, fact: str, column: str | None) -> Ranking:
    """The ranking a check should run against, from a ``Ranking`` or ``Table`` fact.

    Every ordering check takes the same two shapes, so they resolve them the same
    way: a ``Ranking`` is already an ordering, and a ``Table`` becomes one by
    naming the ``column`` to rank it by. Passing ``column`` for a ``Ranking``, or
    omitting it for a ``Table``, is a wiring mistake and raises ``ValueError`` —
    never ``UnsupportedClaim``, because the model did nothing wrong.
    """
    value = registry.as_type(fact, Ranking, Table)
    if isinstance(value, Ranking):
        if column is not None:
            raise ValueError(
                f"fact {fact!r} is already a Ranking; column={column!r} applies only "
                "to a Table fact"
            )
        return value
    if column is None:
        raise ValueError(
            f"fact {fact!r} is a Table; pass column= to say which column to rank it by"
        )
    return value.to_ranking(column)


def is_number(value: Any) -> bool:
    """True for real numbers only; ``bool`` is a type-confusion vector, not a number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def require_field(output: Mapping[str, Any], field: str) -> Any:
    """Read ``field`` off the model's structured output, or fail as a claim.

    A field the model never emitted is a boundary failure, not a ``KeyError``: the
    caller declared that the block would assert something, and it did not.
    """
    if field not in output:
        raise UnsupportedClaim(
            field,
            claimed=None,
            supported="a value",
            reason="the model output declared no such field, so the claim cannot be verified",
        )
    return output[field]


def values_close(
    claimed: float,
    computed: float,
    *,
    tolerance: float = 0.0,
    rel_tolerance: float | None = None,
) -> bool:
    """Numeric comparison with the library's tolerance policy.

    See :data:`TOLERANCE_NOTE` for why ``rel_tolerance=None`` means exact, not ``1e-9``.
    """
    return math.isclose(
        float(claimed), float(computed), abs_tol=tolerance, rel_tol=rel_tolerance or 0.0
    )

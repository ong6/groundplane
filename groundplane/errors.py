"""Exceptions raised when model output crosses the fact boundary."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .boundary import Report

__all__ = ["ClaimsUnsupported", "FactBoundaryError", "UnregisteredFact", "UnsupportedClaim"]


def _json_safe(value: object) -> Any:
    """Return ``value`` if ``json.dumps`` accepts it, else its ``repr``."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)
    return value


class FactBoundaryError(Exception):
    """Base class for all fact-boundary failures."""


class UnregisteredFact(FactBoundaryError):
    """A fact was referenced that the registry does not hold.

    Raised when a boundary permits a fact name that was never registered, or when
    a checker is pointed at a fact key that does not exist.
    """

    def __init__(self, name: str, known: list[str] | None = None) -> None:
        self.name = name
        self.known = sorted(known or [])
        detail = f"; registered facts: {self.known}" if self.known else "; registry is empty"
        super().__init__(f"fact {name!r} is not registered{detail}")


class UnsupportedClaim(FactBoundaryError):
    """Model output asserted something the registered facts do not support.

    Carries the field that failed, what the model claimed, what the code computed,
    and the provenance of the computed fact, so the error message is actionable
    without reopening a trace.
    """

    def __init__(
        self,
        field: str,
        claimed: object,
        supported: object,
        *,
        fact: str | None = None,
        provenance: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.field = field
        self.claimed = claimed
        self.supported = supported
        self.fact = fact
        self.provenance = provenance
        self.reason = reason
        parts = [
            f"unsupported claim in field {field!r}: model said {claimed!r}, "
            f"registered facts support {supported!r}"
        ]
        if fact:
            parts.append(f"fact={fact!r}")
        if provenance:
            parts.append(f"provenance={provenance}")
        if reason:
            parts.append(reason)
        super().__init__(" | ".join(parts))

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe rendering; values ``json.dumps`` rejects are stored as their repr."""
        return {
            "field": self.field,
            "claimed": _json_safe(self.claimed),
            "supported": _json_safe(self.supported),
            "fact": self.fact,
            "provenance": self.provenance,
            "reason": self.reason,
        }


class ClaimsUnsupported(FactBoundaryError):
    """Raised by ``submit(output, mode="all")`` when one or more checks failed.

    ``report`` is the full :class:`~groundplane.boundary.Report`; ``str()`` of the
    exception is ``report.to_text()``, one line per violation.
    """

    def __init__(self, report: Report) -> None:
        self.report = report
        super().__init__(report.to_text())

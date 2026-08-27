"""Exceptions raised when model output crosses the fact boundary."""

from __future__ import annotations

__all__ = ["FactBoundaryError", "UnregisteredFact", "UnsupportedClaim"]


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

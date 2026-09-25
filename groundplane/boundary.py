"""Boundary declaration: which registered facts a block of model output may use.

A boundary is both a context manager and a decorator. Entering it asserts that every
permitted fact actually exists; leaving it runs the attached checks against whatever
structured output the block produced. Output that never gets submitted is itself a
failure, because an unchecked block is an unguarded one.

Two ways to run the checks. ``submit(output)`` stops at the first
:class:`~groundplane.errors.UnsupportedClaim` (``mode="first"``); ``report(output)``
runs every check and returns a :class:`Report` listing each violation, which is the
shape a reask prompt wants. ``submit(output, mode="all")`` does the second and raises
:class:`~groundplane.errors.ClaimsUnsupported` carrying the report.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from types import TracebackType
from typing import Any, Literal, ParamSpec, Protocol

from .errors import ClaimsUnsupported, FactBoundaryError, UnregisteredFact, UnsupportedClaim
from .registry import FactRegistry

__all__ = [
    "Boundary",
    "BoundaryDeclaration",
    "Check",
    "CheckFn",
    "Mode",
    "NamedCheck",
    "Report",
    "boundary",
]

P = ParamSpec("P")

Mode = Literal["first", "all"]

CheckFn = Callable[[FactRegistry, Mapping[str, Any]], None]
"""A bare check: a callable taking the registry and the model's output."""


class Check(Protocol):
    """A check with a name, as the builders (``superlative(...)`` etc.) return.

    ``name`` is what a :class:`Report` lists under ``passed`` / ``failed``. Plain
    callables are accepted anywhere a ``Check`` is, and are named after their
    ``__name__``.
    """

    @property
    def name(self) -> str: ...

    def __call__(self, registry: FactRegistry, output: Mapping[str, Any], /) -> None: ...


@dataclass(frozen=True)
class NamedCheck:
    """A check function paired with the name a report shows for it."""

    name: str
    fn: CheckFn

    def __call__(self, registry: FactRegistry, output: Mapping[str, Any], /) -> None:
        self.fn(registry, output)

    def __repr__(self) -> str:
        return f"NamedCheck({self.name})"


def as_named(check: Check | CheckFn) -> NamedCheck:
    """Coerce a builder result or a bare callable into a :class:`NamedCheck`."""
    if isinstance(check, NamedCheck):
        return check
    name = getattr(check, "name", None)
    if not isinstance(name, str):
        name = getattr(check, "__name__", None)
    if not isinstance(name, str):
        name = repr(check)
    return NamedCheck(name, check)


@dataclass(frozen=True)
class Report:
    """Outcome of running every check of a boundary over one output."""

    violations: tuple[UnsupportedClaim, ...]
    passed: tuple[str, ...]
    failed: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "passed": list(self.passed),
            "failed": list(self.failed),
            "violations": [v.to_dict() for v in self.violations],
        }

    def to_text(self) -> str:
        """One line per violation; empty when the report is ok. Usable as a reask message."""
        return "\n".join(str(v) for v in self.violations)


class Boundary:
    """Handle yielded by :func:`boundary`; collects and validates model output."""

    def __init__(
        self,
        registry: FactRegistry,
        permitted: Sequence[str],
        checks: Sequence[Check | CheckFn],
        *,
        require_output: bool = True,
        mode: Mode = "first",
    ) -> None:
        if mode not in ("first", "all"):
            raise ValueError(f"mode must be 'first' or 'all', got {mode!r}")
        self.registry = registry._restricted(permitted)
        self.permitted = tuple(permitted)
        self.checks: tuple[NamedCheck, ...] = tuple(as_named(c) for c in checks)
        self.require_output = require_output
        self.mode: Mode = mode
        self.output: Mapping[str, Any] | None = None
        self._submitted = False

    def facts(self) -> dict[str, Any]:
        """The permitted facts' values, for handing to a prompt builder."""
        return {name: self.registry.value(name) for name in self.permitted}

    def _accept(self, output: Mapping[str, Any]) -> None:
        if not isinstance(output, Mapping):
            raise TypeError(
                "boundary output must be a mapping of declared fields; "
                "free prose is out of scope for v0 (see README, claim extraction)"
            )
        self.output = output
        self._submitted = True

    def submit(self, output: Mapping[str, Any], *, mode: Mode | None = None) -> Mapping[str, Any]:
        """Hand the model's structured output to the boundary and validate it.

        ``mode="first"`` raises the first :class:`UnsupportedClaim` a check produces.
        ``mode="all"`` runs every check and raises :class:`ClaimsUnsupported` carrying
        the full :class:`Report`. Defaults to the boundary's own ``mode``.
        """
        resolved_mode = self.mode if mode is None else mode
        if resolved_mode not in ("first", "all"):
            raise ValueError(f"mode must be 'first' or 'all', got {resolved_mode!r}")
        self._accept(output)
        if resolved_mode == "all":
            report = self.report(output)
            if not report.ok:
                raise ClaimsUnsupported(report)
            return output
        self.validate()
        return output

    def report(self, output: Mapping[str, Any]) -> Report:
        """Run every check and collect the violations instead of stopping at the first.

        Calling ``report`` counts as submitting: the block was checked, and what the
        caller does with a failing report is their decision. Anything a check raises
        other than :class:`UnsupportedClaim` (``ValueError``, ``KeyError`` from a
        misconfigured check) propagates, because a developer mistake is not a model
        mistake and must not be reported as one.
        """
        self._accept(output)
        violations: list[UnsupportedClaim] = []
        passed: list[str] = []
        failed: list[str] = []
        for check in self.checks:
            try:
                check(self.registry, output)
            except UnsupportedClaim as exc:
                violations.append(exc)
                failed.append(check.name)
            else:
                passed.append(check.name)
        return Report(tuple(violations), tuple(passed), tuple(failed))

    def validate(self) -> None:
        if self.output is None:
            raise FactBoundaryError("nothing submitted to the boundary")
        for check in self.checks:
            check(self.registry, self.output)

    def _finish(self) -> None:
        if self.require_output and not self._submitted:
            raise FactBoundaryError(
                "boundary exited without submit(); model output left unchecked. "
                "Pass require_output=False for a block that produces no claims."
            )


class BoundaryDeclaration:
    """What :func:`boundary` returns: a reusable ``with`` target and a decorator."""

    def __init__(
        self,
        registry: FactRegistry,
        facts: Sequence[str],
        checks: Sequence[Check | CheckFn],
        *,
        require_output: bool = True,
        mode: Mode = "first",
    ) -> None:
        if mode not in ("first", "all"):
            raise ValueError(f"mode must be 'first' or 'all', got {mode!r}")
        self.registry = registry
        self.facts = tuple(facts)
        self.checks: tuple[NamedCheck, ...] = tuple(as_named(c) for c in checks)
        self.require_output = require_output
        self.mode: Mode = mode
        # One Boundary per entry, stacked, so a declaration is reusable and reentrant.
        self._active: list[Boundary] = []

    def _open(self) -> Boundary:
        for name in self.facts:
            if name not in self.registry:
                raise UnregisteredFact(name, self.registry.names())
        return Boundary(
            self.registry,
            self.facts,
            self.checks,
            require_output=self.require_output,
            mode=self.mode,
        )

    def __enter__(self) -> Boundary:
        b = self._open()
        self._active.append(b)
        return b

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        b = self._active.pop()
        if exc_type is None:
            b._finish()

    def __call__(self, fn: Callable[P, Mapping[str, Any]]) -> Callable[P, Mapping[str, Any]]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Mapping[str, Any]:
            with self as b:
                return b.submit(fn(*args, **kwargs))

        return wrapper


def boundary(
    registry: FactRegistry,
    *,
    facts: Sequence[str] = (),
    checks: Sequence[Check | CheckFn] = (),
    require_output: bool = True,
    mode: Mode = "first",
) -> BoundaryDeclaration:
    """Declare a fact boundary.

    As a context manager::

        with boundary(reg, facts=["perf"], checks=[superlative(...)]) as b:
            b.submit(model_output)

    As a decorator, wrapping a function that returns the model's structured output::

        @boundary(reg, facts=["perf"], checks=[superlative(...)])
        def summarize() -> dict: ...

    ``mode`` is the default for ``submit``: ``"first"`` raises the first
    :class:`UnsupportedClaim`; ``"all"`` runs every check and raises
    :class:`ClaimsUnsupported` with the whole :class:`Report`. ``b.report(output)``
    returns that report without raising.
    """
    return BoundaryDeclaration(registry, facts, checks, require_output=require_output, mode=mode)

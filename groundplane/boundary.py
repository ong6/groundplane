"""Boundary declaration: which registered facts a block of model output may use.

A boundary is both a context manager and a decorator. Entering it asserts that every
permitted fact actually exists; leaving it runs the attached checks against whatever
structured output the block produced. Output that never gets submitted is itself a
failure, because an unchecked block is an unguarded one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import wraps
from typing import Any

from .errors import FactBoundaryError, UnregisteredFact
from .registry import FactRegistry

__all__ = ["Boundary", "boundary"]

Check = Callable[[FactRegistry, Mapping[str, Any]], None]


class Boundary:
    """Handle yielded by :func:`boundary`; collects and validates model output."""

    def __init__(
        self,
        registry: FactRegistry,
        permitted: Sequence[str],
        checks: Sequence[Check],
        *,
        require_output: bool = True,
    ) -> None:
        self.registry = registry
        self.permitted = tuple(permitted)
        self.checks = tuple(checks)
        self.require_output = require_output
        self.output: Mapping[str, Any] | None = None
        self._submitted = False

    def facts(self) -> dict[str, Any]:
        """The permitted facts' values, for handing to a prompt builder."""
        return {name: self.registry.value(name) for name in self.permitted}

    def submit(self, output: Mapping[str, Any]) -> Mapping[str, Any]:
        """Hand the model's structured output to the boundary and validate it."""
        if not isinstance(output, Mapping):
            raise TypeError(
                "boundary output must be a mapping of declared fields; "
                "free prose is out of scope for v0 (see README, claim extraction)"
            )
        self.output = output
        self._submitted = True
        self.validate()
        return output

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


@contextmanager
def _boundary_cm(
    registry: FactRegistry,
    facts: Sequence[str],
    checks: Sequence[Check],
    require_output: bool,
) -> Iterator[Boundary]:
    for name in facts:
        if name not in registry:
            raise UnregisteredFact(name, registry.names())
    b = Boundary(registry, facts, checks, require_output=require_output)
    yield b
    b._finish()


def boundary(
    registry: FactRegistry,
    *,
    facts: Sequence[str] = (),
    checks: Sequence[Check] = (),
    require_output: bool = True,
) -> Any:
    """Declare a fact boundary.

    As a context manager::

        with boundary(reg, facts=["perf"], checks=[superlative(...)]) as b:
            b.submit(model_output)

    As a decorator, wrapping a function that returns the model's structured output::

        @boundary(reg, facts=["perf"], checks=[superlative(...)])
        def summarize() -> dict: ...
    """
    cm = _boundary_cm(registry, facts, checks, require_output)

    class _Boundary:
        def __enter__(self) -> Boundary:
            return cm.__enter__()

        def __exit__(self, *exc: Any) -> bool | None:
            return cm.__exit__(*exc)

        def __call__(
            self, fn: Callable[..., Mapping[str, Any]]
        ) -> Callable[..., Mapping[str, Any]]:
            @wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
                with _boundary_cm(registry, facts, checks, require_output) as b:
                    return b.submit(fn(*args, **kwargs))

            return wrapper

    return _Boundary()

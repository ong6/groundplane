"""The fact registry: code-computed values with provenance.

Anything a checker is allowed to treat as ground truth must live here first. A fact
is (name, value, provenance) where provenance names the tool call that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from .errors import UnregisteredFact

__all__ = ["Fact", "Provenance", "FactRegistry", "Ranking"]


@dataclass(frozen=True)
class Provenance:
    """Where a fact came from."""

    tool: str
    args: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self.args.items()))
        return f"{self.tool}({rendered})"


@dataclass(frozen=True)
class Ranking:
    """A deterministic ordering computed in code.

    ``items`` is ordered best-first. ``key`` names the metric it was ordered by.
    ``ties`` lists every item sharing the winning score, so a checker can tell a
    genuine tie from a model picking its favourite.
    """

    key: str
    items: tuple[tuple[str, float], ...]
    higher_is_better: bool = True

    @property
    def winner(self) -> str:
        return self.items[0][0]

    @property
    def winning_score(self) -> float:
        return self.items[0][1]

    @property
    def ties(self) -> tuple[str, ...]:
        best = self.winning_score
        return tuple(name for name, score in self.items if score == best)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.items)

    def score_of(self, name: str) -> float | None:
        for item, score in self.items:
            if item == name:
                return score
        return None

    def rank_of(self, name: str) -> int | None:
        for i, (item, _) in enumerate(self.items):
            if item == name:
                return i + 1
        return None


@dataclass(frozen=True)
class Fact:
    """A typed value the code produced, with the call that produced it."""

    name: str
    value: Any
    provenance: Provenance

    @property
    def type_name(self) -> str:
        return type(self.value).__name__


class FactRegistry:
    """Holds the facts a block of model output is allowed to lean on.

    Facts are write-once: re-registering a name raises, because a silently
    overwritten fact is exactly the failure this library exists to catch.
    """

    def __init__(self) -> None:
        self._facts: dict[str, Fact] = {}

    # -- writing -------------------------------------------------------------

    def record(
        self,
        name: str,
        value: Any,
        *,
        tool: str,
        args: Mapping[str, Any] | None = None,
    ) -> Fact:
        """Register ``value`` under ``name`` as produced by ``tool``."""
        if name in self._facts:
            raise ValueError(
                f"fact {name!r} is already registered "
                f"(from {self._facts[name].provenance}); facts are write-once"
            )
        fact = Fact(name=name, value=value, provenance=Provenance(tool=tool, args=dict(args or {})))
        self._facts[name] = fact
        return fact

    def record_ranking(
        self,
        name: str,
        scores: Mapping[str, float] | Sequence[tuple[str, float]],
        *,
        key: str,
        tool: str,
        args: Mapping[str, Any] | None = None,
        higher_is_better: bool = True,
    ) -> Fact:
        """Compute the argmax ordering in code and register it as a fact.

        This is the deterministic half of the superlative guard: the ordering is
        never taken from the model, only ever computed here.
        """
        pairs = list(scores.items()) if isinstance(scores, Mapping) else list(scores)
        if not pairs:
            raise ValueError(f"cannot rank an empty score set for fact {name!r}")
        ordered = tuple(sorted(pairs, key=lambda kv: kv[1], reverse=higher_is_better))
        ranking = Ranking(key=key, items=ordered, higher_is_better=higher_is_better)
        return self.record(name, ranking, tool=tool, args=args)

    def tool(self, name: str, *, fact: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorate a tool function so its return value is registered on call."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                value = fn(*args, **kwargs)
                self.record(fact, value, tool=name, args=dict(kwargs))
                return value

            wrapper.__name__ = getattr(fn, "__name__", name)
            wrapper.__doc__ = fn.__doc__
            return wrapper

        return decorator

    # -- reading -------------------------------------------------------------

    def get(self, name: str) -> Fact:
        try:
            return self._facts[name]
        except KeyError:
            raise UnregisteredFact(name, list(self._facts)) from None

    def value(self, name: str) -> Any:
        return self.get(name).value

    def ranking(self, name: str) -> Ranking:
        fact = self.get(name)
        if not isinstance(fact.value, Ranking):
            raise TypeError(
                f"fact {name!r} is a {fact.type_name}, not a Ranking; "
                "use record_ranking() so the ordering is computed in code"
            )
        return fact.value

    def names(self) -> list[str]:
        return list(self._facts)

    def __contains__(self, name: object) -> bool:
        return name in self._facts

    def __len__(self) -> int:
        return len(self._facts)

    def __iter__(self) -> Iterator[Fact]:
        return iter(self._facts.values())

"""The fact registry: code-computed values with provenance.

Anything a checker is allowed to treat as ground truth must live here first. A fact
is (name, value, provenance) where provenance names the tool call that produced it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import UnregisteredFact

__all__ = ["Fact", "Provenance", "FactRegistry", "Ranking", "Table", "Domain"]


def _join(parts: Iterable[str]) -> str:
    """``"a, b or c"`` — the way an error message wants a list of alternatives."""
    items = list(dict.fromkeys(parts))
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} or {items[-1]}"


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

    ``items`` is ordered best-first as ``(name, score)`` pairs. ``key`` names the
    metric it was ordered by. ``ties`` lists every item sharing the winning score, so
    a checker can tell a genuine tie from a model picking its favourite.

    A name is whatever the code used to identify the thing — a string, an integer id,
    a tuple. It is never coerced: an ``int`` row key stays an ``int``, so a check
    comparing it against the model's ``101`` is comparing like with like rather than
    ``"101"`` with ``101``.
    """

    key: str
    items: tuple[tuple[Any, float], ...]
    higher_is_better: bool = True

    @property
    def winner(self) -> Any:
        return self.items[0][0]

    @property
    def winning_score(self) -> float:
        return self.items[0][1]

    @property
    def ties(self) -> tuple[Any, ...]:
        best = self.winning_score
        return tuple(name for name, score in self.items if score == best)

    @property
    def names(self) -> tuple[Any, ...]:
        return tuple(name for name, _ in self.items)

    @property
    def tie_blocks(self) -> tuple[tuple[Any, ...], ...]:
        """The ordering split into blocks of equal score, best block first.

        A block of length one is an unambiguous position; a longer block is a set
        of positions the data does not distinguish, so any cut inside it is not a
        fact. Each block is a distinct object, so callers may compare by identity.
        """
        blocks: list[tuple[Any, ...]] = []
        current: list[Any] = []
        last: float | None = None
        for name, score in self.items:
            if last is not None and score != last:
                blocks.append(tuple(current))
                current = []
            current.append(name)
            last = score
        if current:
            blocks.append(tuple(current))
        return tuple(blocks)

    def score_of(self, name: Any) -> float | None:
        for item, score in self.items:
            if item == name:
                return score
        return None

    def rank_of(self, name: Any) -> int | None:
        for i, (item, _) in enumerate(self.items):
            if item == name:
                return i + 1
        return None


@dataclass(frozen=True)
class Table:
    """A set of rows the code retrieved, keyed by one column.

    ``rows`` is ordered as the tool returned them and ``key`` names the column that
    identifies a row. ``exhaustive`` records whether these are *all* the rows the
    query matched: a paginated or capped result set is not a valid base for an
    aggregate, and the flag is what lets a checker say so instead of adding up a
    prefix.
    """

    key: str
    rows: tuple[Mapping[str, Any], ...]
    columns: tuple[str, ...]
    exhaustive: bool = True

    @property
    def keys(self) -> tuple[Any, ...]:
        return tuple(row[self.key] for row in self.rows)

    def row(self, key: Any) -> Mapping[str, Any] | None:
        """The row identified by ``key``, or ``None`` if the table has no such row."""
        for row in self.rows:
            if row[self.key] == key:
                return row
        return None

    def column(self, name: str) -> tuple[Any, ...]:
        """Every value present in ``name``, skipping rows that do not carry it."""
        if name not in self.columns:
            raise KeyError(f"column {name!r} not in table; columns: {list(self.columns)}")
        return tuple(row[name] for row in self.rows if name in row)

    def to_ranking(self, column: str, *, higher_is_better: bool = True) -> Ranking:
        """Order the rows by ``column``, in code, as a :class:`Ranking`."""
        if column not in self.columns:
            raise KeyError(f"column {column!r} not in table; columns: {list(self.columns)}")
        pairs = [(row[self.key], float(row[column])) for row in self.rows if column in row]
        if not pairs:
            raise ValueError(f"no row carries column {column!r}; nothing to rank")
        ordered = tuple(sorted(pairs, key=lambda kv: kv[1], reverse=higher_is_better))
        return Ranking(key=column, items=ordered, higher_is_better=higher_is_better)


@dataclass(frozen=True)
class Domain:
    """A closed set of names the code produced.

    Everything the model is allowed to name lives in ``members``; ``label`` is the
    noun the error message should use ("campaign", "region") so a reask prompt reads
    like the caller's domain rather than the library's.
    """

    members: frozenset[str]
    label: str = "entity"

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.members))

    def __contains__(self, name: object) -> bool:
        return name in self.members

    def __len__(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class Fact:
    """A typed value the code produced, with the call that produced it."""

    name: str
    value: Any
    provenance: Provenance

    @property
    def type_name(self) -> str:
        return type(self.value).__name__


_RECORDER: Mapping[type, str] = {
    Ranking: "record_ranking()",
    Table: "record_table()",
    Domain: "record_domain()",
}


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
        scores: Mapping[Any, float] | Sequence[tuple[Any, float]],
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

    def record_table(
        self,
        name: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        key: str,
        tool: str,
        args: Mapping[str, Any] | None = None,
        columns: Sequence[str] | None = None,
        exhaustive: bool = True,
    ) -> Fact:
        """Register the rows a tool returned as a :class:`Table` fact.

        ``key`` must identify a row: every row carries it and no two rows share a
        value. ``columns`` defaults to the union of the row keys, in first-seen
        order. Pass ``exhaustive=False`` when the result set was paginated, capped,
        or otherwise truncated, so aggregate checks refuse to total a prefix.
        """
        materialised = tuple(dict(row) for row in rows)
        if not materialised:
            raise ValueError(f"cannot record an empty table for fact {name!r}")

        seen_columns: list[str] = []
        for row in materialised:
            for column in row:
                if column not in seen_columns:
                    seen_columns.append(column)
        resolved = tuple(columns) if columns is not None else tuple(seen_columns)
        if key not in resolved:
            raise ValueError(
                f"key column {key!r} is not among the table columns {list(resolved)}"
            )

        seen_keys: set[Any] = set()
        for i, row in enumerate(materialised):
            if key not in row:
                raise ValueError(f"row {i} of fact {name!r} has no key column {key!r}")
            if row[key] in seen_keys:
                raise ValueError(
                    f"key {row[key]!r} appears more than once in fact {name!r}; "
                    f"{key!r} does not identify a row"
                )
            seen_keys.add(row[key])

        table = Table(key=key, rows=materialised, columns=resolved, exhaustive=exhaustive)
        return self.record(name, table, tool=tool, args=args)

    def record_domain(
        self,
        name: str,
        members: Iterable[str],
        *,
        tool: str,
        args: Mapping[str, Any] | None = None,
        label: str = "entity",
    ) -> Fact:
        """Register a closed set of names as a :class:`Domain` fact."""
        collected = tuple(members)
        if not collected:
            raise ValueError(f"cannot record an empty domain for fact {name!r}")
        for member in collected:
            if not isinstance(member, str):
                raise TypeError(
                    f"domain {name!r} member {member!r} is a {type(member).__name__}, "
                    "not a name; a domain holds strings"
                )
        domain = Domain(members=frozenset(collected), label=label)
        return self.record(name, domain, tool=tool, args=args)

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

    def as_type(self, name: str, *types: type) -> Any:
        """The fact's value, required to be one of ``types``.

        Every typed accessor and every check routes its type demand through here, so
        the same user mistake reads the same way wherever it surfaces.
        """
        fact = self.get(name)
        if isinstance(fact.value, types):
            return fact.value
        wanted = _join(t.__name__ for t in types)
        recorders = _join(_RECORDER.get(t, "record()") for t in types)
        raise TypeError(
            f"fact {name!r} is a {fact.type_name}, not a {wanted}; "
            f"record it with {recorders} so it is computed in code"
        )

    def ranking(self, name: str) -> Ranking:
        return self.as_type(name, Ranking)

    def table(self, name: str) -> Table:
        return self.as_type(name, Table)

    def domain(self, name: str) -> Domain:
        return self.as_type(name, Domain)

    def names(self) -> list[str]:
        return list(self._facts)

    def __contains__(self, name: object) -> bool:
        return name in self._facts

    def __len__(self) -> int:
        return len(self._facts)

    def __iter__(self) -> Iterator[Fact]:
        return iter(self._facts.values())

"""The fact registry: code-computed values with provenance.

Anything a checker is allowed to treat as ground truth must live here first. A fact
is (name, value, provenance) where provenance names the tool call that produced it.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import UnregisteredFact

__all__ = [
    "Fact",
    "Provenance",
    "FactRegistry",
    "Ranking",
    "Table",
    "Domain",
    "SparseColumnWarning",
]

OnMissing = Literal["raise", "skip"]


class SparseColumnWarning(UserWarning):
    """A ranking was derived from only the rows that carry the ranked column.

    Emitted by :meth:`Table.to_ranking` under ``on_missing="skip"``. The rows it
    names were never scored, so the argmax is an argmax over a subset and any
    check built on it is only as good as the caller's reason for skipping them.
    """


def require_number(value: Any, *, column: str, key: Any, fact: str | None = None) -> float:
    """``value`` as a float, or a ``ValueError`` that says which cell was bad.

    Every place that turns a recorded cell into arithmetic goes through here, so
    a ``None`` from a left join, a ``"n/a"`` string, or a ``NaN`` from a broken
    division fails naming the fact, column and row instead of as a bare
    ``TypeError`` three frames deep. Booleans are refused too: ``True`` is a flag
    that Python happens to spell as ``1``, not a score.

    Non-finite values are refused rather than coerced because neither ``sorted``
    nor ``==`` has a defensible answer for them: NaN sorts wherever it lands and
    equals nothing, so a ranking or a tie built on one is not a fact.
    """
    where = f"column {column!r} row {key!r}"
    if fact is not None:
        where = f"fact {fact!r} {where}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: value {value!r} is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where}: value {value!r} is not finite")
    return number


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

    def to_ranking(
        self,
        column: str,
        *,
        higher_is_better: bool = True,
        on_missing: OnMissing = "raise",
    ) -> Ranking:
        """Order the rows by ``column``, in code, as a :class:`Ranking`.

        A row without ``column`` cannot be placed, and an argmax that quietly
        leaves it out is a confident answer over a subset — exactly the shape of
        claim this library exists to refuse. So the default, ``on_missing="raise"``,
        refuses to rank at all and names the rows it would have dropped. Pass
        ``on_missing="skip"`` to rank the rows that do carry the column; the
        dropped keys are then reported through a :class:`SparseColumnWarning`
        rather than silently.

        Every ranked cell must be a finite number; ``None``, strings, booleans and
        NaN raise ``ValueError`` naming the row (see :func:`require_number`).
        """
        if column not in self.columns:
            raise KeyError(f"column {column!r} not in table; columns: {list(self.columns)}")
        if on_missing not in ("raise", "skip"):
            raise ValueError(f"on_missing must be 'raise' or 'skip', got {on_missing!r}")
        missing = [row[self.key] for row in self.rows if column not in row]
        if missing:
            if on_missing == "raise":
                raise ValueError(
                    f"column {column!r} is missing on rows {missing!r}; a ranking over the "
                    "remaining rows would be an argmax over a subset "
                    "(pass on_missing='skip' to rank only the rows that carry it)"
                )
            warnings.warn(
                SparseColumnWarning(
                    f"ranking on column {column!r} skipped rows {missing!r} that do not "
                    "carry it; the argmax covers a subset of the table"
                ),
                stacklevel=2,
            )
        pairs = [
            (row[self.key], require_number(row[column], column=column, key=row[self.key]))
            for row in self.rows
            if column in row
        ]
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

        Each entity may be scored once — a repeated name would make ``rank_of`` and
        ``score_of`` answer for whichever copy sorted first — and every score must
        be a finite number, because NaN sorts arbitrarily and ties with nothing, so
        an ordering containing one is not an ordering.
        """
        pairs = list(scores.items()) if isinstance(scores, Mapping) else list(scores)
        if not pairs:
            raise ValueError(f"cannot rank an empty score set for fact {name!r}")

        seen: set[Any] = set()
        duplicates: list[Any] = []
        for entity, score in pairs:
            if entity in seen and entity not in duplicates:
                duplicates.append(entity)
            seen.add(entity)
            require_number(score, column=key, key=entity, fact=name)
        if duplicates:
            raise ValueError(
                f"names {duplicates!r} appear more than once in fact {name!r}; "
                "a ranking scores each entity once"
            )

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
            raise ValueError(f"key column {key!r} is not among the table columns {list(resolved)}")

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

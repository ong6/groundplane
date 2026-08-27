"""Entity check: every name the model uses must be one the code recorded.

The most common failure in the taxonomy is not a wrong number, it is a wrong
noun — a name blended out of the prompt (``"northwest"`` from ``north`` and
``west``), an id carried over from an earlier tool call, or an entity invented
whole. Pydantic cannot catch it because the legal set of names lives in a tool
result, not in the schema.

The domain is whatever a fact already holds: a :class:`~groundplane.registry.Domain`,
the names of a :class:`~groundplane.registry.Ranking`, or the key column of a
:class:`~groundplane.registry.Table`. One recorded fact therefore serves this check
and the ranking or row checks at the same time.
"""

from __future__ import annotations

import difflib
from typing import Any, Mapping, Sequence

from .boundary import Check
from .checks import _require_field
from .errors import UnsupportedClaim
from .registry import Domain, FactRegistry, Ranking, Table

__all__ = ["entities_recorded", "check_entities_recorded"]

_MAX_RENDERED = 20


def _resolve_domain(registry: FactRegistry, fact: str) -> tuple[frozenset[str], str]:
    """Return the closed name set a fact defines, plus a label for the error text."""
    value: Any = registry.get(fact).value
    if isinstance(value, Domain):
        return value.members, value.label
    if isinstance(value, Ranking):
        return frozenset(value.names), "entity"
    if isinstance(value, Table):
        return frozenset(str(key) for key in value.keys), "entity"
    raise TypeError(
        f"fact {fact!r} is a {type(value).__name__}, not a Domain, Ranking or Table; "
        "record it with record_domain(), record_ranking() or record_table() so the "
        "permitted names are computed in code"
    )


def _render(members: Sequence[str]) -> list[str]:
    """The domain as it should appear in a reask prompt: readable, never unbounded."""
    if len(members) <= _MAX_RENDERED:
        return list(members)
    extra = len(members) - _MAX_RENDERED
    return [*members[:_MAX_RENDERED], f"… (+{extra} more)"]


def _traverse(output: Mapping[str, Any], path: Sequence[str], fact: str) -> Mapping[str, Any]:
    """Walk ``path`` into a nested output object, failing as a claim, not a KeyError."""
    current: Any = output
    for i, step in enumerate(path):
        if not isinstance(current, Mapping) or step not in current:
            raise UnsupportedClaim(
                ".".join(path[: i + 1]),
                None,
                "a nested object",
                fact=fact,
                reason=(
                    f"the model output has no object at path {list(path[: i + 1])}, "
                    "so the entity names cannot be located"
                ),
            )
        current = current[step]
    if not isinstance(current, Mapping):
        raise UnsupportedClaim(
            ".".join(path) if path else "<root>",
            current,
            "a nested object",
            fact=fact,
            reason=f"value at path {list(path)} is a {type(current).__name__}, not an object",
        )
    return current


def _reject(
    field: str,
    claimed: object,
    *,
    fact: str,
    provenance: str,
    members: Sequence[str],
    reason: str,
) -> None:
    raise UnsupportedClaim(
        field,
        claimed,
        _render(members),
        fact=fact,
        provenance=provenance,
        reason=reason,
    )


def _check_name(
    claimed: object,
    *,
    field: str,
    fact: str,
    provenance: str,
    members: Sequence[str],
    lookup: Mapping[str, str],
    label: str,
    case_sensitive: bool,
) -> None:
    """Fail unless ``claimed`` is a recorded name, spelling rules applied."""
    if not isinstance(claimed, str):
        _reject(
            field,
            claimed,
            fact=fact,
            provenance=provenance,
            members=members,
            reason=(
                f"claimed {label} is a {type(claimed).__name__}, not a name; "
                f"only recorded {label} names may be used"
            ),
        )
        return
    probe = claimed if case_sensitive else claimed.casefold()
    if probe in lookup:
        return
    near = difflib.get_close_matches(claimed, list(members), n=1, cutoff=0.6)
    hint = f"; did you mean {near[0]!r}?" if near else ""
    _reject(
        field,
        claimed,
        fact=fact,
        provenance=provenance,
        members=members,
        reason=f"{claimed!r} is not a recorded {label}{hint}",
    )


def check_entities_recorded(
    registry: FactRegistry,
    output: Mapping[str, Any],
    *,
    fields: Sequence[str],
    fact: str,
    path: Sequence[str] = (),
    allow_empty: bool = False,
    case_sensitive: bool = True,
) -> None:
    """Verify every entity name in ``fields`` was recorded by the code.

    Each field may hold a single name or a list of names; a list is checked
    member-wise and the first offending member raises. Raises
    :class:`UnsupportedClaim` when a name was never recorded, when a member is not
    a name at all, or — unless ``allow_empty`` — when a list is empty, since an
    empty list asserts that nothing qualified.
    """
    members_set, label = _resolve_domain(registry, fact)
    members = tuple(sorted(members_set))
    provenance = str(registry.get(fact).provenance)
    lookup: dict[str, str] = (
        {name: name for name in members} if case_sensitive else {n.casefold(): n for n in members}
    )
    scope = _traverse(output, path, fact) if path else output

    for name_field in fields:
        claimed = _require_field(scope, name_field)
        if isinstance(claimed, (list, tuple)):
            if not claimed and not allow_empty:
                _reject(
                    name_field,
                    claimed,
                    fact=fact,
                    provenance=provenance,
                    members=members,
                    reason=(
                        f"an empty {label} list is itself a claim — that none qualified — and "
                        "nothing recorded supports it (pass allow_empty=True if it is legal)"
                    ),
                )
            for member in claimed:
                _check_name(
                    member,
                    field=name_field,
                    fact=fact,
                    provenance=provenance,
                    members=members,
                    lookup=lookup,
                    label=label,
                    case_sensitive=case_sensitive,
                )
        else:
            _check_name(
                claimed,
                field=name_field,
                fact=fact,
                provenance=provenance,
                members=members,
                lookup=lookup,
                label=label,
                case_sensitive=case_sensitive,
            )


def entities_recorded(
    *,
    fields: Sequence[str],
    fact: str,
    path: Sequence[str] = (),
    allow_empty: bool = False,
    case_sensitive: bool = True,
) -> Check:
    """Build a boundary check from :func:`check_entities_recorded`."""

    def check(registry: FactRegistry, output: Mapping[str, Any]) -> None:
        check_entities_recorded(
            registry,
            output,
            fields=fields,
            fact=fact,
            path=path,
            allow_empty=allow_empty,
            case_sensitive=case_sensitive,
        )

    return check

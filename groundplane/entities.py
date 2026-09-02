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
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

from ._internal import require_field
from .boundary import NamedCheck
from .errors import UnsupportedClaim
from .registry import Domain, FactRegistry, Ranking, Table

__all__ = ["entities_recorded", "check_entities_recorded"]

_MAX_RENDERED = 20


def _resolve_domain(registry: FactRegistry, fact: str) -> tuple[frozenset[Any], str]:
    """Return the closed name set a fact defines, plus a label for the error text.

    Names keep the type the code gave them. A ``Ranking`` or ``Table`` keyed by
    integer ids yields ``int`` members, so a model answering ``101`` is compared
    with ``101`` and not with ``"101"`` — the same rule the ordering checks follow.
    """
    value: Any = registry.as_type(fact, Domain, Ranking, Table)
    if isinstance(value, Domain):
        return value.members, value.label
    names = value.names if isinstance(value, Ranking) else value.keys
    return frozenset(names), "entity"


def _sorted_members(members: frozenset[Any]) -> tuple[Any, ...]:
    """A stable order for the error text; mixed types fall back to their repr."""
    try:
        return tuple(sorted(members))
    except TypeError:
        return tuple(sorted(members, key=repr))


def _render(members: Sequence[Any]) -> list[Any]:
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
    members: Sequence[Any],
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
    members: Sequence[Any],
    lookup: Mapping[str, str],
    label: str,
    case_sensitive: bool,
) -> None:
    """Fail unless ``claimed`` is a recorded name, spelling rules applied.

    Spelling rules (case folding) are a property of strings, so they apply only
    to string names. Any other name must match a recorded member of the same
    type exactly: ``101`` matches the recorded id ``101``, and neither ``"101"``
    nor ``101.0`` nor ``True`` (which Python would otherwise count as ``1``) does.
    """
    if not isinstance(claimed, str):
        if not isinstance(claimed, Hashable):
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
        if any(type(member) is type(claimed) and member == claimed for member in members):
            return
        _reject(
            field,
            claimed,
            fact=fact,
            provenance=provenance,
            members=members,
            reason=(
                f"{claimed!r} is a {type(claimed).__name__} and not a name recorded as a "
                f"{label}; only recorded {label} names may be used"
            ),
        )
        return
    probe = claimed if case_sensitive else claimed.casefold()
    if probe in lookup:
        return
    spelled = [member for member in members if isinstance(member, str)]
    near = difflib.get_close_matches(claimed, spelled, n=1, cutoff=0.6)
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
    members = _sorted_members(members_set)
    provenance = str(registry.get(fact).provenance)
    spelled = [name for name in members if isinstance(name, str)]
    lookup: dict[str, str] = (
        {name: name for name in spelled}
        if case_sensitive
        else {name.casefold(): name for name in spelled}
    )
    scope = _traverse(output, path, fact) if path else output

    for name_field in fields:
        claimed = require_field(scope, name_field)
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
) -> NamedCheck:
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

    return NamedCheck(f"entities_recorded(fact={fact!r}, fields={list(fields)!r})", check)

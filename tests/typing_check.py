"""Static typing assertions, checked by mypy (listed in ``[tool.mypy] files``).

Not a pytest module on purpose: it never runs, it only has to type-check. The point
is that ``boundary(...)`` is no longer ``Any``: ``with boundary(...) as b`` gives a
``Boundary``, the decorator keeps the wrapped signature, and the builders return
``NamedCheck``, which satisfies the ``Check`` protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from typing_extensions import assert_type

from groundplane import (
    Boundary,
    BoundaryDeclaration,
    Check,
    FactRegistry,
    NamedCheck,
    Report,
    boundary,
    superlative,
)


def _plain(registry: FactRegistry, output: Mapping[str, Any]) -> None: ...


def declaration(reg: FactRegistry) -> None:
    named = superlative(fact="campaign_ctr")
    assert_type(named, NamedCheck)
    protocol_typed: Check = named
    guard = boundary(reg, facts=["campaign_ctr"], checks=[protocol_typed, _plain], mode="all")
    assert_type(guard, BoundaryDeclaration)

    with guard as b:
        assert_type(b, Boundary)
        assert_type(b.report({"winner": "harbour"}), Report)
        assert_type(b.submit({"winner": "harbour"}), Mapping[str, Any])

    @guard
    def summarize(winner: str, *, verbose: bool = False) -> dict[str, Any]:
        return {"winner": winner}

    summarize("harbour", verbose=True)
    summarize(1)  # type: ignore[arg-type]

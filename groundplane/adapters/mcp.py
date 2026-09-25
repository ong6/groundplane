"""MCP adapter: record what an MCP tool returned, with the call as provenance.

An MCP tool result is a payload plus the call that produced it, which is a fact in
everything but name. This module reads the payload out of a ``CallToolResult``
without importing ``mcp``: it duck-types ``structured_content`` /
``structuredContent`` first, then falls back to the text blocks in ``content``.

Structured content is preferred deliberately. A server that declares an output
schema hands back a value; the text blocks are a rendering of it, and a checker
that reads the rendering is back to parsing prose.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from ..registry import Fact, FactRegistry

__all__ = ["payload_of", "record_result", "call_and_record"]


def _member(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _text_blocks(result: Any) -> list[str]:
    blocks = _member(result, "content")
    if not isinstance(blocks, Sequence):
        return []
    texts = [_member(b, "text") for b in blocks if _member(b, "type") == "text"]
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("MCP text content must carry a string")
    return texts


def _finite_json_number(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"MCP JSON text contains a non-finite number: {token}")
    return value


def payload_of(result: Any, *, parse_json: bool = True) -> Any:
    """The value an MCP tool result carries.

    Structured content wins. Otherwise the text blocks are joined, and — unless
    ``parse_json`` is off — parsed as JSON when they parse, so a server that
    returns a JSON string still yields rows rather than a blob.

    Raises ``ValueError`` when the result carries nothing, and when the result is
    flagged ``isError``: an error result is not a fact.
    """
    if _member(result, "isError", False) or _member(result, "is_error", False):
        raise ValueError(f"MCP tool result is an error result: {_text_blocks(result) or result!r}")

    for attr in ("structured_content", "structuredContent"):
        structured = _member(result, attr)
        if structured is not None:
            return structured

    texts = _text_blocks(result)
    if not texts:
        raise ValueError("MCP tool result carries neither structured content nor text")
    joined = "\n".join(texts)
    if parse_json:
        try:
            return json.loads(
                joined, parse_float=_finite_json_number, parse_constant=_finite_json_number
            )
        except json.JSONDecodeError:
            pass
    return joined


def record_result(
    registry: FactRegistry,
    name: str,
    result: Any,
    *,
    tool: str,
    args: Mapping[str, Any] | None = None,
    parse_json: bool = True,
) -> Fact:
    """Register an MCP tool result as a fact, with the tool call as provenance."""
    return registry.record(name, payload_of(result, parse_json=parse_json), tool=tool, args=args)


async def call_and_record(
    session: Any,
    registry: FactRegistry,
    *,
    fact: str,
    tool: str,
    arguments: Mapping[str, Any] | None = None,
    parse_json: bool = True,
) -> Any:
    """Call an MCP tool through ``session`` and record what it returned.

    ``session`` is anything with the MCP client's ``call_tool(name, arguments)``
    coroutine. The recorded provenance is the tool name and the arguments actually
    sent, so the error message a checker later prints names the call to re-run.
    """
    sent = deepcopy(dict(arguments or {}))
    result = await session.call_tool(tool, deepcopy(sent))
    record_result(registry, fact, result, tool=tool, args=sent, parse_json=parse_json)
    return result

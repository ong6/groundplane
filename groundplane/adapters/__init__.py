"""Adapters that put a fact boundary inside someone else's agent loop.

Each adapter is import-free: nothing here imports the framework it adapts, so the
core stays dependency-free and these modules are testable without installing
anything. The extras (``groundplane[langgraph]``, ``groundplane[mcp]``) exist to
pull in the framework itself, not to make these imports work.
"""

from __future__ import annotations

__all__ = ["langgraph", "mcp"]

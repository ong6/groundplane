"""LangGraph adapter: a fact boundary around a node's state update.

A LangGraph node is a callable that takes the graph state and returns a partial
state update. That update is exactly the structured output a boundary wants, so
the adapter is a wrapper rather than an integration: no ``langgraph`` import, no
subclassing, and the wrapped node is still an ordinary callable the graph can hold.

The reason to want it inside the graph rather than after it is the reask edge. A
raised :class:`~groundplane.errors.UnsupportedClaim` kills the run; an
``on_violation`` handler turns the same failure into a state update, so the graph
can route back to the model with the checker's error as the correction prompt::

    graph.add_node("summarise", guarded_node(
        summarise,
        registry=reg,
        facts=["campaign_ctr"],
        checks=[superlative(fact="campaign_ctr")],
        output_key="summary",
        on_violation=lambda exc, state: {"reask": str(exc)},
    ))
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, cast, overload

from ..boundary import Check, CheckFn, boundary
from ..errors import UnsupportedClaim
from ..registry import FactRegistry

__all__ = ["guarded_node", "record_state"]

State = Mapping[str, Any]
Update = Mapping[str, Any]
Node = Callable[..., Update]
AsyncNode = Callable[..., Awaitable[Update]]
OnViolation = Callable[[UnsupportedClaim, State], Update]


@overload
def guarded_node(
    fn: AsyncNode,
    *,
    registry: FactRegistry,
    facts: Sequence[str] = (),
    checks: Sequence[Check | CheckFn] = (),
    output_key: str | None = None,
    on_violation: OnViolation | None = None,
) -> AsyncNode: ...


@overload
def guarded_node(
    fn: Node,
    *,
    registry: FactRegistry,
    facts: Sequence[str] = (),
    checks: Sequence[Check | CheckFn] = (),
    output_key: str | None = None,
    on_violation: OnViolation | None = None,
) -> Node: ...


def guarded_node(
    fn: Node | AsyncNode,
    *,
    registry: FactRegistry,
    facts: Sequence[str] = (),
    checks: Sequence[Check | CheckFn] = (),
    output_key: str | None = None,
    on_violation: OnViolation | None = None,
) -> Node | AsyncNode:
    """Wrap a node so its state update crosses a fact boundary before the graph sees it.

    ``output_key`` names the field of the update holding the model's structured
    output; omit it to check the whole update. ``on_violation`` receives the
    :class:`UnsupportedClaim` and the incoming state and returns the update to use
    instead — the reask path. Without it the claim propagates and the run fails,
    which is the right default: an unchecked answer must not reach a customer.

    Only :class:`UnsupportedClaim` is routed to ``on_violation``. A misconfigured
    check raises ``ValueError``, ``KeyError`` or ``TypeError`` and those still
    propagate, because a developer bug is not something to reask the model about.
    """

    def validate(update: Update, state: State) -> Update:
        payload = update[output_key] if output_key is not None else update
        try:
            with boundary(registry, facts=facts, checks=checks) as b:
                b.submit(payload)
        except UnsupportedClaim as exc:
            if on_violation is None:
                raise
            return on_violation(exc, state)
        return update

    call_method = getattr(fn, "__call__", None)  # noqa: B004 - inspect async callable objects
    if iscoroutinefunction(fn) or iscoroutinefunction(call_method):

        @wraps(fn)
        async def async_node(state: State, *args: Any, **kwargs: Any) -> Update:
            update = await cast(AsyncNode, fn)(state, *args, **kwargs)
            return validate(update, state)

        return async_node

    @wraps(fn)
    def node(state: State, *args: Any, **kwargs: Any) -> Update:
        return validate(cast(Node, fn)(state, *args, **kwargs), state)

    return node


def record_state(
    registry: FactRegistry,
    state: State,
    *,
    keys: Mapping[str, str],
    tool: str,
    args: Mapping[str, Any] | None = None,
) -> None:
    """Register values a previous node already put in the state as facts.

    ``keys`` maps state key to fact name. Use it when the tool call happened in a
    node you do not own — the values are ground truth either way, and provenance
    records which node produced them. Facts are write-once, so calling this twice
    for the same fact raises rather than quietly re-recording a stale value.
    """
    for state_key, fact in keys.items():
        if state_key not in state:
            raise KeyError(f"state has no key {state_key!r}; keys: {sorted(state)}")
        registry.record(fact, state[state_key], tool=tool, args=args)

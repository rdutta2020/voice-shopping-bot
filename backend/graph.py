"""
LangGraph StateGraph for the voice shopping bot.

All nodes call main.py helper functions directly (no HTTP).

Langfuse tracing: each node opens a child span anchored to the parent
shopping-graph agent span via an explicit TraceContext(trace_id, parent_span_id).
This is reliable even when LangGraph copies the contextvars context between tasks.

Flow:
  user text
      │
      ▼
  intent_router ──── add_to_cart ──► cart_agent ──► END
      │                                               ▲
      ├──── get_offers ──► recommendation_agent ──────┘
      │
      └──── other ──► END
"""

import contextlib
from typing import Any, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

# ── State ──────────────────────────────────────────────────────────────────

class ShoppingState(TypedDict):
    text:          str
    intent:        Optional[str]
    confidence:    Optional[str]
    tool_result:   Optional[str]
    reply:         Optional[str]
    # Langfuse parent-trace identifiers, injected by /process and threaded
    # through every node unchanged so each node can create a correctly-parented
    # child span even if LangGraph runs nodes in a copied OTel context.
    _lf_trace_id:  Optional[str]
    _lf_span_id:   Optional[str]

# ── Shared helper ──────────────────────────────────────────────────────────

def _node_span(name: str, input_val: Any, state: ShoppingState):
    """
    Return a Langfuse span context-manager for a graph node, or nullcontext
    if Langfuse is not configured or trace IDs were not provided.

    Uses TraceContext(trace_id, parent_span_id) for explicit parent linkage
    rather than relying on OTel context propagation through LangGraph.
    """
    from main import langfuse  # deferred to avoid circular import at load time

    lf_trace_id = state.get("_lf_trace_id")
    lf_span_id  = state.get("_lf_span_id")

    if langfuse and lf_trace_id:
        from langfuse.types import TraceContext
        return langfuse.start_as_current_observation(
            trace_context=TraceContext(trace_id=lf_trace_id, parent_span_id=lf_span_id),
            name=name,
            as_type="span",
            input=input_val,
        )

    return contextlib.nullcontext(None)

# ── Node 1: intent_router ──────────────────────────────────────────────────

def intent_router(state: ShoppingState) -> ShoppingState:
    """
    Classify the user utterance into add_to_cart / get_offers / other.
    Opens a child span under the parent shopping-graph agent, then calls
    _run_detect_intent which creates a nested claude-detect-intent generation.
    """
    from main import _run_detect_intent

    with _node_span("intent_router", state["text"], state) as span:
        data = _run_detect_intent(state["text"])
        if span:
            span.update(output=data)

    return {
        **state,
        "intent":     data["intent"],
        "confidence": data["confidence"],
    }

# ── Node 2: cart_agent ────────────────────────────────────────────────────

def cart_agent(state: ShoppingState) -> ShoppingState:
    """
    Extract items from speech then add each to the cart.
    _run_extract_items creates a nested claude-extract-items generation.
    """
    from main import _run_extract_items, handle_tool_call

    with _node_span("cart_agent", state["text"], state) as span:
        items   = _run_extract_items(state["text"])
        results = [handle_tool_call("add_to_cart", item) for item in items]
        reply   = "\n".join(results) if results else "No items found to add."
        if span:
            span.update(output={"items_added": len(items), "reply": reply})

    return {
        **state,
        "tool_result": reply,
        "reply":       reply,
    }

# ── Node 3: recommendation_agent ──────────────────────────────────────────

def recommendation_agent(state: ShoppingState) -> ShoppingState:
    """Fetch today's offers via the get_offers tool."""
    from main import handle_tool_call

    with _node_span("recommendation_agent", {}, state) as span:
        result = handle_tool_call("get_offers", {})
        if span:
            span.update(output=result)

    return {
        **state,
        "tool_result": result,
        "reply":       result,
    }

# ── Conditional edge ───────────────────────────────────────────────────────

def route_intent(state: ShoppingState) -> Literal["cart_agent", "recommendation_agent", "__end__"]:
    intent = state.get("intent", "other")
    if intent == "add_to_cart":
        return "cart_agent"
    elif intent == "get_offers":
        return "recommendation_agent"
    else:
        return "__end__"

# ── Build and compile ──────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(ShoppingState)

    graph.add_node("intent_router",        intent_router)
    graph.add_node("cart_agent",           cart_agent)
    graph.add_node("recommendation_agent", recommendation_agent)

    graph.set_entry_point("intent_router")

    graph.add_conditional_edges(
        "intent_router",
        route_intent,
        {
            "cart_agent":            "cart_agent",
            "recommendation_agent":  "recommendation_agent",
            "__end__":               END,
        },
    )

    graph.add_edge("cart_agent",           END)
    graph.add_edge("recommendation_agent", END)

    return graph.compile()

shopping_graph = build_graph()

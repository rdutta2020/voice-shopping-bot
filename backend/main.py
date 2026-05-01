import contextlib
import json
import re

import anthropic
from fastapi import FastAPI
from pydantic import BaseModel

# Langfuse v4 — reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from env.
# If the keys are absent the server still starts; tracing is simply skipped.
try:
    from langfuse import Langfuse
    langfuse = Langfuse()
except Exception:
    langfuse = None
    print("⚠️  Langfuse not configured — set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST")

# graph.py is imported at the bottom (after handle_tool_call is defined) so
# that graph.py can do `from main import ...` without a circular-import error.

app = FastAPI()
client = anthropic.Anthropic()

MODEL = "claude-sonnet-4-5"

def parse_json_response(text: str) -> dict:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    return json.loads(text.strip())

# ── Request / Response models ──────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str

class IntentResponse(BaseModel):
    intent: str        # "add_to_cart" | "get_offers" | "other"
    confidence: str    # "high" | "medium" | "low"

class CartItem(BaseModel):
    item: str
    quantity: float
    unit: str

class ExtractResponse(BaseModel):
    items: list[CartItem]

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# ── Tool definitions ───────────────────────────────────────────────────────

CART_TOOLS = [
    {
        "name": "add_to_cart",
        "description": "Adds an item to the shopping cart. Call this when user mentions buying, ordering or needing any product.",
        "input_schema": {
            "type": "object",
            "properties": {
                "item":     {"type": "string", "description": "Item name e.g. sugar, oil, rice"},
                "quantity": {"type": "number", "description": "How many units"},
                "unit":     {"type": "string", "description": "kg, litre, dozen, piece, packet etc"}
            },
            "required": ["item", "quantity", "unit"]
        }
    },
    {
        "name": "get_offers",
        "description": "Gets current offers and deals. Call this when user asks about offers, discounts or recommendations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Optional product category"}
            }
        }
    },
    {
        "name": "view_cart",
        "description": "Shows all items in the cart. Call this when user wants to see or review their order.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]

# ── In-memory cart ─────────────────────────────────────────────────────────
cart = []

def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """FastAPI executes the tool — Claude never does this directly."""
    if tool_name == "add_to_cart":
        item     = tool_input["item"]
        quantity = tool_input["quantity"]
        unit     = tool_input["unit"]
        existing = next((i for i in cart if i["item"].lower() == item.lower()), None)
        if existing:
            existing["quantity"] += quantity
        else:
            cart.append({"item": item, "quantity": quantity, "unit": unit})
        return f"✅ Added {quantity} {unit} of {item} to cart. Cart has {len(cart)} item(s)."

    elif tool_name == "view_cart":
        if not cart:
            return "Cart is empty."
        lines = [f"{i+1}. {c['item']} - {c['quantity']} {c['unit']}" for i, c in enumerate(cart)]
        return "🛒 Your Cart:\n" + "\n".join(lines)

    elif tool_name == "get_offers":
        return """🏷️ Today's Offers:
1. Buy 50kg sugar, get 2kg free
2. Sunflower oil 15L pack — 10% discount
3. Basmati rice bulk order — free delivery above 100kg
4. Toor dal — fresh stock arrived"""

    return "Tool not found"

# ── Core logic helpers ─────────────────────────────────────────────────────
# These are plain sync functions called directly by graph nodes AND endpoints.
# Each wraps its Claude call in a Langfuse generation so every API call is
# recorded regardless of whether a parent span exists.

def _run_detect_intent(text: str) -> dict:
    # Open a generation observation. If langfuse is None, nullcontext is a no-op.
    ctx = langfuse.start_as_current_observation(
        name="claude-detect-intent",
        as_type="generation",
        model=MODEL,
        input=text,
    ) if langfuse else contextlib.nullcontext(None)

    with ctx as gen:
        response = client.messages.create(
            model=MODEL,
            max_tokens=100,
            system="""You are an intent classifier for a retail shopping voice bot.
Classify the user text into exactly one intent:
- add_to_cart: user wants to order or buy items
- get_offers: user wants deals, offers or recommendations
- other: greetings, questions, anything else

Respond ONLY with valid JSON. Example:
{"intent": "add_to_cart", "confidence": "high"}""",
            messages=[{"role": "user", "content": text}]
        )
        raw = response.content[0].text
        if gen:
            gen.update(
                output=raw,
                usage_details={
                    "input":  response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                },
            )

    return parse_json_response(raw)


def _run_extract_items(text: str) -> list[dict]:
    ctx = langfuse.start_as_current_observation(
        name="claude-extract-items",
        as_type="generation",
        model=MODEL,
        input=text,
    ) if langfuse else contextlib.nullcontext(None)

    with ctx as gen:
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            system="""You are an item extractor for a retail shopping voice bot in India.
Extract all items, quantities and units from the user's speech.
Common Indian units: kg, gram, litre, ml, dozen, piece, packet, bag, box

Respond ONLY with valid JSON. Example:
{"items": [{"item": "sugar", "quantity": 10, "unit": "kg"}]}""",
            messages=[{"role": "user", "content": text}]
        )
        raw = response.content[0].text
        if gen:
            gen.update(
                output=raw,
                usage_details={
                    "input":  response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                },
            )

    return parse_json_response(raw).get("items", [])

# ── Endpoint 1: Detect Intent ──────────────────────────────────────────────

@app.post("/detect-intent")
async def detect_intent(request: TextRequest) -> IntentResponse:
    # Outer span gives the trace a meaningful name in Langfuse.
    # _run_detect_intent creates a nested generation inside it.
    ctx = langfuse.start_as_current_observation(
        name="detect-intent", as_type="span", input=request.text
    ) if langfuse else contextlib.nullcontext(None)

    with ctx as span:
        result = _run_detect_intent(request.text)
        if span:
            span.update(output=result)

    return IntentResponse(**result)

# ── Endpoint 2: Extract Items ──────────────────────────────────────────────

@app.post("/extract-items")
async def extract_items(request: TextRequest) -> ExtractResponse:
    ctx = langfuse.start_as_current_observation(
        name="extract-items", as_type="span", input=request.text
    ) if langfuse else contextlib.nullcontext(None)

    with ctx as span:
        items = _run_extract_items(request.text)
        if span:
            span.update(output={"items": items})

    return ExtractResponse(items=[CartItem(**i) for i in items])

# ── Endpoint 3: Smart Chat with Tool Calling ───────────────────────────────

@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    # The entire multi-turn interaction is one span. Token usage is accumulated
    # across turns so we get the true total cost for this request.
    ctx = langfuse.start_as_current_observation(
        name="chat", as_type="span", input=request.message
    ) if langfuse else contextlib.nullcontext(None)

    with ctx as span:
        messages = [{"role": "user", "content": request.message}]
        total_input_tokens  = 0
        total_output_tokens = 0

        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=(
                "You are a voice shopping assistant for retail shop owners in India."
                " Help them build their order list."
                " Use tools to add items, view cart and get offers."
            ),
            tools=CART_TOOLS,
            messages=messages
        )
        total_input_tokens  += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        while response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Claude wants to call: {block.name} with {block.input}")
                    result = handle_tool_call(block.name, block.input)
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",      "content": tool_results})

            response = client.messages.create(
                model=MODEL,
                max_tokens=1000,
                system="You are a voice shopping assistant for retail shop owners in India.",
                tools=CART_TOOLS,
                messages=messages
            )
            total_input_tokens  += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

        final_reply = next(
            (block.text for block in response.content if hasattr(block, "text")),
            "Done!"
        )

        if span:
            # Record aggregate token usage and final reply on the span
            span.update(
                output=final_reply,
                usage_details={
                    "input":  total_input_tokens,
                    "output": total_output_tokens,
                },
            )

    return ChatResponse(reply=final_reply)

# ── Endpoint 4: LangGraph Pipeline ────────────────────────────────────────
# Imported here (after handle_tool_call is defined) so graph.py can safely
# do `from main import ...` without a circular-import error.

from graph import shopping_graph  # noqa: E402

class ProcessRequest(BaseModel):
    text: str

class ProcessResponse(BaseModel):
    intent: str
    confidence: str
    reply: str

@app.post("/process")
async def process(request: ProcessRequest) -> ProcessResponse:
    """
    Run the full LangGraph pipeline.

    Langfuse trace shape:
        [trace]
          └─ agent: shopping-graph          ← created here
               ├─ span: intent_router       ← created in graph.py node
               │    └─ generation: claude-detect-intent
               └─ span: cart_agent OR recommendation_agent
                    └─ generation: claude-extract-items  (cart path only)

    The parent trace_id and span_id are captured after opening the agent
    observation, then threaded through LangGraph state as plain strings.
    Each graph node uses TraceContext(trace_id, parent_span_id) to attach
    its child span to the correct parent — this is robust against LangGraph
    running nodes in a copied contextvars context.
    """
    ctx = langfuse.start_as_current_observation(
        name="shopping-graph", as_type="agent", input=request.text
    ) if langfuse else contextlib.nullcontext(None)

    with ctx as agent_span:
        # Capture IDs while the observation is the current one in OTel context
        lf_trace_id = langfuse.get_current_trace_id() if langfuse else None
        lf_span_id  = langfuse.get_current_observation_id() if langfuse else None

        initial_state = {
            "text":          request.text,
            "intent":        None,
            "confidence":    None,
            "tool_result":   None,
            "reply":         None,
            "_lf_trace_id":  lf_trace_id,   # passed to graph nodes for explicit parent linkage
            "_lf_span_id":   lf_span_id,
        }

        final_state = shopping_graph.invoke(initial_state)

        if agent_span:
            agent_span.update(output={
                "intent": final_state.get("intent"),
                "reply":  final_state.get("reply"),
            })

    return ProcessResponse(
        intent=final_state.get("intent", "other"),
        confidence=final_state.get("confidence", "low"),
        reply=final_state.get("reply") or "I'm not sure how to help with that.",
    )

from fastapi import FastAPI
from pydantic import BaseModel
import anthropic
import json
import re

app = FastAPI()

def parse_json_response(text: str) -> dict:
    text = text.strip()
    # Remove markdown code fences: ```json ... ``` or ``` ... ```
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    return json.loads(text.strip())
client = anthropic.Anthropic()

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

# ── Tool definitions — FastAPI tells Claude what tools exist ───────────────

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

# ── In-memory cart (same as MCP server for now) ────────────────────────────
# On Day 3 this will call the actual MCP server
cart = []

def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """FastAPI executes the tool — Claude never does this directly"""

    if tool_name == "add_to_cart":
        item     = tool_input["item"]
        quantity = tool_input["quantity"]
        unit     = tool_input["unit"]
        # Check if item exists already
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

# ── Endpoint 1: Detect Intent ──────────────────────────────────────────────

@app.post("/detect-intent")
async def detect_intent(request: TextRequest) -> IntentResponse:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        system="""You are an intent classifier for a retail shopping voice bot.
Classify the user text into exactly one intent:
- add_to_cart: user wants to order or buy items
- get_offers: user wants deals, offers or recommendations
- other: greetings, questions, anything else

Respond ONLY with valid JSON. Example:
{"intent": "add_to_cart", "confidence": "high"}""",
        messages=[{"role": "user", "content": request.text}]
    )
    result = parse_json_response(response.content[0].text)
    return IntentResponse(**result)

# ── Endpoint 2: Extract Items ──────────────────────────────────────────────

@app.post("/extract-items")
async def extract_items(request: TextRequest) -> ExtractResponse:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        system="""You are an item extractor for a retail shopping voice bot in India.
Extract all items, quantities and units from the user's speech.
Common Indian units: kg, gram, litre, ml, dozen, piece, packet, bag, box

Respond ONLY with valid JSON. Example:
{"items": [{"item": "sugar", "quantity": 10, "unit": "kg"}]}""",
        messages=[{"role": "user", "content": request.text}]
    )
    result = parse_json_response(response.content[0].text)
    items = [CartItem(**i) for i in result["items"]]
    return ExtractResponse(items=items)

# ── Endpoint 3: Smart Chat with Tool Calling ───────────────────────────────
# This is the upgraded /chat — Claude now actually adds to cart!

@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    messages = [{"role": "user", "content": request.message}]

    # Step 1: Call Claude with tool definitions
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        system="You are a voice shopping assistant for retail shop owners in India. Help them build their order list. Use tools to add items, view cart and get offers.",
        tools=CART_TOOLS,        # ← telling Claude what tools exist
        messages=messages
    )

    # Step 2: Check if Claude wants to call a tool
    while response.stop_reason == "tool_use":
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                print(f"Claude wants to call: {block.name} with {block.input}")

                # Step 3: FastAPI calls the tool — NOT Claude!
                result = handle_tool_call(block.name, block.input)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        # Step 4: Send tool result back to Claude
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user",      "content": tool_results})

        # Step 5: Claude gives final human-friendly reply
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            system="You are a voice shopping assistant for retail shop owners in India.",
            tools=CART_TOOLS,
            messages=messages
        )

    # Extract final text reply
    final_reply = next(
        (block.text for block in response.content if hasattr(block, "text")), 
        "Done!"
    )
    return ChatResponse(reply=final_reply)
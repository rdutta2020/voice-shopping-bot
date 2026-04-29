import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

// In-memory cart — like ArrayList in Java
const cart = [];

const server = new McpServer({
  name: "shopping-cart-server",
  version: "1.0.0"
});

// Tool 1: Add to Cart
server.tool(
  "add_to_cart",
  "Adds an item to the shopping cart",
  {
    item:     z.string().describe("Item name"),
    quantity: z.number().describe("How many units"),
    unit:     z.string().describe("kg, litre, dozen etc")
  },
  async ({ item, quantity, unit }) => {
    const existing = cart.find(i => i.item.toLowerCase() === item.toLowerCase());
    if (existing) {
      existing.quantity += quantity;
    } else {
      cart.push({ item, quantity, unit });
    }
    return {
      content: [{
        type: "text",
        text: `✅ Added ${quantity} ${unit} of ${item}. Cart has ${cart.length} item(s).`
      }]
    };
  }
);

// Tool 2: View Cart
server.tool(
  "view_cart",
  "Shows all items in the shopping cart",
  {},
  async () => {
    if (cart.length === 0) {
      return { content: [{ type: "text", text: "Cart is empty." }] };
    }
    const lines = cart.map((c, i) => `${i+1}. ${c.item} - ${c.quantity} ${c.unit}`).join("\n");
    return {
      content: [{ type: "text", text: `🛒 Cart:\n${lines}` }]
    };
  }
);

// Tool 3: Get Offers
server.tool(
  "get_offers",
  "Gets current offers and deals",
  { category: z.string().optional() },
  async ({ category }) => {
    return {
      content: [{
        type: "text",
        text: `🏷️ Offers:\n1. 50kg sugar — 2kg free\n2. Sunflower oil 15L — 10% off\n3. Basmati rice bulk — free delivery`
      }]
    };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
console.error("🛒 MCP Cart Server running...");
# FastAPI is like Spring Boot / Ktor for Python
from fastapi import FastAPI
from pydantic import BaseModel  # Like a data class in Kotlin
import anthropic                # Anthropic SDK — like Retrofit client
import os                       # To read environment variables

# Create the app — like new Application() in Android
app = FastAPI()

# Create Claude client — like building a Retrofit instance
# Reads ANTHROPIC_API_KEY from environment automatically
client = anthropic.Anthropic()

# Request body — like a Kotlin data class
# equivalent to: data class ChatRequest(val message: String)
class ChatRequest(BaseModel):
    message: str

# Response body — like a Kotlin data class
# equivalent to: data class ChatResponse(val reply: String)
class ChatResponse(BaseModel):
    reply: str

# POST endpoint — like @POST("/chat") in Retrofit
# equivalent to: @PostMapping("/chat") in Spring Boot
@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:

    # Call Claude API — like calling retrofit.create(ApiService::class.java)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,

        # System prompt — makes Claude behave as shopping assistant
        # This is like passing a config to your API client
        system="You are a voice shopping assistant for retail shop owners in India. Help them build their shopping order list by identifying items and quantities from their speech.",

        # The user's message — like the request body
        messages=[
            {"role": "user", "content": request.message}
        ]
    )

    # Extract text from response — like response.body()?.reply in Retrofit
    reply_text = response.content[0].text

    # Return response — Python automatically converts this to JSON
    return ChatResponse(reply=reply_text)
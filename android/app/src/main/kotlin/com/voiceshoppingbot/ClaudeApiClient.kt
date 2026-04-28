package com.voiceshoppingbot

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class ClaudeApiClient(private val apiKey: String) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    private val history = mutableListOf<Pair<String, String>>()

    suspend fun sendMessage(userMessage: String): String = withContext(Dispatchers.IO) {
        history.add("user" to userMessage)

        val messages = JSONArray()
        history.forEach { (role, content) ->
            messages.put(JSONObject().apply {
                put("role", role)
                put("content", content)
            })
        }

        val body = JSONObject().apply {
            put("model", "claude-sonnet-4-6")
            put("max_tokens", 512)
            put("system", SYSTEM_PROMPT)
            put("messages", messages)
        }

        val request = Request.Builder()
            .url("https://api.anthropic.com/v1/messages")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .header("x-api-key", apiKey)
            .header("anthropic-version", "2023-06-01")
            .build()

        val response = client.newCall(request).execute()
        val responseBody = response.body?.string() ?: throw Exception("Empty response from Claude")

        if (!response.isSuccessful) {
            throw Exception("Claude API error ${response.code}: $responseBody")
        }

        val reply = JSONObject(responseBody)
            .getJSONArray("content")
            .getJSONObject(0)
            .getString("text")

        history.add("assistant" to reply)
        reply
    }

    fun clearHistory() = history.clear()

    fun shutdown() {
        client.dispatcher.executorService.shutdown()
    }

    companion object {
        private const val SYSTEM_PROMPT =
            "You are a voice shopping assistant for retail shop owners in India. " +
            "Help them build their shopping order list by identifying items and quantities " +
            "from their speech. Keep a running mental note of what has been ordered so far " +
            "in the conversation and confirm each addition. Keep responses short and " +
            "conversational since they will be spoken aloud. Use plain text only — no " +
            "markdown, bullet points, or special characters."
    }
}

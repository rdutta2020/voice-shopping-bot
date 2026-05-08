package com.voiceshoppingbot

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class BackendApiClient {

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(90, TimeUnit.SECONDS) // /process runs 2 Claude calls internally
        .build()

    /**
     * POST {"text": text} to /process.
     * Returns the "reply" field from {"intent", "confidence", "reply"}.
     */
    suspend fun sendMessage(text: String): String = withContext(Dispatchers.IO) {
        val body = JSONObject().apply { put("text", text) }

        val request = Request.Builder()
            .url(BASE_URL)
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val response = client.newCall(request).execute()
        val responseBody = response.body?.string()
            ?: throw Exception("Empty response from backend")

        if (!response.isSuccessful) {
            throw Exception("Backend error ${response.code}: $responseBody")
        }

        JSONObject(responseBody).getString("reply")
    }

    fun shutdown() {
        client.dispatcher.executorService.shutdown()
    }

    companion object {
        // 10.0.2.2 is the Android emulator's alias for the host machine's loopback.
        // Change to your LAN IP when testing on a physical device.
        private const val BASE_URL = "http://10.0.2.2:8000/process"
    }
}

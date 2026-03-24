package com.toori.sdk

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

class TooriClient(
    private val baseUrl: String = "http://127.0.0.1:7777",
    private val apiKey: String? = null,
    private val client: OkHttpClient = OkHttpClient(),
) {
    fun settings(): String = execute("/v1/settings", "GET", null)

    fun query(query: String, sessionId: String = "default", topK: Int = 6): String {
        val body = """{"query":"$query","session_id":"$sessionId","top_k":$topK}"""
        return execute("/v1/query", "POST", body)
    }

    private fun execute(path: String, method: String, body: String?): String {
        val builder = Request.Builder().url("$baseUrl$path")
        if (apiKey != null) {
            builder.addHeader("X-API-Key", apiKey)
        }
        if (method == "POST") {
            builder.post((body ?: "{}").toRequestBody("application/json".toMediaType()))
        } else {
            builder.get()
        }
        client.newCall(builder.build()).execute().use { response ->
            require(response.isSuccessful) { "Request failed: ${response.code}" }
            return response.body!!.string()
        }
    }
}

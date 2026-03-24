package com.toori.app.data

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.Base64

class TooriRuntimeClient(
    private val baseUrl: String = System.getenv("TOORI_RUNTIME_URL") ?: "http://10.0.2.2:7777",
    private val client: OkHttpClient = OkHttpClient(),
) {
    fun analyze(imageBytes: ByteArray, sessionId: String, prompt: String?): Pair<Answer?, List<SearchHit>> {
        val payload = JSONObject()
            .put("image_base64", Base64.getEncoder().encodeToString(imageBytes))
            .put("session_id", sessionId)
            .put("decode_mode", "auto")
        if (!prompt.isNullOrBlank()) {
            payload.put("query", prompt)
        }
        val request = Request.Builder()
            .url("$baseUrl/v1/analyze")
            .post(payload.toString().toRequestBody("application/json".toMediaType()))
            .build()
        client.newCall(request).execute().use { response ->
            require(response.isSuccessful) { "Analyze failed: ${response.code}" }
            val body = JSONObject(response.body!!.string())
            val answer = body.optJSONObject("answer")?.let { Answer(it.getString("text"), it.getString("provider")) }
            val hits = parseHits(body.getJSONArray("hits"))
            return answer to hits
        }
    }

    fun search(query: String, sessionId: String, topK: Int): Pair<Answer?, List<SearchHit>> {
        val payload = JSONObject()
            .put("query", query)
            .put("session_id", sessionId)
            .put("top_k", topK)
        val request = Request.Builder()
            .url("$baseUrl/v1/query")
            .post(payload.toString().toRequestBody("application/json".toMediaType()))
            .build()
        client.newCall(request).execute().use { response ->
            require(response.isSuccessful) { "Search failed: ${response.code}" }
            val body = JSONObject(response.body!!.string())
            val answer = body.optJSONObject("answer")?.let { Answer(it.getString("text"), it.getString("provider")) }
            return answer to parseHits(body.getJSONArray("hits"))
        }
    }

    fun fetchObservations(sessionId: String): List<Observation> {
        val request = Request.Builder()
            .url("$baseUrl/v1/observations?session_id=$sessionId&limit=48")
            .get()
            .build()
        client.newCall(request).execute().use { response ->
            require(response.isSuccessful) { "Observation fetch failed: ${response.code}" }
            val body = JSONObject(response.body!!.string())
            val observations = body.getJSONArray("observations")
            return List(observations.length()) { index ->
                val item = observations.getJSONObject(index)
                Observation(
                    id = item.getString("id"),
                    session_id = item.getString("session_id"),
                    created_at = item.getString("created_at"),
                    thumbnail_path = item.getString("thumbnail_path"),
                    summary = item.optString("summary").ifBlank { null },
                    providers = item.getJSONArray("providers").toList().map { it.toString() },
                )
            }
        }
    }

    private fun parseHits(array: JSONArray): List<SearchHit> = List(array.length()) { index ->
        val item = array.getJSONObject(index)
        SearchHit(
            observation_id = item.getString("observation_id"),
            score = item.getDouble("score"),
            summary = item.optString("summary").ifBlank { null },
            thumbnail_path = item.getString("thumbnail_path"),
            created_at = item.getString("created_at"),
        )
    }

    private fun JSONArray.toList(): List<Any?> = List(length()) { index -> get(index) }
}

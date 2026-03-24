package com.toori.app.data

data class RuntimeSettings(
    val runtime_profile: String = "hybrid",
    val sampling_fps: Float = 1f,
    val top_k: Int = 6,
    val retention_days: Int = 30,
    val primary_perception_provider: String = "tflite",
    val reasoning_backend: String = "cloud",
    val local_reasoning_disabled: Boolean = true,
)

data class Observation(
    val id: String,
    val session_id: String,
    val created_at: String,
    val thumbnail_path: String,
    val summary: String? = null,
    val providers: List<String> = emptyList(),
)

data class SearchHit(
    val observation_id: String,
    val score: Double,
    val summary: String? = null,
    val thumbnail_path: String,
    val created_at: String,
)

data class Answer(
    val text: String,
    val provider: String,
)

data class ProviderHealth(
    val name: String,
    val healthy: Boolean,
    val message: String,
)

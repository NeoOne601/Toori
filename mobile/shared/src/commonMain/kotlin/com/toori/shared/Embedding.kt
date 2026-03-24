package com.toori.shared

data class ProviderConfig(
    val name: String,
    val enabled: Boolean = true,
    val baseUrl: String? = null,
    val model: String? = null,
    val modelPath: String? = null,
)

data class RuntimeSettings(
    val runtimeProfile: String = "hybrid",
    val samplingFps: Float = 1f,
    val topK: Int = 6,
    val retentionDays: Int = 30,
    val primaryPerceptionProvider: String = "platform-native",
    val reasoningBackend: String = "cloud",
    val localReasoningDisabled: Boolean = true,
    val providers: Map<String, ProviderConfig> = emptyMap(),
)

data class Observation(
    val id: String,
    val sessionId: String,
    val createdAt: String,
    val thumbnailPath: String,
    val summary: String? = null,
    val sourceQuery: String? = null,
    val confidence: Float = 0f,
    val novelty: Float = 0f,
    val providers: List<String> = emptyList(),
)

data class SearchHit(
    val observationId: String,
    val score: Float,
    val summary: String? = null,
    val thumbnailPath: String,
    val sessionId: String,
    val createdAt: String,
)

data class Answer(
    val text: String,
    val provider: String,
    val confidence: Float = 0f,
)

interface Encoder {
    fun encode(input: ByteArray): List<Float>
}

expect fun getPlatformEncoder(): Encoder

object EncoderFactory {
    fun create(): Encoder = getPlatformEncoder()
}

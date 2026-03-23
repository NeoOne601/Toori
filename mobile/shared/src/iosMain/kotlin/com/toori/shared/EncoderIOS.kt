package com.toori.shared

actual fun getPlatformEncoder(): Encoder = object : Encoder {
    // Simple embedding placeholder: generate a non-empty string based on input hash
    override fun encode(input: String): String {
        val hash = input.hashCode()
        // Ensure non-zero embedding representation
        return "iOSEmbedding:${hash}"
    }
}

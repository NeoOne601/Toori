package com.toori.shared

actual fun getPlatformEncoder(): Encoder = object : Encoder {
    override fun encode(input: ByteArray): List<Float> {
        if (input.isEmpty()) {
            return List(128) { 0f }
        }
        val descriptor = MutableList(128) { 0f }
        input.forEachIndexed { index, byte ->
            val weight = ((index % 11) + 1).toFloat() / 11f
            descriptor[index % 128] += ((byte.toInt() and 0xFF) / 255f) * weight
        }
        val magnitude = kotlin.math.sqrt(descriptor.sumOf { (it * it).toDouble() }).toFloat().takeIf { it > 0f } ?: 1f
        return descriptor.map { it / magnitude }
    }
}

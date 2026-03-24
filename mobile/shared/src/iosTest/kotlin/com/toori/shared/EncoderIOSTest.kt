package com.toori.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EncoderIOSTest {
    @Test
    fun testEncoderProducesNonZeroEmbedding() {
        val encoder = getPlatformEncoder()
        val result = encoder.encode("test input".encodeToByteArray())
        assertEquals(128, result.size)
        assertTrue(result.any { it > 0f }, "Result should contain non-zero values")
    }
}

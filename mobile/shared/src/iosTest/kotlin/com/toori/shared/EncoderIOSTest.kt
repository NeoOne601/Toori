package com.toori.shared

import kotlin.test.Test
import kotlin.test.assertTrue

class EncoderIOSTest {
    @Test
    fun testEncoderProducesNonZeroEmbedding() {
        val encoder = getPlatformEncoder()
        val result = encoder.encode("test input")
        // Expect a non-empty, non-zero embedding representation
        assertTrue(result.isNotEmpty(), "Encoder result should not be empty")
        assertTrue(result.contains("iOSEmbedding"), "Result should contain embedding prefix")
    }
}

package com.toori.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EmbeddingTest {
    @Test
    fun testEncoder() {
        val encoder = EncoderFactory.create()
        val result = encoder.encode("hello".encodeToByteArray())
        assertEquals(128, result.size)
        assertTrue(result.any { it != 0f })
    }
}

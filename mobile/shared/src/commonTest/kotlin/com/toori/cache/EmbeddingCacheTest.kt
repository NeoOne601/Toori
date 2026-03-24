package com.toori.cache

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class EmbeddingCacheTest {
    @Test
    fun testInsertAndRetrieve() {
        val cache = getEmbeddingCache()
        val key = "sampleKey"
        val embedding = listOf(0.1f, 0.2f, 0.3f)
        cache.put(key, embedding)
        val retrieved = cache.get(key)
        assertNotNull(retrieved, "Embedding should be retrieved after insertion")
        assertEquals(embedding, retrieved, "Retrieved embedding should match inserted value")
    }
}

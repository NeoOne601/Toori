package com.toori.cache

import java.util.LinkedHashMap

/**
 * Simple in‑memory LRU cache for embeddings.
 *
 * - Max 10 000 entries.
 * - Time‑to‑live 48 hours.
 * - Uses access‑order LinkedHashMap to evict the least‑recently‑used entry when the size limit is exceeded.
 */
class EmbeddingCache {
    private val maxEntries = 10_000
    private val ttlMs = 48L * 60 * 60 * 1000 // 48 hours in milliseconds

    private data class Entry(val embedding: List<Float>, val timestamp: Long)

    // LinkedHashMap with accessOrder = true provides LRU behaviour.
    private val map = object : LinkedHashMap<String, Entry>(16, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, Entry>?): Boolean {
            return size > maxEntries
        }
    }

    /**
     * Store an embedding for the given key.
     */
    fun put(key: String, embedding: List<Float>) {
        val now = System.currentTimeMillis()
        map[key] = Entry(embedding, now)
    }

    /**
     * Retrieve an embedding for the given key, or null if missing or expired.
     */
    fun get(key: String): List<Float>? {
        val entry = map[key] ?: return null
        val now = System.currentTimeMillis()
        // If the entry is older than TTL, remove and treat as missing.
        if (now - entry.timestamp > ttlMs) {
            map.remove(key)
            return null
        }
        return entry.embedding
    }
}

// Platform‑specific accessor – actual implementations are provided per target.
expect fun getEmbeddingCache(): EmbeddingCache

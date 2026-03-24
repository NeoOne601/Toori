package com.toori.app.ui

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.toori.app.data.Answer
import com.toori.app.data.Observation
import com.toori.app.data.RuntimeSettings
import com.toori.app.data.SearchHit
import com.toori.app.data.TooriRuntimeClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream

class TooriViewModel(application: Application) : AndroidViewModel(application) {
    private val client = TooriRuntimeClient()

    private val _status = MutableStateFlow("Idle")
    private val _answer = MutableStateFlow<Answer?>(null)
    private val _hits = MutableStateFlow<List<SearchHit>>(emptyList())
    private val _observations = MutableStateFlow<List<Observation>>(emptyList())
    private val _settings = MutableStateFlow(RuntimeSettings())

    val status: StateFlow<String> = _status.asStateFlow()
    val answer: StateFlow<Answer?> = _answer.asStateFlow()
    val hits: StateFlow<List<SearchHit>> = _hits.asStateFlow()
    val observations: StateFlow<List<Observation>> = _observations.asStateFlow()
    val settings: StateFlow<RuntimeSettings> = _settings.asStateFlow()

    var sessionId: String = "android-live"
    var prompt: String = ""
    var searchText: String = ""

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { client.fetchObservations(sessionId) }
                .onSuccess {
                    _observations.value = it
                    _status.value = "Connected to runtime"
                }
                .onFailure { _status.value = it.message ?: "Runtime unavailable" }
        }
    }

    fun analyzeBitmap(bitmap: Bitmap) {
        viewModelScope.launch(Dispatchers.IO) {
            _status.value = "Analyzing frame"
            val bytes = ByteArrayOutputStream().use { output ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)
                output.toByteArray()
            }
            runCatching { client.analyze(bytes, sessionId, prompt.ifBlank { null }) }
                .onSuccess { result ->
                    _answer.value = result.first
                    _hits.value = result.second
                    refresh()
                }
                .onFailure { _status.value = it.message ?: "Analyze failed" }
        }
    }

    fun search() {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { client.search(searchText, sessionId, _settings.value.top_k) }
                .onSuccess { result ->
                    _answer.value = result.first
                    _hits.value = result.second
                }
                .onFailure { _status.value = it.message ?: "Search failed" }
        }
    }

    fun fileToBitmap(path: String): Bitmap? = BitmapFactory.decodeFile(path)
}

package com.toori.app.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.io.File

@Composable
fun TooriAndroidApp(viewModel: TooriViewModel) {
    val status by viewModel.status.collectAsState()
    var selectedTab by remember { mutableStateOf("Lens") }

    Scaffold(
        bottomBar = {
            NavigationBar {
                listOf("Lens", "Search", "Replay", "Integrations", "Settings").forEach { tab ->
                    NavigationBarItem(
                        selected = selectedTab == tab,
                        onClick = { selectedTab = tab },
                        label = { Text(tab) },
                        icon = {},
                    )
                }
            }
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(text = status, style = MaterialTheme.typography.bodySmall)
            when (selectedTab) {
                "Lens" -> LensScreen(viewModel)
                "Search" -> SearchScreen(viewModel)
                "Replay" -> ReplayScreen(viewModel)
                "Integrations" -> IntegrationsScreen()
                else -> SettingsScreen(viewModel)
            }
        }
    }
}

@Composable
private fun LensScreen(viewModel: TooriViewModel) {
    val context = LocalContext.current
    val answer by viewModel.answer.collectAsState()
    val hits by viewModel.hits.collectAsState()
    val imageCapture = remember { ImageCapture.Builder().build() }

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        CameraPreview(imageCapture = imageCapture)
        OutlinedTextField(
            value = viewModel.prompt,
            onValueChange = { viewModel.prompt = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Prompt") },
        )
        Button(onClick = {
            val file = File(context.cacheDir, "toori_capture.png")
            val output = ImageCapture.OutputFileOptions.Builder(file).build()
            imageCapture.takePicture(
                output,
                ContextCompat.getMainExecutor(context),
                object : ImageCapture.OnImageSavedCallback {
                    override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                        val bitmap = BitmapFactory.decodeFile(file.absolutePath)
                        if (bitmap != null) {
                            viewModel.analyzeBitmap(bitmap)
                        }
                    }

                    override fun onError(exception: ImageCaptureException) {
                        viewModel.prompt = exception.message ?: "Capture failed"
                    }
                }
            )
        }) {
            Text("Capture and analyze")
        }
        answer?.let { Text(it.text) }
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(hits) { hit ->
                Text("${hit.summary ?: hit.observation_id} • ${"%.2f".format(hit.score)}")
            }
        }
    }
}

@Composable
private fun SearchScreen(viewModel: TooriViewModel) {
    val hits by viewModel.hits.collectAsState()
    val answer by viewModel.answer.collectAsState()
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        OutlinedTextField(
            value = viewModel.searchText,
            onValueChange = { viewModel.searchText = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Search memory") },
        )
        Button(onClick = { viewModel.search() }) { Text("Run search") }
        answer?.let { Text(it.text) }
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(hits) { hit -> Text(hit.summary ?: hit.observation_id) }
        }
    }
}

@Composable
private fun ReplayScreen(viewModel: TooriViewModel) {
    val observations by viewModel.observations.collectAsState()
    LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        items(observations) { observation ->
            Column {
                Text(observation.summary ?: observation.id)
                Text(observation.created_at, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

@Composable
private fun IntegrationsScreen() {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Runtime endpoints")
        Text("POST /v1/analyze")
        Text("POST /v1/query")
        Text("GET /v1/providers/health")
        Text("WS /v1/events")
    }
}

@Composable
private fun SettingsScreen(viewModel: TooriViewModel) {
    val settings by viewModel.settings.collectAsState()
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Primary perception: ${settings.primary_perception_provider}")
        Text("Reasoning backend: ${settings.reasoning_backend}")
        Text("Top K: ${settings.top_k}")
        Text("Retention days: ${settings.retention_days}")
    }
}

@Composable
private fun CameraPreview(imageCapture: ImageCapture) {
    val context = LocalContext.current
    AndroidView(
        modifier = Modifier
            .fillMaxWidth()
            .padding(bottom = 8.dp),
        factory = { ctx ->
            val previewView = PreviewView(ctx)
            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    context as androidx.lifecycle.LifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture,
                )
            }, ContextCompat.getMainExecutor(ctx))
            previewView
        }
    )
}

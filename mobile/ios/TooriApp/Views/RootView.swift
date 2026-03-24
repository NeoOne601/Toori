import AVFoundation
import SwiftUI

struct RootView: View {
    @EnvironmentObject private var viewModel: LensAppViewModel

    var body: some View {
        TabView {
            LensView()
                .tabItem { Label("Lens", systemImage: "camera.viewfinder") }
            SearchView()
                .tabItem { Label("Search", systemImage: "magnifyingglass") }
            ReplayView()
                .tabItem { Label("Replay", systemImage: "film.stack") }
            IntegrationsView()
                .tabItem { Label("Integrations", systemImage: "puzzlepiece.extension") }
            SettingsView()
                .tabItem { Label("Settings", systemImage: "slider.horizontal.3") }
        }
        .task {
            await viewModel.bootstrap()
        }
        .tint(.orange)
    }
}

private struct LensView: View {
    @EnvironmentObject private var viewModel: LensAppViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    CameraPreview(session: viewModel.camera.session)
                        .frame(height: 360)
                        .clipShape(RoundedRectangle(cornerRadius: 24))
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Live Prompt").font(.headline)
                        TextField("Ask what the lens should explain", text: $viewModel.prompt, axis: .vertical)
                            .textFieldStyle(.roundedBorder)
                        Button("Capture and analyze") {
                            Task { await viewModel.captureAndAnalyze() }
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    .padding()
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 24))
                    if let answer = viewModel.latestAnswer {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Latest answer").font(.headline)
                            Text(answer.text)
                            Text("Provider: \(answer.provider)")
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24))
                    }
                    ForEach(viewModel.latestHits, id: \.observation_id) { hit in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(hit.summary ?? hit.observation_id).font(.headline)
                            Text("Score \(hit.score, specifier: "%.2f")")
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 20))
                    }
                }
                .padding()
            }
            .navigationTitle("Lens")
            .toolbar { Text(viewModel.status).font(.caption).foregroundStyle(.secondary) }
        }
    }
}

private struct SearchView: View {
    @EnvironmentObject private var viewModel: LensAppViewModel

    var body: some View {
        NavigationStack {
            List {
                Section("Search memory") {
                    TextField("blue notebook, hallway motion...", text: $viewModel.searchText)
                    Button("Run search") {
                        Task { await viewModel.runSearch() }
                    }
                }
                if let answer = viewModel.searchAnswer {
                    Section("Answer") {
                        Text(answer.text)
                    }
                }
                Section("Results") {
                    ForEach(viewModel.searchHits, id: \.observation_id) { hit in
                        VStack(alignment: .leading) {
                            Text(hit.summary ?? hit.observation_id)
                            Text(hit.created_at).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Search")
        }
    }
}

private struct ReplayView: View {
    @EnvironmentObject private var viewModel: LensAppViewModel

    var body: some View {
        NavigationStack {
            List(viewModel.observations) { observation in
                VStack(alignment: .leading, spacing: 4) {
                    Text(observation.summary ?? observation.id).font(.headline)
                    Text(observation.created_at).font(.caption).foregroundStyle(.secondary)
                    Text("Providers: \(observation.providers.joined(separator: ", "))")
                        .font(.caption)
                }
            }
            .navigationTitle("Replay")
        }
    }
}

private struct IntegrationsView: View {
    @EnvironmentObject private var viewModel: LensAppViewModel

    var body: some View {
        NavigationStack {
            List {
                Section("Runtime") {
                    Text("POST /v1/analyze")
                    Text("POST /v1/query")
                    Text("GET /v1/providers/health")
                    Text("WS /v1/events")
                }
                Section("Provider health") {
                    ForEach(viewModel.health, id: \.name) { provider in
                        VStack(alignment: .leading) {
                            Text(provider.name).font(.headline)
                            Text(provider.message).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Integrations")
        }
    }
}

private struct SettingsView: View {
    @EnvironmentObject private var viewModel: LensAppViewModel

    var body: some View {
        NavigationStack {
            Form {
                if viewModel.settings != nil {
                    Section("Runtime") {
                        TextField(
                            "Runtime profile",
                            text: Binding(
                                get: { viewModel.settings?.runtime_profile ?? "" },
                                set: { viewModel.settings?.runtime_profile = $0 }
                            )
                        )
                        Stepper(
                            value: Binding(
                                get: { viewModel.settings?.top_k ?? 6 },
                                set: { viewModel.settings?.top_k = $0 }
                            ),
                            in: 1...20
                        ) {
                            Text("Top K: \(viewModel.settings?.top_k ?? 6)")
                        }
                        Toggle(
                            "Disable local reasoning",
                            isOn: Binding(
                                get: { viewModel.settings?.local_reasoning_disabled ?? true },
                                set: { viewModel.settings?.local_reasoning_disabled = $0 }
                            )
                        )
                    }
                    Section("Providers") {
                        TextField(
                            "Primary perception",
                            text: Binding(
                                get: { viewModel.settings?.primary_perception_provider ?? "" },
                                set: { viewModel.settings?.primary_perception_provider = $0 }
                            )
                        )
                        TextField(
                            "Reasoning backend",
                            text: Binding(
                                get: { viewModel.settings?.reasoning_backend ?? "" },
                                set: { viewModel.settings?.reasoning_backend = $0 }
                            )
                        )
                    }
                    Button("Save settings") {
                        Task { await viewModel.saveSettings() }
                    }
                } else {
                    Text("Loading settings")
                }
            }
            .navigationTitle("Settings")
        }
    }
}

private struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = session
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.videoPreviewLayer.session = session
    }
}

private final class PreviewView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }
}

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
    @FocusState private var isPromptFocused: Bool
    @State private var currentZoom: CGFloat = 1.0

    var body: some View {
        ZStack {
            // Immersive Background Camera
            CameraPreview(session: viewModel.camera.session)
                .ignoresSafeArea()
                .onTapGesture {
                    isPromptFocused = false
                }
                .gesture(
                    MagnificationGesture()
                        .onChanged { value in
                            let delta = value / currentZoom
                            currentZoom = value
                            viewModel.camera.setZoom(viewModel.camera.zoomFactor * delta)
                        }
                        .onEnded { _ in
                            currentZoom = 1.0
                        }
                )

            // Top Status Pill
            VStack {
                if !viewModel.status.isEmpty && viewModel.status != "Idle" {
                    StatusPill(text: viewModel.status)
                        .padding(.top, 12)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }
                Spacer()
            }

            // Central Results (if any)
            if let answer = viewModel.latestAnswer {
                VStack {
                    Spacer()
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Discovery").font(.system(size: 14, weight: .bold)).capsuleStyle()
                            Spacer()
                            Button {
                                viewModel.latestAnswer = nil
                            } label: {
                                Image(systemName: "xmark.circle.fill").foregroundStyle(.white.opacity(0.4))
                            }
                        }
                        Text(answer.text)
                            .font(.system(size: 16, weight: .medium))
                            .foregroundStyle(.white)
                        Text("Source: \(answer.provider)")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.5))
                    }
                    .padding(20)
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
                    .padding(.horizontal, 20)
                    .padding(.bottom, 120) // Above floating bar
                }
            }

            // Bottom Floating Interaction Bar
            VStack {
                Spacer()
                HStack(spacing: 12) {
                    HStack {
                        Image(systemName: "sparkles")
                            .foregroundStyle(.orange)
                        TextField("Ask the lens...", text: $viewModel.prompt)
                            .focused($isPromptFocused)
                            .submitLabel(.search)
                            .onSubmit {
                                analyze()
                            }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(.white.opacity(0.12))
                    .clipShape(Capsule())

                    Button {
                        analyze()
                    } label: {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 38))
                            .symbolRenderingMode(.multicolor)
                    }
                }
                .padding(12)
                .background(.ultraThinMaterial)
                .clipShape(Capsule())
                .padding(.horizontal, 20)
                .padding(.bottom, 20)
                .shadow(color: .black.opacity(0.3), radius: 20, x: 0, y: 10)
            }
        }
        .animation(.spring(), value: viewModel.status)
        .preferredColorScheme(.dark)
    }

    private func analyze() {
        isPromptFocused = false
        Task {
            await viewModel.captureAndAnalyze()
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

private struct StatusPill: View {
    let text: String
    var body: some View {
        Text(text)
            .font(.system(size: 13, weight: .bold))
            .foregroundStyle(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(backgroundColor.opacity(0.8))
            .clipShape(Capsule())
            .shadow(radius: 10)
    }
    
    private var backgroundColor: Color {
        if text.contains("Searching") { return .gray }
        if text.contains("Discovered") || text.contains("Connected") { return .orange }
        return .red // Error state
    }
}

private extension Text {
    func capsuleStyle() -> some View {
        self
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(.white.opacity(0.2))
            .clipShape(Capsule())
    }
}

private struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
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

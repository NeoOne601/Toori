import SwiftUI

struct RootView: View {
    @EnvironmentObject private var viewModel: LensAppViewModel
    @State private var showingSettings = false
    @State private var showingSearch = false

    var body: some View {
        NavigationStack {
            ZStack {
                LensView()
                    .edgesIgnoringSafeArea(.all)
                
                VStack {
                    HStack {
                        Button {
                            showingSettings = true
                        } label: {
                            Image(systemName: "gearshape.fill")
                                .font(.title2)
                                .padding()
                                .background(.ultraThinMaterial)
                                .clipShape(Circle())
                        }
                        
                        Spacer()
                        
                        Text(viewModel.status)
                            .font(.caption2)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(.ultraThinMaterial)
                            .clipShape(Capsule())
                        
                        Spacer()
                        
                        Button {
                            showingSearch = true
                        } label: {
                            Image(systemName: "magnifyingglass")
                                .font(.title2)
                                .padding()
                                .background(.ultraThinMaterial)
                                .clipShape(Circle())
                        }
                    }
                    .padding()
                    
                    Spacer()
                    
                    VStack(spacing: 16) {
                        if let summary = viewModel.groundedSummary {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text("DISCOVERY")
                                        .font(.caption2.bold())
                                        .foregroundColor(.secondary)
                                    Spacer()
                                    if let confidence = viewModel.confidenceLabel {
                                        Text(confidence)
                                            .font(.caption2.bold())
                                            .foregroundColor(confidence == "Grounded" ? .green : .orange)
                                    }
                                }
                                Text(summary)
                                    .font(.headline)
                                    .multilineTextAlignment(.leading)
                            }
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(.ultraThinMaterial)
                            .cornerRadius(12)
                        }
                        
                        if let answer = viewModel.latestAnswer {
                            VStack(alignment: .leading, spacing: 8) {
                                Text(answer.text)
                                    .font(.subheadline)
                                    .fixedSize(horizontal: false, vertical: true)
                                
                                HStack {
                                    Text(answer.provider.uppercased())
                                        .font(.system(size: 8, weight: .black))
                                        .padding(.horizontal, 4)
                                        .padding(.vertical, 2)
                                        .background(.primary)
                                        .foregroundColor(.black)
                                        .cornerRadius(2)
                                    
                                    Spacer()
                                    
                                    Text("\(Int(answer.confidence * 100))% confidence")
                                        .font(.system(size: 8))
                                        .foregroundColor(.secondary)
                                }
                            }
                            .padding()
                            .background(.ultraThinMaterial)
                            .cornerRadius(12)
                        }
                        
                        HStack(spacing: 12) {
                            TextField("Ask reality...", text: $viewModel.prompt)
                                .textFieldStyle(.plain)
                                .padding()
                                .background(.ultraThinMaterial)
                                .cornerRadius(25)
                                .onSubmit {
                                    Task { await viewModel.captureAndAnalyze() }
                                }
                            
                            Button {
                                Task { await viewModel.captureAndAnalyze() }
                            } label: {
                                Image(systemName: "arrow.up.circle.fill")
                                    .font(.system(size: 44))
                                    .symbolRenderingMode(.hierarchical)
                            }
                        }
                    }
                    .padding()
                }
            }
            .sheet(isPresented: $showingSettings) {
                SettingsView()
                    .environmentObject(viewModel)
            }
            .sheet(isPresented: $showingSearch) {
                SearchView()
                    .environmentObject(viewModel)
            }
            .task {
                await viewModel.bootstrap()
            }
        }
    }
}

private struct SettingsView: View {
    @EnvironmentObject private var viewModel: LensAppViewModel
    
    let perceptionProviders = ["dinov2", "onnx", "basic", "cloud"]
    let reasoningBackends = ["mlx", "ollama", "cloud"]
    let gemmaModels = [
        "mlx-community/gemma-4-e2b-it-4bit": "Gemma 4 (2B) - Default",
        "mlx-community/gemma-4-e4b-it-4bit": "Gemma 4 (4B) - Heavy"
    ]

    var body: some View {
        NavigationStack {
            Form {
                if let settings = viewModel.settings {
                    Section("Reality Intelligence") {
                        Toggle("Open-Vocab Labels", isOn: Binding(
                            get: { viewModel.settings?.live_features.open_vocab_labels_enabled ?? true },
                            set: { viewModel.settings?.live_features.open_vocab_labels_enabled = $0 }
                        ))
                        
                        Toggle("TVLC Context Analysis", isOn: Binding(
                            get: { viewModel.settings?.live_features.tvlc_enabled ?? true },
                            set: { viewModel.settings?.live_features.tvlc_enabled = $0 }
                        ))
                        
                        Toggle("JEPA Temporal Ticks", isOn: Binding(
                            get: { viewModel.settings?.live_features.live_lens_use_jepa_tick ?? true },
                            set: { viewModel.settings?.live_features.live_lens_use_jepa_tick = $0 }
                        ))
                    }
                    
                    Section("Local Reasoning (MLX)") {
                        Toggle("Disable Local AI", isOn: Binding(
                            get: { viewModel.settings?.local_reasoning_disabled ?? false },
                            set: { viewModel.settings?.local_reasoning_disabled = $0 }
                        ))
                        
                        Picker("Gemma Model", selection: Binding(
                            get: { viewModel.settings?.providers["mlx"]?.model ?? "mlx-community/gemma-4-e2b-it-4bit" },
                            set: { viewModel.settings?.providers["mlx"]?.model = $0 }
                        )) {
                            ForEach(gemmaModels.keys.sorted(), id: \.self) { key in
                                Text(gemmaModels[key] ?? key).tag(key)
                            }
                        }
                    }

                    Section("Pipeline Architecture") {
                        Picker("Perception", selection: Binding(
                            get: { viewModel.settings?.primary_perception_provider ?? "dinov2" },
                            set: { viewModel.settings?.primary_perception_provider = $0 }
                        )) {
                            ForEach(perceptionProviders, id: \.self) { Text($0).tag($0) }
                        }
                        
                        Picker("Reasoning", selection: Binding(
                            get: { viewModel.settings?.reasoning_backend ?? "mlx" },
                            set: { viewModel.settings?.reasoning_backend = $0 }
                        )) {
                            ForEach(reasoningBackends, id: \.self) { Text($0).tag($0) }
                        }
                        
                        Stepper("Top K: \(viewModel.settings?.top_k ?? 6)", value: Binding(
                            get: { viewModel.settings?.top_k ?? 6 },
                            set: { viewModel.settings?.top_k = $0 }
                        ), in: 1...20)
                    }
                    
                    Section {
                        Button("Save & Synchronize") {
                            Task { await viewModel.saveSettings() }
                        }
                        .frame(maxWidth: .infinity)
                        .foregroundColor(.blue)
                    }
                } else {
                    ProgressView("Fetching settings...")
                }
            }
            .navigationTitle("Reality Settings")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        viewModel.hideKeyboard()
                        // Sheet dismiss is handled by the binding
                    }
                }
            }
        }
    }
}

private struct LensView: View {
    var body: some View {
        ZStack {
            Color.black
            VStack {
                Image(systemName: "camera.fill")
                    .font(.system(size: 60))
                    .foregroundColor(.white.opacity(0.2))
                Text("Lens Active")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.4))
            }
        }
    }
}

private struct SearchView: View {
    @EnvironmentObject var viewModel: LensAppViewModel
    
    var body: some View {
        NavigationStack {
            VStack {
                TextField("Search reality memories...", text: $viewModel.searchText)
                    .textFieldStyle(.roundedBorder)
                    .padding()
                    .onSubmit {
                        Task { await viewModel.runSearch() }
                    }
                
                List(viewModel.searchHits) { hit in
                    VStack(alignment: .leading) {
                        Text(hit.summary ?? "Observation \(hit.id)")
                            .font(.headline)
                        Text(hit.created_at)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Reality Search")
        }
    }
}

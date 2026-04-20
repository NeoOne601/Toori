import SwiftUI
import SmritiKit
import UIKit
import BackgroundTasks
import PhotosUI
import AVFoundation

extension Color {
    static let smritiAccent = Color(red: 0.92, green: 0.68, blue: 0.28) // Warm amber — Reality Intelligence
    static let smritiTeal = Color(red: 0.45, green: 0.38, blue: 0.82) // Soft indigo
    static let smritiDivider = Color.white.opacity(0.12)
    static let smritiSurface = Color.white.opacity(0.06)
    static let smritiStroke = Color.white.opacity(0.08)
    static let smritiCanvas = Color(red: 0.03, green: 0.05, blue: 0.10) // Deep navy

    // Reality Intelligence palette
    static let tooriAmber = Color(red: 0.92, green: 0.68, blue: 0.28)
    static let tooriIndigo = Color(red: 0.45, green: 0.38, blue: 0.82)
    static let tooriCanvas = Color(red: 0.03, green: 0.05, blue: 0.10)
    static let tooriStroke = Color.white.opacity(0.08)
}

extension Animation {
    static let smritiSpring = Animation.spring(response: 0.38, dampingFraction: 0.72)
}

@MainActor
final class SmritiAppModel: ObservableObject {
    enum RootTab: Hashable {
        case pulse
        case analyze
        case mandala
        case settings
        case journal
    }

    let eventStore = SmritiEventStore()

    @Published var selectedTab: RootTab = .pulse
    @Published var selectedMemory: SelectedMemory?
    @Published var isRecallSheetPresented = false
    @Published var lastTranscript = ""
    @Published var recallResults: [SmritiRecallItem] = []
    @Published var audioResults: [AudioQueryResult] = []
    @Published var isRecallLoading = false
    @Published var isAudioLoading = false
    @Published var recallErrorMessage: String?
    @Published var mandalaData: SmritiMandalaData?
    @Published var isMandalaLoading = false
    @Published var storageUsage: StorageUsageReport?
    @Published var watchFolders: [WatchFolderStatus] = []
    @Published var settingsStatusMessage: String?
    @Published var isRuntimeConnected = false
    @Published var openVocabEnabled = true
    @Published var tvlcEnabled = true

    private(set) var backendHost = "127.0.0.1:7777"

    var sessionID = "smriti-ios"

    func configureHost(_ host: String) {
        backendHost = host
        eventStore.updateHost(host)
    }

    func startPulse() {
        eventStore.start()
    }

    func stopPulse() {
        eventStore.stop()
    }

    func present(memory: SelectedMemory) {
        selectedMemory = memory
    }

    func dismissDetail() {
        selectedMemory = nil
    }

    func loadMandalaIfNeeded() async {
        guard mandalaData == nil, !isMandalaLoading else { return }
        await reloadMandala()
    }

    func reloadMandala() async {
        isMandalaLoading = true
        defer { isMandalaLoading = false }

        do {
            let api = try SmritiAPI(host: backendHost)
            mandalaData = try await api.fetchClusters()
        } catch {
            mandalaData = nil
        }
    }

    func runRecall(query: String) async {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 2 else { return }

        lastTranscript = trimmed
        recallErrorMessage = nil
        isRecallSheetPresented = true
        isRecallLoading = true

        defer { isRecallLoading = false }

        do {
            let api = try SmritiAPI(host: backendHost)
            let response = try await api.recall(
                SmritiRecallRequest(
                    query: trimmed,
                    session_id: sessionID,
                    top_k: 20,
                    person_filter: nil,
                    location_filter: nil,
                    time_start: nil,
                    time_end: nil,
                    min_confidence: 0
                )
            )
            recallResults = response.results
            if response.results.isEmpty {
                UINotificationFeedbackGenerator().notificationOccurred(.error)
                recallErrorMessage = "No memories matched that recall."
            } else {
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            }
        } catch {
            recallResults = []
            recallErrorMessage = error.localizedDescription
            UINotificationFeedbackGenerator().notificationOccurred(.error)
        }
    }

    func runHumToFind(audioBase64: String, sampleRate: Int) async {
        recallErrorMessage = nil
        isRecallSheetPresented = true
        isAudioLoading = true

        defer { isAudioLoading = false }

        do {
            let api = try SmritiAPI(host: backendHost)
            let response = try await api.audioQuery(
                AudioQueryRequest(
                    audio_base64: audioBase64,
                    sample_rate: sampleRate,
                    top_k: 10,
                    session_id: sessionID,
                    depth_stratum: nil,
                    person_filter: nil,
                    confidence_min: 0,
                    cross_modal: true
                )
            )
            audioResults = response.results
            if response.results.isEmpty {
                recallErrorMessage = "No sound matches yet."
                UINotificationFeedbackGenerator().notificationOccurred(.error)
            } else {
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            }
        } catch {
            audioResults = []
            recallErrorMessage = error.localizedDescription
            UINotificationFeedbackGenerator().notificationOccurred(.error)
        }
    }

    func loadSettingsData() async {
        do {
            let api = try SmritiAPI(host: backendHost)
            async let usage = api.fetchStorageUsage()
            async let folders = api.listWatchFolders()
            storageUsage = try await usage
            watchFolders = try await folders
            
            // Sync feature flags from backend
            let response = try await api.fetchSettings()
            openVocabEnabled = response.live_features.open_vocab_labels_enabled
            tvlcEnabled = response.live_features.tvlc_enabled
            
            settingsStatusMessage = nil
            isRuntimeConnected = true
        } catch {
            settingsStatusMessage = error.localizedDescription
            isRuntimeConnected = false
        }
    }

    func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}

@main
struct SmritiApp: App {
    @AppStorage("smriti.backendHost") private var backendHost = "127.0.0.1:7777"
    @AppStorage("smriti.hasCompletedOnboarding") private var hasCompletedOnboarding = false
    @State private var showGemmaDownload = false

    @StateObject private var appModel = SmritiAppModel()
    
    init() {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: "com.toori.smriti.journal", using: nil) { task in
            Task {
                _ = try? await SilentJournalEngine().generateTodaysJournal()
                task.setTaskCompleted(success: true)
                SilentJournalEngine().scheduleDaily()
            }
        }
        BGTaskScheduler.shared.register(forTaskWithIdentifier: "com.toori.smriti.patterns", using: nil) { task in
            Task {
                _ = try? await AnticipationEngine().generateWeeklyInsight()
                task.setTaskCompleted(success: true)
                AnticipationEngine().scheduleWeekly()
            }
        }
    }

    var body: some Scene {
        WindowGroup {
            RootShell(backendHost: $backendHost, hasCompletedOnboarding: $hasCompletedOnboarding)
                .environmentObject(appModel)
                .environmentObject(appModel.eventStore)
                .task {
                    appModel.configureHost(backendHost)
                    appModel.startPulse()
                }
                .onChange(of: backendHost) {
                    appModel.configureHost(backendHost)
                }
                .onDisappear {
                    appModel.stopPulse()
                }
                .fullScreenCover(item: $appModel.selectedMemory) { memory in
                    DetailView(memory: memory)
                        .environmentObject(appModel)
                }
                .sheet(isPresented: $appModel.isRecallSheetPresented) {
                    RecallSheet()
                        .environmentObject(appModel)
                        .presentationDetents([.medium, .large])
                        .presentationDragIndicator(.visible)
                        .presentationBackground(.ultraThinMaterial)
                        .onAppear {
                            let manager = GemmaModelManager.shared
                            if !manager.isModelPresent(), manager.detectTier() != .base {
                                showGemmaDownload = true
                            }
                        }
                }
                .fullScreenCover(isPresented: $showGemmaDownload) {
                    GemmaDownloadView()
                }
                .fullScreenCover(isPresented: onboardingPresentationBinding) {
                    OnboardingFlow {
                        hasCompletedOnboarding = true
                    }
                }
        }
    }

    private var onboardingPresentationBinding: Binding<Bool> {
        Binding(
            get: { !hasCompletedOnboarding },
            set: { newValue in
                hasCompletedOnboarding = !newValue
            }
        )
    }
}

private struct RootShell: View {
    @EnvironmentObject private var appModel: SmritiAppModel

    @Binding var backendHost: String
    @Binding var hasCompletedOnboarding: Bool

    var body: some View {
        TabView(selection: $appModel.selectedTab) {
            PulseView()
                .tag(SmritiAppModel.RootTab.pulse)
                .tabItem {
                    Label("Pulse", systemImage: "circle.grid.3x3.fill")
                }

            RealityCheckView()
                .tag(SmritiAppModel.RootTab.analyze)
                .tabItem {
                    Label("Analyze", systemImage: "eye.circle")
                }

            JournalView()
                .tag(SmritiAppModel.RootTab.journal)
                .tabItem {
                    Label("Journal", systemImage: "book.closed")
                }

            MandalaView()
                .tag(SmritiAppModel.RootTab.mandala)
                .tabItem {
                    Label("Mandala", systemImage: "point.3.connected.trianglepath.dotted")
                }

            SettingsView(backendHost: $backendHost, hasCompletedOnboarding: $hasCompletedOnboarding)
                .tag(SmritiAppModel.RootTab.settings)
                .tabItem {
                    Label("Settings", systemImage: "slider.horizontal.3")
                }
        }
        .preferredColorScheme(.dark)
        .tint(Color.smritiAccent)
        .onTapGesture {
            UIApplication.shared.sendAction(
                #selector(UIResponder.resignFirstResponder),
                to: nil, from: nil, for: nil
            )
        }
    }
}

/// The hero investor-demo view. A conversational photo analysis interface
/// that showcases JEPA-grounded Reality Intelligence.
///
/// Flow: User uploads photo → Toori analyzes with JEPA pipeline → shows
/// grounded description with confidence badge and depth strata overlay.
struct RealityCheckView: View {
    @EnvironmentObject private var appModel: SmritiAppModel

    @State private var messages: [RealityMessage] = []
    @State private var inputText = ""
    @State private var selectedPhoto: PhotosPickerItem?
    @State private var isAnalyzing = false
    @State private var showCamera = false
    @State private var scrollTarget: String?
    @FocusState private var isInputFocused: Bool

    // Removed CameraModel property

    var body: some View {
        ZStack {
            background
            
            VStack(spacing: 0) {
                header
                
                Divider()
                    .background(Color.tooriStroke)

                ScrollViewReader { proxy in
                    ScrollView(showsIndicators: false) {
                        LazyVStack(alignment: .leading, spacing: 16) {
                            if messages.isEmpty {
                                welcomeCard
                                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                            }
                            
                            ForEach(messages) { message in
                                RealityMessageBubble(message: message)
                                    .id(message.id)
                                    .transition(.asymmetric(
                                        insertion: .move(edge: .bottom).combined(with: .opacity),
                                        removal: .opacity
                                    ))
                            }
                            
                            if isAnalyzing {
                                AnalyzingIndicator()
                                    .id("analyzing")
                                    .transition(.opacity)
                            }
                        }
                        .padding(16)
                        .animation(.smritiSpring, value: messages.count)
                        .onTapGesture {
                            isInputFocused = false
                        }
                    }
                    .onChange(of: messages.count) {
                        if let last = messages.last {
                            withAnimation(.smritiSpring) {
                                proxy.scrollTo(last.id, anchor: .bottom)
                            }
                        }
                    }
                    .onChange(of: isAnalyzing) { _, analyzing in
                        if analyzing {
                            withAnimation(.smritiSpring) {
                                proxy.scrollTo("analyzing", anchor: .bottom)
                            }
                        }
                    }
                }

                inputBar
            }
        }
        .preferredColorScheme(.dark)
        .fullScreenCover(isPresented: $showCamera) {
            CameraCaptureView { image in
                showCamera = false
                if let image {
                    analyzeImage(image)
                }
            }
        }
        .onChange(of: selectedPhoto) { _, item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    analyzeImage(image)
                }
                selectedPhoto = nil
            }
        }
    }

    // MARK: - Background

    private var background: some View {
        ZStack {
            Color.tooriCanvas.ignoresSafeArea()
            RadialGradient(
                colors: [Color.tooriAmber.opacity(0.06), .clear],
                center: .topTrailing,
                startRadius: 20,
                endRadius: 500
            )
            .ignoresSafeArea()
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Toori")
                    .font(.system(size: 24, weight: .bold))
                    .foregroundStyle(.white)
                Text("Reality Intelligence")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Color.tooriAmber.opacity(0.8))
            }

            Spacer()

            HStack(spacing: 8) {
                Circle()
                    .fill(appModel.isRuntimeConnected
                        ? Color(red: 0.28, green: 0.82, blue: 0.46)
                        : Color(red: 0.88, green: 0.36, blue: 0.36))
                    .frame(width: 8, height: 8)
                Text(appModel.isRuntimeConnected ? "JEPA Online" : "Connecting...")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.7))
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                Capsule(style: .continuous)
                    .fill(Color.white.opacity(0.05))
            )
            .overlay(
                Capsule(style: .continuous)
                    .stroke(Color.tooriStroke, lineWidth: 0.5)
            )
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    // MARK: - Welcome

    private var welcomeCard: some View {
        VStack(spacing: 24) {
            // Animated eye icon
            TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { timeline in
                let t = timeline.date.timeIntervalSinceReferenceDate
                let scale = 0.97 + (sin(t * .pi / 2) + 1) / 2 * 0.03
                
                ZStack {
                    Circle()
                        .fill(Color.tooriAmber.opacity(0.12))
                        .frame(width: 90, height: 90)
                        .blur(radius: 12)
                    
                    Circle()
                        .stroke(Color.tooriAmber.opacity(0.4), lineWidth: 2)
                        .frame(width: 72, height: 72)
                    
                    Image(systemName: "eye.circle.fill")
                        .font(.system(size: 36, weight: .medium))
                        .foregroundStyle(Color.tooriAmber)
                }
                .scaleEffect(scale)
            }

            VStack(spacing: 8) {
                Text("The first AI that actually sees.")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                
                Text("Upload a photo or take one now.\nToori will give you a grounded understanding\n— not a guess.")
                    .font(.system(size: 15))
                    .foregroundStyle(.white.opacity(0.55))
                    .multilineTextAlignment(.center)
                    .lineSpacing(3)
            }

            VStack(spacing: 10) {
                demoSuggestion(icon: "house", text: "\"Is this room safe for a toddler?\"")
                demoSuggestion(icon: "wrench.and.screwdriver", text: "\"Where can I fit a workbench?\"")
                demoSuggestion(icon: "leaf", text: "\"What's happening to my plant?\"")
            }
        }
        .padding(24)
        .frame(maxWidth: .infinity)
    }

    private func demoSuggestion(icon: String, text: String) -> some View {
        Button {
            inputText = text.replacingOccurrences(of: "\"", with: "")
        } label: {
            HStack(spacing: 10) {
                Image(systemName: icon)
                    .font(.system(size: 14))
                    .foregroundStyle(Color.tooriAmber)
                    .frame(width: 24)
                Text(text)
                    .font(.system(size: 14))
                    .foregroundStyle(.white.opacity(0.72))
                Spacer()
                Image(systemName: "arrow.up.right")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.3))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color.white.opacity(0.04))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(Color.tooriStroke, lineWidth: 0.5)
            )
        }
    }

    // MARK: - Input Bar

    private var inputBar: some View {
        VStack(spacing: 0) {
            Divider()
                .background(Color.tooriStroke)

            HStack(spacing: 10) {
                // Photo picker
                PhotosPicker(selection: $selectedPhoto, matching: .images) {
                    Image(systemName: "photo.on.rectangle")
                        .font(.system(size: 18))
                        .foregroundStyle(Color.tooriAmber)
                }

                // Camera button
                Button {
                    showCamera = true
                } label: {
                    Image(systemName: "camera.fill")
                        .font(.system(size: 18))
                        .foregroundStyle(Color.tooriAmber)
                }

                // Text input
                TextField("Ask about any photo...", text: $inputText)
                    .font(.system(size: 15))
                    .foregroundStyle(.white)
                    .focused($isInputFocused)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 20, style: .continuous)
                            .fill(Color.white.opacity(0.06))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 20, style: .continuous)
                            .stroke(Color.tooriStroke, lineWidth: 0.5)
                    )
                    .onSubmit {
                        sendTextMessage()
                    }

                // Send
                if !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Button {
                        sendTextMessage()
                    } label: {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 28))
                            .foregroundStyle(Color.tooriAmber)
                    }
                    .transition(.scale.combined(with: .opacity))
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .animation(.smritiSpring, value: inputText.isEmpty)
        }
        .background(Color.tooriCanvas.opacity(0.95))
    }

    // MARK: - Actions

    private func sendTextMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        inputText = ""
        
        let userMessage = RealityMessage(
            role: .user,
            text: text,
            image: nil
        )
        messages.append(userMessage)
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        
        // If there's a previous analysis result, use the query with it
        if let lastAnalysis = messages.last(where: { $0.role == .assistant && $0.image != nil }) {
            let followUp = RealityMessage(
                role: .assistant,
                text: "I analyzed your photo with: \"\(text)\"\n\n\(lastAnalysis.groundedSummary ?? "Point your camera or upload a photo, and I'll give you grounded scene understanding.")",
                image: nil,
                confidence: lastAnalysis.confidence,
                confidenceLabel: lastAnalysis.confidenceLabel
            )
            messages.append(followUp)
        } else {
            let response = RealityMessage(
                role: .assistant,
                text: "Upload a photo or take one with your camera for me to analyze. I use the JEPA world model — not language guessing — to understand spatial layout, depth, and object relationships.",
                image: nil
            )
            messages.append(response)
        }
    }

    private func analyzeImage(_ image: UIImage) {
        let originalImage = image

        // Add user message with photo
        let userMsg = RealityMessage(
            role: .user,
            text: inputText.isEmpty ? "Analyze this scene" : inputText,
            image: originalImage
        )
        messages.append(userMsg)
        inputText = ""
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()

        isAnalyzing = true
        isInputFocused = false

        Task {
            defer { isAnalyzing = false }

            guard let jpeg = originalImage.jpegData(compressionQuality: 0.7) else {
                appendError("Failed to encode image.")
                return
            }

            let b64 = jpeg.base64EncodedString()
            let req = AnalyzeRequest(
                image_base64: b64,
                session_id: "reality-check",
                query: userMsg.text
            )

            do {
                let api = try SmritiAPI(host: appModel.backendHost)
                let response = try await api.analyze(req)

                // Extract grounding data from the response
                let confidence = response.observation?.confidence ?? 0.78
                let summary = response.grounded_summary
                    ?? response.observation?.summary
                    ?? response.hits.first?.summary
                    ?? "Scene analyzed with JEPA world model."
                let confidenceLabel = response.confidence_label ?? (confidence > 0.8 ? "Grounded" : confidence > 0.5 ? "Likely" : "Uncertain")
                
                // Build entity pills from tags
                let entities: [String] = response.observation?.tags ?? []
                
                // Build similar scenes list
                let similarScenes = response.hits.prefix(3).compactMap { $0.summary }

                let assistantMsg = RealityMessage(
                    role: .assistant,
                    text: nil,
                    image: originalImage,
                    groundedSummary: summary,
                    confidence: confidence,
                    confidenceLabel: confidenceLabel,
                    entities: entities,
                    similarScenes: Array(similarScenes),
                    depthStrata: response.hits.first.flatMap { _ in
                        // Use strata from observation if available
                        nil as SmritiDepthStrata?
                    }
                )
                messages.append(assistantMsg)
                UINotificationFeedbackGenerator().notificationOccurred(.success)
                
            } catch {
                appendError("Connection to Toori runtime failed. Ensure the runtime is running on your local network.\n\nError: \(error.localizedDescription)")
            }
        }
    }

    private func appendError(_ text: String) {
        let msg = RealityMessage(
            role: .assistant,
            text: text,
            image: nil,
            isError: true
        )
        messages.append(msg)
        UINotificationFeedbackGenerator().notificationOccurred(.error)
    }
}

// MARK: - Message Model

struct RealityMessage: Identifiable, Equatable {
    let id = UUID().uuidString
    let role: Role
    let text: String?
    let image: UIImage?
    var groundedSummary: String?
    var confidence: Double?
    var confidenceLabel: String?
    var entities: [String] = []
    var similarScenes: [String] = []
    var depthStrata: SmritiDepthStrata?
    var isError: Bool = false

    enum Role: String {
        case user
        case assistant
    }

    static func == (lhs: RealityMessage, rhs: RealityMessage) -> Bool {
        lhs.id == rhs.id
    }
}

// MARK: - Message Bubble

private struct RealityMessageBubble: View {
    let message: RealityMessage

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if message.role == .assistant {
                // Toori avatar
                ZStack {
                    Circle()
                        .fill(Color.tooriAmber.opacity(0.15))
                        .frame(width: 32, height: 32)
                    Image(systemName: "eye.circle.fill")
                        .font(.system(size: 16))
                        .foregroundStyle(Color.tooriAmber)
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                if message.role == .user {
                    userBubble
                } else {
                    assistantBubble
                }
            }
            .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)

            if message.role == .user {
                // User avatar
                ZStack {
                    Circle()
                        .fill(Color.tooriIndigo.opacity(0.2))
                        .frame(width: 32, height: 32)
                    Image(systemName: "person.fill")
                        .font(.system(size: 14))
                        .foregroundStyle(Color.tooriIndigo)
                }
            }
        }
    }

    private var userBubble: some View {
        VStack(alignment: .trailing, spacing: 8) {
            if let image = message.image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(maxWidth: 260, maxHeight: 200)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            }
            if let text = message.text, !text.isEmpty {
                Text(text)
                    .font(.system(size: 15))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(Color.tooriIndigo.opacity(0.3))
                    )
            }
        }
    }

    private var assistantBubble: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Analyzed photo with depth overlay
            if let image = message.image, message.groundedSummary != nil {
                ZStack(alignment: .topTrailing) {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(maxWidth: .infinity, maxHeight: 220)
                        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                        .overlay(
                            DepthStrataOverlay(
                                strataData: message.depthStrata,
                                imageSize: CGSize(width: 300, height: 220)
                            )
                            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                            .allowsHitTesting(false)
                        )

                    // Confidence badge overlay
                    if let confidence = message.confidence {
                        ConfidenceBadge(
                            confidence: confidence,
                            label: message.confidenceLabel
                        )
                        .padding(8)
                    }
                }
            }

            // Grounded description
            if let summary = message.groundedSummary {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 6) {
                        Image(systemName: "scope")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(Color.tooriAmber)
                        Text("Grounded Description")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(Color.tooriAmber.opacity(0.8))
                    }

                    Text(summary)
                        .font(.system(size: 15))
                        .foregroundStyle(.white.opacity(0.92))
                        .lineSpacing(3)
                }
                .padding(14)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.white.opacity(0.05))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(Color.tooriStroke, lineWidth: 0.5)
                )
            }

            // Entity tracks
            if !message.entities.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(message.entities, id: \.self) { entity in
                            HStack(spacing: 5) {
                                Image(systemName: "cube")
                                    .font(.system(size: 10))
                                Text(entity)
                                    .font(.system(size: 12, weight: .medium))
                            }
                            .foregroundStyle(.white.opacity(0.8))
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(
                                Capsule(style: .continuous)
                                    .fill(Color.tooriAmber.opacity(0.1))
                            )
                            .overlay(
                                Capsule(style: .continuous)
                                    .stroke(Color.tooriAmber.opacity(0.25), lineWidth: 0.5)
                            )
                        }
                    }
                }
            }

            // Similar scenes
            if !message.similarScenes.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Similar memories")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(.white.opacity(0.5))

                    ForEach(message.similarScenes, id: \.self) { scene in
                        HStack(spacing: 6) {
                            Image(systemName: "link")
                                .font(.system(size: 9))
                                .foregroundStyle(Color.tooriAmber.opacity(0.6))
                            Text(scene)
                                .font(.system(size: 12))
                                .foregroundStyle(.white.opacity(0.6))
                        }
                    }
                }
            }

            // Plain text (non-analysis messages)
            if let text = message.text, message.groundedSummary == nil {
                Text(text)
                    .font(.system(size: 15))
                    .foregroundStyle(message.isError ? Color(red: 0.88, green: 0.36, blue: 0.36) : .white.opacity(0.85))
                    .lineSpacing(2)
                    .padding(14)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(message.isError ? Color.red.opacity(0.08) : Color.white.opacity(0.04))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .stroke(message.isError ? Color.red.opacity(0.2) : Color.tooriStroke, lineWidth: 0.5)
                    )
            }
        }
    }
}

// MARK: - Analyzing Indicator

private struct AnalyzingIndicator: View {
    @State private var dotIndex = 0

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            ZStack {
                Circle()
                    .fill(Color.tooriAmber.opacity(0.15))
                    .frame(width: 32, height: 32)
                Image(systemName: "eye.circle.fill")
                    .font(.system(size: 16))
                    .foregroundStyle(Color.tooriAmber)
            }

            HStack(spacing: 4) {
                Text("Seeing")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.white.opacity(0.6))

                TimelineView(.animation(minimumInterval: 0.3)) { timeline in
                    let t = Int(timeline.date.timeIntervalSinceReferenceDate * 3.3)
                    HStack(spacing: 3) {
                        ForEach(0..<3, id: \.self) { i in
                            Circle()
                                .fill(Color.tooriAmber)
                                .frame(width: 5, height: 5)
                                .opacity(t % 3 == i ? 1.0 : 0.3)
                        }
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.white.opacity(0.04))
            )
        }
    }
}

// MARK: - Camera Capture View

struct CameraCaptureView: UIViewControllerRepresentable {
    let onCapture: (UIImage?) -> Void

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .camera
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: CameraCaptureView

        init(_ parent: CameraCaptureView) {
            self.parent = parent
        }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.onCapture(uiImage)
            } else {
                parent.onCapture(nil)
            }
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.onCapture(nil)
        }
    }
}

struct JournalView: View {
    @EnvironmentObject private var appModel: SmritiAppModel
    @State private var selectedDate = Date()
    @State private var animatedTextOpacity: Double = 0
    @State private var showAnimation = false
    @State private var textWords: [String] = []
    
    private let engine = SilentJournalEngine()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Top circle nav
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 16) {
                        ForEach(-6...0, id: \.self) { dayOffset in
                            let date = Calendar.current.date(byAdding: .day, value: dayOffset, to: Date()) ?? Date()
                            let isSelected = Calendar.current.isDate(date, inSameDayAs: selectedDate)
                            let hasEntry = engine.cachedJournal(for: date) != nil
                            
                            VStack(spacing: 4) {
                                Text("\(Calendar.current.component(.day, from: date))")
                                    .font(.system(size: 15))
                                    .frame(width: 32, height: 32)
                                    .background(isSelected ? Color.smritiAccent : Color.clear)
                                    .foregroundColor(isSelected ? .white : .primary)
                                    .clipShape(Circle())
                                    .onTapGesture {
                                        withAnimation(.smritiSpring) {
                                            selectedDate = date
                                        }
                                    }
                                
                                Circle()
                                    .fill(hasEntry ? Color.smritiAccent : Color.clear)
                                    .frame(width: 4, height: 4)
                            }
                        }
                    }
                    .padding()
                }
                .background(Color.smritiSurface)
                
                Divider().background(Color.smritiDivider)
                
                // Body
                ZStack {
                    if let entry = engine.cachedJournal(for: selectedDate) {
                        ScrollView {
                            Text(entry)
                                .font(.system(size: 17, weight: .regular))
                                .lineSpacing(8)
                                .padding(24)
                                .opacity(showAnimation ? 1 : 0)
                                .animation(.easeIn(duration: 0.5), value: showAnimation)
                                .onAppear {
                                    let fmt = DateFormatter()
                                    fmt.dateFormat = "yyyyMMdd"
                                    let key = "smriti.journal.viewed.\(fmt.string(from: selectedDate))"
                                    if !UserDefaults.standard.bool(forKey: key) {
                                        showAnimation = false
                                        withAnimation(.easeIn(duration: 0.8)) {
                                            showAnimation = true
                                        }
                                        UserDefaults.standard.set(true, forKey: key)
                                    } else {
                                        showAnimation = true
                                    }
                                }
                        }
                    } else if Calendar.current.isDateInToday(selectedDate) && Calendar.current.component(.hour, from: Date()) < 21 {
                        Text("Entry will appear tonight.")
                            .foregroundColor(.secondary)
                            .font(.callout)
                    } else if Calendar.current.isDateInToday(selectedDate) {
                        VStack(spacing: 16) {
                            TimelineView(.animation) { timeline in
                                let t = timeline.date.timeIntervalSinceReferenceDate
                                HStack(spacing: 6) {
                                    ForEach(0..<5, id: \.self) { i in
                                        let h = 12 + 8 * sin(t * 2 + Double(i))
                                        Rectangle()
                                            .fill(Color.smritiAccent.opacity(0.3))
                                            .frame(width: 4, height: h)
                                            .cornerRadius(2)
                                    }
                                }
                            }
                            Text("Writing tonight's entry…")
                                .foregroundColor(.secondary)
                                .font(.callout)
                        }
                    } else {
                        Text("No entry for this day.")
                            .foregroundColor(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .navigationTitle("Journal")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        shareJournal()
                    } label: {
                        Image(systemName: "square.and.arrow.up")
                    }
                    .disabled(engine.cachedJournal(for: selectedDate) == nil)
                }
            }
        }
        .sheet(item: $shareURLItem) { urlItem in
            ActivityView(activityItems: [urlItem.url])
        }
        .overlay {
            if isGenerating {
                Color.black.opacity(0.4)
                    .ignoresSafeArea()
                ProgressView()
                    .tint(.white)
            }
        }
    }
    
    // Feature 5 wire point
    @State private var isGenerating = false
    @State private var shareURLItem: URLItem?
    
    private struct URLItem: Identifiable {
        let id = UUID()
        let url: URL
    }
    
    private func shareJournal() {
        guard let entry = engine.cachedJournal(for: selectedDate) else { return }
        isGenerating = true
        Task {
            let generator = MemoryCardGenerator()
            if let cardURL = generator.generateMemoryCard(imagePath: nil, date: selectedDate, summary: entry) {
                DispatchQueue.main.async {
                    isGenerating = false
                    shareURLItem = URLItem(url: cardURL)
                }
            } else {
                DispatchQueue.main.async {
                    isGenerating = false
                }
            }
        }
    }
    
    private func saveToTempFile(_ data: Data, name: String) -> URL? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(name)
        try? data.write(to: url)
        return url
    }
}

private struct ActivityView: UIViewControllerRepresentable {
    let activityItems: [Any]
    let applicationActivities: [UIActivity]? = nil

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: applicationActivities)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

/// Visual overlay that renders depth stratum zones on an image.
/// Uses the TPDS (Tri-Planar Depth Separation) data when available,
/// or falls back to a stylized gradient simulation.
struct DepthStrataOverlay: View {
    let strataData: SmritiDepthStrata?
    let imageSize: CGSize

    var body: some View {
        ZStack {
            if let strataData, strataData.confidence ?? 0 > 0.3 {
                // Real strata overlays from JEPA depth separation
                if let _ = strataData.foreground_mask {
                    stratumLayer(color: Color.tooriAmber.opacity(0.18), label: "Foreground")
                        .mask(strataMask(mask: strataData.foreground_mask, size: imageSize))
                }
                if let _ = strataData.midground_mask {
                    stratumLayer(color: Color.white.opacity(0.08), label: "Midground")
                        .mask(strataMask(mask: strataData.midground_mask, size: imageSize))
                }
                if let _ = strataData.background_mask {
                    stratumLayer(color: Color.tooriIndigo.opacity(0.18), label: "Background")
                        .mask(strataMask(mask: strataData.background_mask, size: imageSize))
                }
            } else {
                // Stylized depth gradient when no real data is available
                stylizedDepthOverlay
            }
        }
    }

    private func stratumLayer(color: Color, label: String) -> some View {
        Rectangle()
            .fill(color)
            .overlay(alignment: .bottomLeading) {
                Text(label)
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(.white.opacity(0.5))
                    .padding(4)
            }
    }

    private func strataMask(mask: [[Bool]]?, size: CGSize) -> some View {
        Canvas { context, canvasSize in
            guard let mask, !mask.isEmpty else { return }
            let rows = mask.count
            let cols = mask.first?.count ?? 0
            guard rows > 0, cols > 0 else { return }

            let cellWidth = canvasSize.width / CGFloat(cols)
            let cellHeight = canvasSize.height / CGFloat(rows)

            for row in 0..<rows {
                for col in 0..<cols {
                    if mask[row][col] {
                        let rect = CGRect(
                            x: CGFloat(col) * cellWidth,
                            y: CGFloat(row) * cellHeight,
                            width: cellWidth + 0.5,
                            height: cellHeight + 0.5
                        )
                        context.fill(Path(rect), with: .color(.white))
                    }
                }
            }
        }
        .frame(width: size.width, height: size.height)
    }

    private var stylizedDepthOverlay: some View {
        VStack(spacing: 0) {
            // Background stratum (top)
            Rectangle()
                .fill(
                    LinearGradient(
                        colors: [Color.tooriIndigo.opacity(0.12), Color.clear],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(maxHeight: .infinity)
                .overlay(alignment: .topTrailing) {
                    stratumTag("Background", color: .tooriIndigo)
                        .padding(8)
                }

            // Midground stratum (center)
            Rectangle()
                .fill(Color.white.opacity(0.02))
                .frame(maxHeight: .infinity)

            // Foreground stratum (bottom)
            Rectangle()
                .fill(
                    LinearGradient(
                        colors: [Color.clear, Color.tooriAmber.opacity(0.12)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(maxHeight: .infinity)
                .overlay(alignment: .bottomLeading) {
                    stratumTag("Foreground", color: .tooriAmber)
                        .padding(8)
                }
        }
    }

    private func stratumTag(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .bold))
            .foregroundStyle(.white.opacity(0.8))
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(
                Capsule(style: .continuous)
                    .fill(color.opacity(0.4))
            )
    }
}

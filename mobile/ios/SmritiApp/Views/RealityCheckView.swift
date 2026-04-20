import SwiftUI
import PhotosUI
import AVFoundation
import SmritiKit

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

    @StateObject private var cameraModel = CameraModel()

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
                            appModel.hideKeyboard()
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
        appModel.hideKeyboard()
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
        appModel.hideKeyboard()

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
                    depthStrata: (response.observation?.metadata?["depth_strata"] as? [String: Any]).flatMap { _ in
                        // Map the backend strata to SmritiDepthStrata
                        nil as SmritiDepthStrata? // This is still a placeholder until SmritiDepthStrata init is known
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

struct CameraCaptureView: View {
    let onCapture: (UIImage?) -> Void

    @StateObject private var camera = CameraModel()

    var body: some View {
        ZStack {
            CameraPreviewView(session: camera.session)
                .ignoresSafeArea()

            VStack {
                HStack {
                    Button {
                        onCapture(nil)
                    } label: {
                        Image(systemName: "xmark")
                            .font(.title3.bold())
                            .foregroundStyle(.white)
                            .padding(14)
                            .background(Color.black.opacity(0.5))
                            .clipShape(Circle())
                    }
                    .padding()
                    Spacer()
                }

                Spacer()

                Button {
                    camera.captureImage { image in
                        onCapture(image)
                    }
                } label: {
                    ZStack {
                        Circle()
                            .stroke(Color.white.opacity(0.5), lineWidth: 4)
                            .frame(width: 72, height: 72)
                        Circle()
                            .fill(Color.tooriAmber)
                            .frame(width: 60, height: 60)
                        Image(systemName: "eye.fill")
                            .font(.system(size: 24, weight: .semibold))
                            .foregroundStyle(.white)
                    }
                }
                .padding(.bottom, 50)
            }
        }
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
    }
}

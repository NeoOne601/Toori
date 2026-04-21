import SwiftUI
import PhotosUI
import AVFoundation

// MARK: - RealityCheckView
/// The hero Reality Intelligence interface for Toori Lens.
///
/// Architecture:
///   User photo / camera → analyzeImage() → TooriAPIClient.analyze()
///   → JEPA pipeline on iMac backend → grounded_summary + ECGD confidence
///   → displayed with DepthStrataOverlay + ConfidenceBadge
///
/// Wired entirely to LensAppViewModel — zero SmritiKit dependency.
struct RealityCheckView: View {
    @EnvironmentObject private var vm: LensAppViewModel

    @State private var messages:       [RealityMessage] = []
    @State private var inputText       = ""
    @State private var selectedPhoto:  PhotosPickerItem?
    @State private var isAnalyzing     = false
    @State private var showCamera      = false
    @FocusState private var inputFocused: Bool

    var body: some View {
        ZStack {
            background

            VStack(spacing: 0) {
                header
                Divider().background(Color.tooriStroke)
                messageList
                inputBar
            }
        }
        .preferredColorScheme(.dark)
        .fullScreenCover(isPresented: $showCamera) {
            LensCameraCaptureView { image in
                showCamera = false
                if let image { analyzeImage(image) }
            }
        }
        .onChange(of: selectedPhoto) { item in
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
                colors: [Color.tooriAmber.opacity(0.05), .clear],
                center: .topTrailing,
                startRadius: 20, endRadius: 520
            ).ignoresSafeArea()
            RadialGradient(
                colors: [Color.tooriIndigo.opacity(0.04), .clear],
                center: .bottomLeading,
                startRadius: 20, endRadius: 400
            ).ignoresSafeArea()
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Toori")
                    .font(.system(size: 24, weight: .black))
                    .foregroundStyle(.white)
                Text("Reality Intelligence")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Color.tooriAmber.opacity(0.85))
                    .tracking(0.5)
            }

            Spacer()

            runtimeStatusPill
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 14)
    }

    private var runtimeStatusPill: some View {
        HStack(spacing: 7) {
            // Animated pulse dot
            TimelineView(.animation(minimumInterval: 1.0)) { tl in
                let connected = vm.isRuntimeConnected
                Circle()
                    .fill(connected ? Color.tooriGrounded : Color.tooriUncertain)
                    .frame(width: 7, height: 7)
                    .shadow(color: (connected ? Color.tooriGrounded : Color.tooriUncertain).opacity(0.8),
                            radius: connected ? 4 : 2)
            }
            Text(vm.isRuntimeConnected ? "JEPA Online" : vm.status)
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(.white.opacity(0.7))
                .lineLimit(1)
        }
        .padding(.horizontal, 11)
        .padding(.vertical, 7)
        .background(
            Capsule(style: .continuous)
                .fill(Color.white.opacity(0.04))
        )
        .overlay(
            Capsule(style: .continuous)
                .stroke(Color.tooriStroke, lineWidth: 0.6)
        )
    }

    // MARK: - Message List

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView(showsIndicators: false) {
                LazyVStack(alignment: .leading, spacing: 18) {
                    if messages.isEmpty {
                        welcomeCard
                            .transition(.opacity.combined(with: .move(edge: .bottom)))
                    }

                    ForEach(messages) { msg in
                        RealityMessageBubble(message: msg)
                            .id(msg.id)
                            .transition(.asymmetric(
                                insertion: .move(edge: .bottom).combined(with: .opacity),
                                removal:   .opacity
                            ))
                    }

                    if isAnalyzing {
                        AnalyzingIndicator()
                            .id("analyzing")
                            .transition(.opacity)
                    }
                }
                .padding(16)
                .animation(.tooriSpring, value: messages.count)
                .onTapGesture { inputFocused = false }
            }
            .onChange(of: messages.count) { _ in
                if let last = messages.last {
                    withAnimation(.tooriSpring) { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
            .onChange(of: isAnalyzing) { v in
                if v { withAnimation(.tooriSpring) { proxy.scrollTo("analyzing", anchor: .bottom) } }
            }
        }
    }

    // MARK: - Welcome Card

    private var welcomeCard: some View {
        VStack(spacing: 28) {
            // Animated eye — the Toori seeing icon
            TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { tl in
                let t = tl.date.timeIntervalSinceReferenceDate
                let pulse = 0.97 + (sin(t * .pi * 0.7) + 1) / 2 * 0.03

                ZStack {
                    Circle()
                        .fill(Color.tooriAmber.opacity(0.10))
                        .frame(width: 100, height: 100)
                        .blur(radius: 16)
                    Circle()
                        .stroke(Color.tooriAmber.opacity(0.35), lineWidth: 1.5)
                        .frame(width: 78, height: 78)
                    Image(systemName: "eye.circle.fill")
                        .font(.system(size: 38, weight: .medium))
                        .foregroundStyle(Color.tooriAmber)
                }
                .scaleEffect(pulse)
                .animation(.tooriReveal, value: pulse)
            }

            VStack(spacing: 10) {
                Text("The first AI that actually sees.")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)

                Text("Upload a photo or point your camera.\nToori returns grounded geometry — not a guess.")
                    .font(.system(size: 14))
                    .foregroundStyle(.white.opacity(0.5))
                    .multilineTextAlignment(.center)
                    .lineSpacing(4)
            }

            VStack(spacing: 10) {
                suggestionChip(icon: "house", label: "\"Is this room safe for a toddler?\"")
                suggestionChip(icon: "leaf",  label: "\"What's happening to my plant?\"")
                suggestionChip(icon: "eye",   label: "\"Describe my surroundings\"")
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 32)
        .frame(maxWidth: .infinity)
    }

    private func suggestionChip(icon: String, label: String) -> some View {
        Button {
            inputText = label.replacingOccurrences(of: "\"", with: "")
        } label: {
            HStack(spacing: 10) {
                Image(systemName: icon)
                    .font(.system(size: 13))
                    .foregroundStyle(Color.tooriAmber)
                    .frame(width: 22)
                Text(label)
                    .font(.system(size: 13))
                    .foregroundStyle(.white.opacity(0.7))
                Spacer()
                Image(systemName: "arrow.up.right")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.25))
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
            Divider().background(Color.tooriStroke)

            HStack(spacing: 10) {
                // Photo library picker
                PhotosPicker(selection: $selectedPhoto, matching: .images) {
                    Image(systemName: "photo.on.rectangle")
                        .font(.system(size: 19))
                        .foregroundStyle(Color.tooriAmber)
                }
                .id("photo_picker")

                // Live camera
                Button { showCamera = true } label: {
                    Image(systemName: "camera.fill")
                        .font(.system(size: 19))
                        .foregroundStyle(Color.tooriAmber)
                }
                .id("camera_button")

                // Text prompt
                TextField("Ask about any scene…", text: $inputText)
                    .font(.system(size: 15))
                    .foregroundStyle(.white)
                    .focused($inputFocused)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 22, style: .continuous)
                            .fill(Color.white.opacity(0.06))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 22, style: .continuous)
                            .stroke(inputFocused ? Color.tooriAmber.opacity(0.4) : Color.tooriStroke,
                                    lineWidth: 0.7)
                    )
                    .animation(.tooriSpring, value: inputFocused)
                    .onSubmit { sendTextMessage() }

                // Send button (appears when there's text)
                if !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Button { sendTextMessage() } label: {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 30))
                            .foregroundStyle(Color.tooriAmber)
                    }
                    .transition(.scale.combined(with: .opacity))
                    .id("send_button")
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .animation(.tooriSpring, value: inputText.isEmpty)
        }
        .background(Color.tooriCanvas.opacity(0.96))
    }

    // MARK: - Actions

    private func sendTextMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        inputText = ""
        inputFocused = false
        UIImpactFeedbackGenerator(style: .light).impactOccurred()

        messages.append(RealityMessage(role: .user, text: text, image: nil))

        // If a prior image was analyzed, use that result as context
        if let prev = messages.last(where: { $0.role == .assistant && $0.groundedSummary != nil }) {
            let reply = "Based on the scene I analyzed:\n\n\(prev.groundedSummary ?? "")."
            messages.append(RealityMessage(role: .assistant, text: reply, image: nil,
                                           confidence: prev.confidence, confidenceLabel: prev.confidenceLabel))
        } else {
            let reply = "Point your camera at something or upload a photo. I use JEPA world-model geometry — not guessing — to understand spatial layout, depth, and object relationships."
            messages.append(RealityMessage(role: .assistant, text: reply, image: nil))
        }
    }

    private func analyzeImage(_ image: UIImage) {
        let userMsg = RealityMessage(
            role: .user,
            text: inputText.isEmpty ? "Analyze this scene" : inputText,
            image: image
        )
        messages.append(userMsg)
        inputText  = ""
        inputFocused = false
        isAnalyzing = true
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()

        Task {
            defer { isAnalyzing = false }

            guard let jpeg = image.jpegData(compressionQuality: 0.75) else {
                appendError("Failed to encode the image.")
                return
            }

            do {
                let response = try await vm.api.analyze(
                    imageData: jpeg,
                    sessionId: vm.sessionId,
                    prompt: userMsg.text
                )

                let confidence     = response.observation.confidence
                let summary        = response.grounded_summary
                                   ?? response.observation.summary
                                   ?? response.hits.first?.summary
                                   ?? "Scene analyzed with JEPA world model."
                let confLabel      = response.confidence_label
                                   ?? (confidence > 0.8 ? "Grounded" : confidence > 0.5 ? "Likely" : "Uncertain")
                let entities       = response.observation.tags ?? []
                let similar        = response.hits.prefix(3).compactMap { $0.summary }

                let assistant = RealityMessage(
                    role:              .assistant,
                    text:              nil,
                    image:             image,
                    groundedSummary:   summary,
                    confidence:        confidence,
                    confidenceLabel:   confLabel,
                    entities:          entities,
                    similarScenes:     Array(similar)
                )
                messages.append(assistant)
                UINotificationFeedbackGenerator().notificationOccurred(.success)

            } catch {
                let ns = error as NSError
                if ns.domain == NSURLErrorDomain && ns.code == NSURLErrorNetworkConnectionLost {
                    appendError("iMac memory overflow — the backend is restarting. Try again in a moment.")
                } else if ns.domain == NSURLErrorDomain && ns.code == NSURLErrorTimedOut {
                    appendError("iMac is deep-thinking (swap thrash). Request timed out.")
                } else {
                    appendError("Couldn't reach the Toori runtime on your local network.\n\(error.localizedDescription)")
                }
            }
        }
    }

    private func appendError(_ text: String) {
        messages.append(RealityMessage(role: .assistant, text: text, image: nil, isError: true))
        UINotificationFeedbackGenerator().notificationOccurred(.error)
    }
}

// MARK: - RealityMessageBubble

private struct RealityMessageBubble: View {
    let message: RealityMessage

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if message.role == .assistant {
                tooriAvatar
            }

            VStack(alignment: .leading, spacing: 10) {
                if message.role == .user { userBubble } else { assistantBubble }
            }
            .frame(maxWidth: .infinity,
                   alignment: message.role == .user ? .trailing : .leading)

            if message.role == .user {
                userAvatar
            }
        }
    }

    // MARK: Avatars

    private var tooriAvatar: some View {
        ZStack {
            Circle().fill(Color.tooriAmber.opacity(0.13)).frame(width: 33, height: 33)
            Image(systemName: "eye.circle.fill")
                .font(.system(size: 17)).foregroundStyle(Color.tooriAmber)
        }
    }

    private var userAvatar: some View {
        ZStack {
            Circle().fill(Color.tooriIndigo.opacity(0.18)).frame(width: 33, height: 33)
            Image(systemName: "person.fill")
                .font(.system(size: 15)).foregroundStyle(Color.tooriIndigo)
        }
    }

    // MARK: User Bubble

    private var userBubble: some View {
        VStack(alignment: .trailing, spacing: 8) {
            if let img = message.image {
                Image(uiImage: img)
                    .resizable().scaledToFill()
                    .frame(maxWidth: 260, maxHeight: 200)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            }
            if let t = message.text, !t.isEmpty {
                Text(t)
                    .font(.system(size: 15))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14).padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(Color.tooriIndigo.opacity(0.28))
                    )
            }
        }
    }

    // MARK: Assistant Bubble

    private var assistantBubble: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Photo with TPDS depth overlay + ECGD badge
            if let img = message.image, message.groundedSummary != nil {
                ZStack(alignment: .topTrailing) {
                    Image(uiImage: img)
                        .resizable().scaledToFill()
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

                    if let conf = message.confidence {
                        ConfidenceBadge(confidence: conf, label: message.confidenceLabel)
                            .padding(8)
                    }
                }
            }

            // Grounded description card
            if let summary = message.groundedSummary {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 6) {
                        Image(systemName: "scope")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundStyle(Color.tooriAmber)
                        Text("Grounded Description")
                            .font(.system(size: 10, weight: .black))
                            .foregroundStyle(Color.tooriAmber.opacity(0.8))
                            .tracking(0.4)
                    }
                    Text(summary)
                        .font(.system(size: 15))
                        .foregroundStyle(.white.opacity(0.92))
                        .lineSpacing(3)
                }
                .padding(14)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.white.opacity(0.04))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(Color.tooriStroke, lineWidth: 0.5)
                )
            }

            // Entity pills (JEPA object tracks)
            if !message.entities.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 7) {
                        ForEach(message.entities, id: \.self) { entity in
                            HStack(spacing: 5) {
                                Image(systemName: "cube")
                                    .font(.system(size: 9))
                                Text(entity)
                                    .font(.system(size: 11, weight: .semibold))
                            }
                            .foregroundStyle(.white.opacity(0.8))
                            .padding(.horizontal, 10).padding(.vertical, 6)
                            .background(
                                Capsule(style: .continuous)
                                    .fill(Color.tooriAmber.opacity(0.09))
                            )
                            .overlay(
                                Capsule(style: .continuous)
                                    .stroke(Color.tooriAmber.opacity(0.22), lineWidth: 0.5)
                            )
                        }
                    }
                }
            }

            // Similar scenes (V-JEPA temporal continuity)
            if !message.similarScenes.isEmpty {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Similar memories")
                        .font(.system(size: 10, weight: .black))
                        .foregroundStyle(.white.opacity(0.4))
                    ForEach(message.similarScenes, id: \.self) { scene in
                        HStack(spacing: 6) {
                            Image(systemName: "link")
                                .font(.system(size: 9))
                                .foregroundStyle(Color.tooriAmber.opacity(0.55))
                            Text(scene)
                                .font(.system(size: 12))
                                .foregroundStyle(.white.opacity(0.58))
                        }
                    }
                }
            }

            // Plain text (non-analysis or error)
            if let text = message.text, message.groundedSummary == nil {
                Text(text)
                    .font(.system(size: 15))
                    .foregroundStyle(message.isError
                                     ? Color.tooriUncertain
                                     : .white.opacity(0.85))
                    .lineSpacing(2)
                    .padding(14)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(message.isError
                                  ? Color.red.opacity(0.07)
                                  : Color.white.opacity(0.03))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .stroke(message.isError
                                    ? Color.red.opacity(0.18)
                                    : Color.tooriStroke, lineWidth: 0.5)
                    )
            }
        }
    }
}

// MARK: - AnalyzingIndicator

private struct AnalyzingIndicator: View {
    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            ZStack {
                Circle().fill(Color.tooriAmber.opacity(0.13)).frame(width: 33, height: 33)
                Image(systemName: "eye.circle.fill")
                    .font(.system(size: 17)).foregroundStyle(Color.tooriAmber)
            }

            HStack(spacing: 6) {
                Text("Seeing")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.65))

                TimelineView(.animation(minimumInterval: 0.3)) { tl in
                    let t = Int(tl.date.timeIntervalSinceReferenceDate * 3.4)
                    HStack(spacing: 3) {
                        ForEach(0..<3, id: \.self) { i in
                            Circle()
                                .fill(Color.tooriAmber)
                                .frame(width: 5, height: 5)
                                .opacity(t % 3 == i ? 1.0 : 0.25)
                        }
                    }
                }
            }
            .padding(.horizontal, 14).padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.white.opacity(0.04))
            )
        }
    }
}

// MARK: - LensCameraCaptureView

/// Native UIImagePickerController wrapper for live camera capture.
struct LensCameraCaptureView: UIViewControllerRepresentable {
    let onCapture: (UIImage?) -> Void

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ vc: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: LensCameraCaptureView
        init(_ p: LensCameraCaptureView) { parent = p }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            parent.onCapture(info[.originalImage] as? UIImage)
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.onCapture(nil)
        }
    }
}

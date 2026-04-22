import SwiftUI
import PhotosUI
import AVFoundation
import Speech

// MARK: - RealityCheckView
/// The hero Reality Intelligence interface for Toori Lens.
///
/// Architecture:
///   User photo / camera → analyzeImage() → TooriAPIClient.analyze()
///   → JEPA pipeline on iMac backend → grounded_summary + ECGD confidence
///   → displayed with DepthStrataOverlay + ConfidenceBadge
///
/// Sprint additions (A1/A2/B1/D1):
///   - Persistent UUID session (survives backgrounding)
///   - Compare Mode — two-slot UI → POST /v1/analyze/compare
///   - Calibration Sheet — known-object scale → POST /v1/calibrate
///   - Multi-turn follow-up — text query on same session
///   - Domain badge auto-detection from entity labels
///   - Contextual suggestion chips after first analysis
///   - Long-press header → "New Conversation"
///
/// Wired entirely to LensAppViewModel — zero SmritiKit dependency.
struct RealityCheckView: View {
    @EnvironmentObject private var vm: LensAppViewModel

    @State private var messages:       [RealityMessage] = []
    @State private var inputText       = ""
    @State private var selectedPhoto:  PhotosPickerItem?
    @State private var selectedPhotoB: PhotosPickerItem?
    @State private var isAnalyzing     = false
    @State private var showCamera      = false
    @FocusState private var inputFocused: Bool

    // Voice input
    @StateObject private var voice = VoiceInputController()

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
        .sheet(isPresented: $vm.showCalibrationSheet) {
            CalibrationSheet()
                .environmentObject(vm)
        }
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
                    if vm.compareMode && vm.imageA == nil {
                        vm.imageA = image
                    } else if vm.compareMode && vm.imageA != nil {
                        vm.imageB = image
                    } else {
                        analyzeImage(image)
                    }
                }
                selectedPhoto = nil
            }
        }
        .onChange(of: selectedPhotoB) { item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    vm.imageB = image
                }
                selectedPhotoB = nil
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
                    .onLongPressGesture(minimumDuration: 0.6) {
                        withAnimation(.tooriSpring) {
                            messages.removeAll()
                            vm.newConversation()
                        }
                        UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
                    }
                Text("Reality Intelligence")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Color.tooriAmber.opacity(0.85))
                    .tracking(0.5)
            }

            Spacer()

            HStack(spacing: 10) {
                // Calibration button
                Button {
                    vm.showCalibrationSheet = true
                } label: {
                    ZStack {
                        Circle()
                            .fill(vm.isCalibrated ? Color.tooriGrounded.opacity(0.18) : Color.white.opacity(0.05))
                            .frame(width: 34, height: 34)
                        Image(systemName: vm.isCalibrated ? "ruler.fill" : "ruler")
                            .font(.system(size: 15))
                            .foregroundStyle(vm.isCalibrated ? Color.tooriGrounded : Color.white.opacity(0.5))
                    }
                }
                .id("calibration_button")

                runtimeStatusPill
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 14)
    }

    private var runtimeStatusPill: some View {
        HStack(spacing: 7) {
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
                    } else if let lastAnalysis = messages.last(where: { $0.role == .assistant && !$0.entities.isEmpty }) {
                        contextChips(for: lastAnalysis.entities)
                            .transition(.opacity.combined(with: .move(edge: .bottom)))
                    }

                    ForEach(messages) { msg in
                        RealityMessageBubble(message: msg, isCalibrated: vm.isCalibrated)
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
                suggestionChip(icon: "house",              label: "\"Is this room safe for a toddler?\"")
                suggestionChip(icon: "leaf",               label: "\"What's happening to my plant?\"")
                suggestionChip(icon: "arrow.left.arrow.right", label: "\"Compare two photos\"")
                suggestionChip(icon: "ruler",              label: "\"How wide is my desk?\"")
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 32)
        .frame(maxWidth: .infinity)
    }

    private func suggestionChip(icon: String, label: String) -> some View {
        Button {
            let cleaned = label.replacingOccurrences(of: "\"", with: "")
            if cleaned == "Compare two photos" {
                withAnimation(.tooriSpring) { vm.compareMode = true }
            } else {
                inputText = cleaned
            }
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

    // MARK: - Contextual chips (after first analysis)

    @ViewBuilder
    private func contextChips(for entities: [String]) -> some View {
        let chips = buildContextChips(entities: entities)
        if !chips.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(chips, id: \.label) { chip in
                        Button {
                            if chip.label == "Compare with another photo" {
                                withAnimation(.tooriSpring) { vm.compareMode = true }
                            } else {
                                inputText = chip.label
                            }
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: chip.icon)
                                    .font(.system(size: 11))
                                    .foregroundStyle(Color.tooriAmber)
                                Text(chip.label)
                                    .font(.system(size: 12, weight: .medium))
                                    .foregroundStyle(.white.opacity(0.75))
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(
                                Capsule(style: .continuous)
                                    .fill(Color.tooriAmber.opacity(0.08))
                            )
                            .overlay(
                                Capsule(style: .continuous)
                                    .stroke(Color.tooriAmber.opacity(0.2), lineWidth: 0.5)
                            )
                        }
                    }
                }
                .padding(.horizontal, 2)
            }
        }
    }

    private struct ContextChip { let icon: String; let label: String }

    private func buildContextChips(entities: [String]) -> [ContextChip] {
        var chips: [ContextChip] = []
        let lower = entities.map { $0.lowercased() }
        if lower.contains(where: { $0.contains("leaf") || $0.contains("stem") || $0.contains("plant") || $0.contains("flower") }) {
            chips.append(ContextChip(icon: "leaf", label: "Is this plant healthy?"))
        }
        if lower.contains(where: { $0.contains("outlet") || $0.contains("cord") || $0.contains("step") || $0.contains("stair") }) {
            chips.append(ContextChip(icon: "exclamationmark.shield", label: "Any safety concerns?"))
        }
        if lower.contains(where: { $0.contains("engine") || $0.contains("brake") || $0.contains("tire") || $0.contains("bolt") }) {
            chips.append(ContextChip(icon: "wrench.adjustable", label: "What needs attention?"))
        }
        if lower.contains(where: { $0.contains("lesion") || $0.contains("skin") || $0.contains("wound") }) {
            chips.append(ContextChip(icon: "stethoscope", label: "Describe any changes"))
        }
        chips.append(ContextChip(icon: "arrow.left.arrow.right", label: "Compare with another photo"))
        if !vm.isCalibrated {
            chips.append(ContextChip(icon: "ruler", label: "Measure this space"))
        }
        return Array(chips.prefix(5))
    }

    // MARK: - Input Bar

    private var inputBar: some View {
        VStack(spacing: 0) {
            Divider().background(Color.tooriStroke)

            if vm.compareMode {
                compareModeBar
            } else {
                standardInputBar
            }
        }
        .background(Color.tooriCanvas.opacity(0.96))
        .animation(.tooriSpring, value: vm.compareMode)
    }

    private var standardInputBar: some View {
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

            // Compare mode toggle
            Button {
                withAnimation(.tooriSpring) { vm.compareMode = true }
            } label: {
                Image(systemName: "arrow.left.arrow.right")
                    .font(.system(size: 17))
                    .foregroundStyle(Color.white.opacity(0.35))
            }
            .id("compare_toggle")

            // Mic button
            Button {
                if voice.isListening {
                    voice.stop { transcript in
                        if let t = transcript, !t.isEmpty { inputText = t }
                    }
                } else {
                    voice.start()
                }
            } label: {
                ZStack {
                    Circle()
                        .fill(voice.isListening ? Color.tooriAmber : Color.white.opacity(0.06))
                        .frame(width: 34, height: 34)
                    if voice.isListening {
                        LiveWaveBars(level: voice.audioLevel)
                            .frame(width: 20, height: 14)
                    } else {
                        Image(systemName: "mic.fill")
                            .font(.system(size: 14))
                            .foregroundStyle(Color.tooriAmber)
                    }
                }
            }
            .id("mic_button")

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

    // MARK: - Compare Mode Bar

    private var compareModeBar: some View {
        VStack(spacing: 10) {
            HStack(spacing: 12) {
                imageSlot("Before", image: $vm.imageA, picker: $selectedPhoto)

                VStack(spacing: 4) {
                    Image(systemName: "arrow.left.arrow.right")
                        .font(.system(size: 15))
                        .foregroundStyle(Color.tooriAmber)

                    Button("Compare") {
                        runCompare()
                    }
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(Color.tooriCanvas)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(
                        Capsule(style: .continuous)
                            .fill(vm.imageA != nil && vm.imageB != nil ? Color.tooriAmber : Color.white.opacity(0.2))
                    )
                    .disabled(vm.imageA == nil || vm.imageB == nil)
                    .id("compare_run_button")
                }

                imageSlot("After", image: $vm.imageB, picker: $selectedPhotoB)
            }

            HStack {
                TextField("Add a question (optional)…", text: $inputText)
                    .font(.system(size: 13))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(Color.white.opacity(0.05))
                    )

                Button {
                    withAnimation(.tooriSpring) {
                        vm.compareMode = false
                        vm.imageA = nil
                        vm.imageB = nil
                        inputText = ""
                    }
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 22))
                        .foregroundStyle(Color.white.opacity(0.35))
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
    }

    private func imageSlot(_ label: String, image: Binding<UIImage?>, picker: Binding<PhotosPickerItem?>) -> some View {
        PhotosPicker(selection: picker, matching: .images) {
            ZStack {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color.white.opacity(0.05))
                    .frame(width: 100, height: 80)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .stroke(image.wrappedValue != nil ? Color.tooriAmber.opacity(0.5) : Color.tooriStroke,
                                    style: StrokeStyle(lineWidth: 1, dash: [4]))
                    )

                if let img = image.wrappedValue {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 100, height: 80)
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                } else {
                    VStack(spacing: 4) {
                        Image(systemName: "plus")
                            .font(.system(size: 18))
                            .foregroundStyle(Color.tooriAmber.opacity(0.7))
                        Text(label)
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.4))
                    }
                }
            }
        }
    }

    // MARK: - Actions

    private func sendTextMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        inputText = ""
        inputFocused = false
        UIImpactFeedbackGenerator(style: .light).impactOccurred()

        messages.append(RealityMessage(role: .user, text: text, image: nil))

        // If last assistant message has a grounded context, try a real backend query
        if messages.contains(where: { $0.role == .assistant && $0.groundedSummary != nil }) {
            isAnalyzing = true
            Task {
                defer { isAnalyzing = false }
                if let response = await vm.sendTextQuery(text) {
                    let summary = response.grounded_summary
                                ?? response.observation.summary
                                ?? "Scene analyzed with JEPA world model."
                    let confLabel = response.confidence_label
                                 ?? (response.observation.confidence > 0.8 ? "Grounded"
                                     : response.observation.confidence > 0.5 ? "Likely" : "Uncertain")
                    let entities = response.observation.tags ?? []
                    let badge = domainBadge(for: entities)
                    let assistant = RealityMessage(
                        role: .assistant, text: nil, image: nil,
                        groundedSummary: summary,
                        confidence: response.observation.confidence,
                        confidenceLabel: confLabel,
                        entities: entities,
                        domainBadge: badge
                    )
                    messages.append(assistant)
                    haptic(for: confLabel)
                } else {
                    // Fallback echo from prior context
                    if let prev = messages.last(where: { $0.role == .assistant && $0.groundedSummary != nil }) {
                        let reply = "Based on the scene I analyzed:\n\n\(prev.groundedSummary ?? "")."
                        messages.append(RealityMessage(role: .assistant, text: reply, image: nil,
                                                       confidence: prev.confidence, confidenceLabel: prev.confidenceLabel))
                    } else {
                        messages.append(RealityMessage(role: .assistant,
                            text: "I need a photo to analyze first. Upload one or use the camera.", image: nil))
                    }
                }
            }
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

                let confidence  = response.observation.confidence
                let summary     = response.grounded_summary
                               ?? response.observation.summary
                               ?? response.hits.first?.summary
                               ?? "Scene analyzed with JEPA world model."
                let confLabel   = response.confidence_label
                               ?? (confidence > 0.8 ? "Grounded" : confidence > 0.5 ? "Likely" : "Uncertain")
                let entities    = response.observation.tags ?? []
                let similar     = response.hits.prefix(3).compactMap { $0.summary }
                let badge       = domainBadge(for: entities)

                let assistant = RealityMessage(
                    role:            .assistant,
                    text:            nil,
                    image:           image,
                    groundedSummary: summary,
                    confidence:      confidence,
                    confidenceLabel: confLabel,
                    entities:        entities,
                    similarScenes:   Array(similar),
                    domainBadge:     badge
                )
                messages.append(assistant)
                haptic(for: confLabel)

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

    private func runCompare() {
        let query = inputText.isEmpty ? nil : inputText
        inputText = ""
        inputFocused = false
        isAnalyzing = true
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()

        let imgA = vm.imageA
        let imgB = vm.imageB
        messages.append(RealityMessage(
            role: .user,
            text: query ?? "Compare these two photos",
            image: imgA
        ))

        Task {
            defer { isAnalyzing = false }
            guard let result = await vm.compareImages(query: query) else {
                appendError("Comparison failed. Make sure both photos are set and the backend is reachable.")
                return
            }
            var assistant = RealityMessage(
                role:            .assistant,
                text:            nil,
                image:           imgB,
                groundedSummary: result.grounded_diff,
                confidence:      Double(result.similarity_pct) / 100.0,
                confidenceLabel: result.confidence_label,
                compareResult:   result
            )
            messages.append(assistant)
            haptic(for: result.confidence_label)
            withAnimation(.tooriSpring) {
                vm.imageA = nil
                vm.imageB = nil
                vm.compareMode = false
            }
        }
    }

    private func appendError(_ text: String) {
        messages.append(RealityMessage(role: .assistant, text: text, image: nil, isError: true))
        UINotificationFeedbackGenerator().notificationOccurred(.error)
    }

    private func haptic(for confidenceLabel: String?) {
        switch confidenceLabel {
        case "Grounded": UINotificationFeedbackGenerator().notificationOccurred(.success)
        case "Likely":   UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        default:         UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
        }
    }

    private func domainBadge(for entities: [String]) -> String? {
        let lower = entities.map { $0.lowercased() }
        if lower.contains(where: { $0.contains("leaf") || $0.contains("stem") || $0.contains("plant") || $0.contains("flower") || $0.contains("root") }) {
            return "🌿 Plant"
        }
        if lower.contains(where: { $0.contains("outlet") || $0.contains("step") || $0.contains("stair") || $0.contains("sharp") }) {
            return "🏠 Safety"
        }
        if lower.contains(where: { $0.contains("engine") || $0.contains("brake") || $0.contains("tire") || $0.contains("bolt") || $0.contains("weld") }) {
            return "🔧 Mechanical"
        }
        if lower.contains(where: { $0.contains("lesion") || $0.contains("skin") || $0.contains("wound") || $0.contains("nail") }) {
            return "🩺 Medical"
        }
        if lower.contains(where: { $0.contains("drywall") || $0.contains("beam") || $0.contains("concrete") || $0.contains("tile") }) {
            return "🏗 Construction"
        }
        return nil
    }
}

// MARK: - RealityMessageBubble

private struct RealityMessageBubble: View {
    let message: RealityMessage
    let isCalibrated: Bool

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

            // Compare result (B1)
            if let compare = message.compareResult {
                CompareResultBubble(result: compare, imageA: nil, imageB: message.image)
            }

            // Photo with TPDS depth overlay + ECGD badge (when NOT compare result)
            else if let img = message.image, message.groundedSummary != nil {
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

                    // Domain badge (top-left)
                    if let badge = message.domainBadge {
                        Text(badge)
                            .font(.system(size: 10, weight: .bold))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 9)
                            .padding(.vertical, 5)
                            .background(Capsule(style: .continuous).fill(Color.tooriAmber.opacity(0.82)))
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    if let conf = message.confidence {
                        ConfidenceBadge(confidence: conf, label: message.confidenceLabel)
                            .padding(8)
                    }
                }
            }

            // Grounded description card
            if let summary = message.groundedSummary, message.compareResult == nil {
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

            // Entity pills — with metric sizes when calibrated
            if !message.entities.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 7) {
                        ForEach(message.entities, id: \.self) { entity in
                            let widthCm = message.metricEntities[entity]
                            HStack(spacing: 5) {
                                Image(systemName: isCalibrated ? "ruler" : "cube")
                                    .font(.system(size: 9))
                                if isCalibrated, let w = widthCm {
                                    Text("\(entity) · \(Int(w))cm")
                                        .font(.system(size: 11, weight: .semibold))
                                } else {
                                    Text(entity)
                                        .font(.system(size: 11, weight: .semibold))
                                }
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

            // Similar scenes
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

            // Plain text or error
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

// MARK: - CompareResultBubble (B1)

private struct CompareResultBubble: View {
    let result: CompareResponse
    let imageA: UIImage?
    let imageB: UIImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack(spacing: 6) {
                Image(systemName: "arrow.left.arrow.right")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(Color.tooriAmber)
                Text("Scene Comparison")
                    .font(.system(size: 10, weight: .black))
                    .foregroundStyle(Color.tooriAmber.opacity(0.8))
                    .tracking(0.4)
                Spacer()
                Text(result.change_summary)
                    .font(.system(size: 10))
                    .foregroundStyle(.white.opacity(0.45))
            }

            // Similarity bar
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .fill(Color.white.opacity(0.06))
                        .frame(height: 5)
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .fill(similarityColor(pct: result.similarity_pct))
                        .frame(width: geo.size.width * CGFloat(result.similarity_pct) / 100.0, height: 5)
                        .animation(.tooriReveal, value: result.similarity_pct)
                }
            }
            .frame(height: 5)

            HStack(spacing: 8) {
                Text("\(result.similarity_pct)% similar")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(similarityColor(pct: result.similarity_pct))
                Spacer()
                Text(result.confidence_label)
                    .font(.system(size: 10, weight: .black))
                    .foregroundStyle(confidenceColor(result.confidence_label))
            }

            // Grounded diff card
            if !result.grounded_diff.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("What Changed")
                        .font(.system(size: 10, weight: .black))
                        .foregroundStyle(.white.opacity(0.4))
                    Text(result.grounded_diff)
                        .font(.system(size: 14))
                        .foregroundStyle(.white.opacity(0.9))
                        .lineSpacing(3)
                }
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(Color.white.opacity(0.04))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .stroke(Color.tooriStroke, lineWidth: 0.5)
                )
            }

            // Changed region pills
            if !result.changed_regions.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(result.changed_regions) { region in
                            HStack(spacing: 4) {
                                if region.is_novel {
                                    Image(systemName: "sparkle")
                                        .font(.system(size: 8))
                                        .foregroundStyle(Color.tooriAmber)
                                }
                                Text(region.label)
                                    .font(.system(size: 10, weight: .semibold))
                                    .foregroundStyle(.white.opacity(0.75))
                            }
                            .padding(.horizontal, 9)
                            .padding(.vertical, 5)
                            .background(
                                Capsule(style: .continuous)
                                    .fill(region.is_novel ? Color.tooriAmber.opacity(0.15) : Color.white.opacity(0.06))
                            )
                        }
                    }
                }
            }
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

    private func similarityColor(pct: Int) -> Color {
        if pct >= 70 { return Color.tooriGrounded }
        if pct >= 40 { return Color.tooriAmber }
        return Color.tooriUncertain
    }

    private func confidenceColor(_ label: String) -> Color {
        switch label {
        case "Grounded": return Color.tooriGrounded
        case "Likely":   return Color.tooriAmber
        default:         return Color.tooriUncertain
        }
    }
}

// MARK: - CalibrationSheet (D1)

private struct CalibrationSheet: View {
    @EnvironmentObject private var vm: LensAppViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List(KnownObjectPreset.allCases) { preset in
                Button {
                    Task { await vm.calibrate(preset: preset) }
                } label: {
                    HStack {
                        Image(systemName: "ruler")
                            .foregroundStyle(Color.tooriAmber)
                        Text(preset.displayName)
                            .foregroundStyle(.primary)
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Calibrate Scale")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .safeAreaInset(edge: .bottom) {
                if let msg = vm.calibrationMessage {
                    Text(msg)
                        .font(.system(size: 13))
                        .foregroundStyle(Color.tooriGrounded)
                        .multilineTextAlignment(.center)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.tooriGrounded.opacity(0.08))
                }
            }
        }
        .preferredColorScheme(.dark)
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

// MARK: - LiveWaveBars (voice visualiser)

private struct LiveWaveBars: View {
    let level: Float

    var body: some View {
        HStack(spacing: 2) {
            ForEach(0..<5, id: \.self) { i in
                TimelineView(.animation(minimumInterval: 0.1)) { tl in
                    let t = tl.date.timeIntervalSinceReferenceDate
                    let h = max(4.0, CGFloat(level) * 14.0 * abs(sin(t * Double(i + 1) * 1.7 + Double(i))))
                    RoundedRectangle(cornerRadius: 2, style: .continuous)
                        .fill(Color.tooriCanvas)
                        .frame(width: 3, height: h)
                }
            }
        }
    }
}

// MARK: - VoiceInputController (SFSpeech, no SmritiKit dependency)

final class VoiceInputController: ObservableObject {
    @Published var isListening = false
    @Published var audioLevel: Float = 0

    private var recognizer: SFSpeechRecognizer?
    private var audioEngine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var lastTranscript: String?

    func start() {
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            guard status == .authorized else { return }
            DispatchQueue.main.async { self?._startListening() }
        }
    }

    func stop(completion: @escaping (String?) -> Void) {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        task?.finish()
        let captured = lastTranscript
        task = nil; request = nil; lastTranscript = nil
        DispatchQueue.main.async {
            self.isListening = false
            self.audioLevel  = 0
            completion(captured)
        }
    }

    private func _startListening() {
        guard !isListening else { return }
        recognizer = SFSpeechRecognizer(locale: .current)
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        request = req

        let node = audioEngine.inputNode
        let fmt  = node.outputFormat(forBus: 0)
        node.installTap(onBus: 0, bufferSize: 1024, format: fmt) { [weak self] buf, _ in
            self?.request?.append(buf)
            let samples = buf.floatChannelData?[0]
            let count = Int(buf.frameLength)
            if let s = samples {
                let rms = sqrt((0..<count).reduce(0.0) { $0 + Double(s[$1] * s[$1]) } / Double(count))
                DispatchQueue.main.async { self?.audioLevel = Float(min(1.0, rms * 20)) }
            }
        }

        try? audioEngine.start()
        isListening = true

        task = recognizer?.recognitionTask(with: req) { [weak self] result, _ in
            if let text = result?.bestTranscription.formattedString {
                self?.lastTranscript = text
            }
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

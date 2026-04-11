import SwiftUI
import UIKit

struct RecallSheet: View {
    @EnvironmentObject private var appModel: SmritiAppModel

    @State private var humRevealPayload: HumRevealPayload?
    @State private var humRevealImage: Image? = nil
    @State private var humRevealVisible: Bool = false
    @State private var humButtonFrame: CGRect = .zero
    @State private var lastHumRevealSignature: String?

    var body: some View {
        NavigationStack {
            ScrollView(showsIndicators: false) {
                VStack(alignment: .leading, spacing: 20) {
                    header

                    if appModel.isRecallLoading || appModel.isAudioLoading {
                        ProgressView()
                            .tint(Color.smritiAccent)
                            .frame(maxWidth: .infinity, alignment: .center)
                    }

                    if let recallErrorMessage = appModel.recallErrorMessage, !recallErrorMessage.isEmpty {
                        Text(recallErrorMessage)
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(.white.opacity(0.74))
                            .padding(14)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(
                                RoundedRectangle(cornerRadius: 16, style: .continuous)
                                    .fill(Color.white.opacity(0.06))
                            )
                    }

                    VStack(alignment: .leading, spacing: 12) {
                        Text("Voice recall")
                            .font(.system(size: 17, weight: .semibold))
                            .foregroundStyle(.white)

                        if appModel.recallResults.isEmpty, !appModel.isRecallLoading {
                            Text("Say a memory in plain language and Smriti will pull it into view.")
                                .font(.system(size: 13))
                                .foregroundStyle(.white.opacity(0.52))
                        } else {
                            ForEach(appModel.recallResults) { item in
                                RecallResultCard(
                                    thumbnailPath: item.thumbnail_path,
                                    title: item.primary_description,
                                    subtitle: item.displaySubtitle,
                                    surprise: item.surpriseProxy,
                                    matchedBySound: false
                                ) {
                                    appModel.present(memory: .recall(item))
                                }
                            }
                        }
                    }

                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Hum to find")
                                .font(.system(size: 17, weight: .semibold))
                                .foregroundStyle(.white)
                            Spacer()
                            HumToFindButton()
                                .background(
                                    GeometryReader { proxy in
                                        Color.clear.preference(
                                            key: HumButtonFramePreferenceKey.self,
                                            value: proxy.frame(in: .global)
                                        )
                                    }
                                )
                        }

                        if !appModel.audioResults.isEmpty {
                            ScrollView(.horizontal, showsIndicators: false) {
                                HStack(spacing: 12) {
                                    ForEach(appModel.audioResults) { result in
                                        RecallResultCard(
                                            thumbnailPath: result.thumbnail_path,
                                            title: result.primaryText,
                                            subtitle: "Cross-modal match • \(result.audio_score.formatted(.number.precision(.fractionLength(2))))",
                                            surprise: result.audio_score,
                                            matchedBySound: true,
                                            onTap: nil
                                        )
                                        .frame(width: 280)
                                    }
                                }
                                .padding(.vertical, 2)
                            }
                        } else if !appModel.isAudioLoading {
                            Text("Record three seconds of humming to search the visual memory space by sound.")
                                .font(.system(size: 13))
                                .foregroundStyle(.white.opacity(0.52))
                        }
                    }
                }
                .padding(20)
            }
            .background(Color.black.opacity(0.92))
            .toolbar {
                ToolbarItem(placement: .principal) {
                    Text("Recall")
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundStyle(.white)
                }
            }
        }
        .preferredColorScheme(.dark)
        .fullScreenCover(item: $humRevealPayload) { payload in
            HumRevealView(payload: payload) {
                humRevealPayload = nil
            }
        }
        .onPreferenceChange(HumButtonFramePreferenceKey.self) { frame in
            humButtonFrame = frame
        }
        .onAppear {
            lastHumRevealSignature = appModel.audioResults.first.map(\.id)
        }
        .onChange(of: appModel.audioResults, initial: false) { _, newResults in
            scheduleHumReveal(for: newResults)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("“\(appModel.lastTranscript)”")
                .font(.system(size: 22, weight: .semibold))
                .foregroundStyle(.white)
            Text("Results rise here as soon as the voice query settles.")
                .font(.system(size: 13))
                .foregroundStyle(.white.opacity(0.56))
        }
    }

    private func scheduleHumReveal(for results: [AudioQueryResult]) {
        guard let first = results.first else {
            lastHumRevealSignature = nil
            humRevealPayload = nil
            humRevealVisible = false
            humRevealImage = nil
            return
        }

        let signature = first.id
        guard signature != lastHumRevealSignature else { return }
        lastHumRevealSignature = signature

        if let thumbPath = first.thumbnail_path, !thumbPath.isEmpty,
           let uiImage = UIImage(contentsOfFile: thumbPath) {
            humRevealImage = Image(uiImage: uiImage)
        }
        humRevealVisible = true

        let startPoint: CGPoint
        if humButtonFrame == .zero {
            startPoint = CGPoint(
                x: UIScreen.main.bounds.width - 52,
                y: UIScreen.main.bounds.height - 92
            )
        } else {
            startPoint = CGPoint(
                x: humButtonFrame.midX,
                y: humButtonFrame.midY
            )
        }
        humRevealPayload = HumRevealPayload(result: first, startPoint: startPoint)
    }
}

private struct HumButtonFramePreferenceKey: PreferenceKey {
    static var defaultValue: CGRect = .zero

    static func reduce(value: inout CGRect, nextValue: () -> CGRect) {
        value = nextValue()
    }
}

private struct HumRevealPayload: Identifiable {
    let result: AudioQueryResult
    let startPoint: CGPoint

    var id: String { result.id }
}

private struct HumRevealView: View {
    let payload: HumRevealPayload
    var onDismiss: () -> Void

    @State private var progress: CGFloat = 0
    @State private var badgeScale: CGFloat = 0.6

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                Color.black.ignoresSafeArea()
                revealImage(in: proxy)

                VStack {
                    HStack {
                        Spacer()
                        badge
                    }
                    .padding(.top, 56)
                    .padding(.trailing, 20)
                    Spacer()
                }

                VStack {
                    Spacer()
                    if !payload.result.primaryText.isEmpty {
                        Text(payload.result.primaryText)
                            .font(.system(size: 22, weight: .semibold))
                            .foregroundStyle(.white)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 24)
                            .padding(.bottom, 36)
                    }
                }
            }
            .contentShape(Rectangle())
            .onTapGesture {
                dismiss()
            }
            .onAppear {
                UINotificationFeedbackGenerator().notificationOccurred(.success)
                withAnimation(.smritiSpring) {
                    progress = 1
                    badgeScale = 1
                }

                Task {
                    try? await Task.sleep(for: .seconds(2))
                    guard !Task.isCancelled else { return }
                    await MainActor.run {
                        dismiss()
                    }
                }
            }
        }
    }

    private func dismiss() {
        onDismiss()
    }

    @ViewBuilder
    private func revealImage(in proxy: GeometryProxy) -> some View {
        if let image = loadImage() {
            let baseDiameter: CGFloat = 56
            let maxScale = max(proxy.size.width, proxy.size.height) / baseDiameter * 1.9

            Image(uiImage: image)
                .resizable()
                .scaledToFill()
                .frame(width: proxy.size.width, height: proxy.size.height)
                .clipped()
                .blur(radius: (1 - progress) * 40)
                .mask(
                    Circle()
                        .frame(width: baseDiameter, height: baseDiameter)
                        .position(payload.startPoint)
                        .scaleEffect(1 + progress * maxScale)
                )
                .animation(.smritiSpring, value: progress)
        } else {
            ZStack {
                LinearGradient(
                    colors: [Color.smritiAccent.opacity(0.34), Color.smritiTeal.opacity(0.16)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                Image(systemName: "photo")
                    .font(.system(size: 54, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.82))
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
            .mask(
                Circle()
                    .frame(width: 56, height: 56)
                    .position(payload.startPoint)
                    .scaleEffect(1 + progress * 14)
            )
            .animation(.smritiSpring, value: progress)
        }
    }

    private var badge: some View {
        Text("matched by sound")
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                Capsule(style: .continuous)
                    .fill(Color.smritiAccent.opacity(0.24))
            )
            .overlay(
                Capsule(style: .continuous)
                    .stroke(Color.smritiStroke, lineWidth: 0.5)
            )
            .scaleEffect(badgeScale)
            .animation(.smritiSpring, value: badgeScale)
    }

    private func loadImage() -> UIImage? {
        guard let thumbnailPath = payload.result.thumbnail_path, !thumbnailPath.isEmpty else {
            return nil
        }
        return UIImage(contentsOfFile: thumbnailPath)
    }
}

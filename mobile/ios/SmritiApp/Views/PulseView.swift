import SwiftUI
import UIKit

struct PulseView: View {
    @EnvironmentObject private var appModel: SmritiAppModel
    @EnvironmentObject private var eventStore: SmritiEventStore

    private let columns = Array(repeating: GridItem(.fixed(108), spacing: 18), count: 3)

    @State private var bloomingIDs: Set<String> = []
    @State private var dimmingAll: Bool = false
    @State private var bloomCaption: String?
    @State private var bloomResetTask: Task<Void, Never>?
    @State private var lastBloomSignature: String?

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            background

            VStack(spacing: 28) {
                header

                if let bloomCaption {
                    bloomCaptionView(bloomCaption)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }

                LazyVGrid(columns: columns, spacing: 20) {
                    ForEach(Array(eventStore.orbSlots.enumerated()), id: \.element.id) { index, slot in
                        bloomAwareOrb(slot: slot, index: index)
                            .transition(
                                .asymmetric(
                                    insertion: .move(edge: .bottom).combined(with: .opacity),
                                    removal: .opacity
                                )
                            )
                    }
                }
                .animation(.smritiSpring, value: eventStore.orbSlots)

                footer
            }
            .padding(.horizontal, 24)
            .padding(.top, 18)
            .padding(.bottom, 32)

            VoiceRecallButton()
                .padding(.trailing, 24)
                .padding(.bottom, 28)
        }
        .ignoresSafeArea()
        .task {
            appModel.startPulse()
        }
        .onChange(of: appModel.recallResults, initial: false) { _, newResults in
            triggerMemoryBloom(for: newResults)
        }
    }

    private var background: some View {
        ZStack {
            Color.smritiCanvas
            RadialGradient(
                colors: [Color.smritiAccent.opacity(0.16), .clear],
                center: .topTrailing,
                startRadius: 20,
                endRadius: 420
            )
            RadialGradient(
                colors: [Color.smritiTeal.opacity(0.12), .clear],
                center: .bottomLeading,
                startRadius: 20,
                endRadius: 320
            )
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("The Pulse")
                        .font(.system(size: 28, weight: .semibold))
                        .foregroundStyle(.white)
                    Text("Live memories arriving from Smriti")
                        .font(.system(size: 13))
                        .foregroundStyle(.white.opacity(0.6))
                }

                Spacer()

                connectionBadge
            }

            Text("Tap an orb to open the moment. Long press to reveal its label.")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.white.opacity(0.45))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func bloomCaptionView(_ text: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "sparkles")
                .font(.system(size: 11, weight: .semibold))
            Text(text)
                .font(.system(size: 12, weight: .semibold))
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(
            Capsule(style: .continuous)
                .fill(Color.smritiAccent.opacity(0.18))
        )
        .overlay(
            Capsule(style: .continuous)
                .stroke(Color.smritiStroke, lineWidth: 0.5)
        )
    }

    private func bloomAwareOrb(slot: PulseOrbSlot, index: Int) -> some View {
        let isObservation = observation(for: slot) != nil
        let observationID = observation(for: slot)?.id
        let shouldBloom = observationID.map { bloomingIDs.contains($0) } ?? false
        let shouldDim = dimmingAll && !shouldBloom

        return OrbView(slot: slot, index: index) {
            if case .observation(let observation) = slot {
                appModel.present(memory: .observation(observation))
            }
        }
        .scaleEffect(shouldBloom ? 1.18 : 1.0)
        .opacity(shouldDim ? 0.3 : 1.0)
        .overlay {
            if shouldBloom, isObservation {
                Circle()
                    .stroke(Color.smritiAccent.opacity(0.92), lineWidth: 4)
                    .frame(width: 112, height: 112)
                    .shadow(color: Color.smritiAccent.opacity(0.3), radius: 12, y: 0)
                    .transition(.opacity.combined(with: .scale))
            }
        }
        .animation(.smritiSpring, value: bloomingIDs)
        .animation(.easeOut(duration: 0.25), value: dimmingAll)
    }

    private func observation(for slot: PulseOrbSlot) -> ObservationSummary? {
        if case .observation(let observation) = slot {
            return observation
        }
        return nil
    }

    private func triggerMemoryBloom(for results: [SmritiRecallItem]) {
        guard !results.isEmpty else {
            clearBloom()
            return
        }

        let meanSurprise = results.map(\.surpriseProxy).reduce(0, +) / Double(results.count)
        guard meanSurprise > 0.65 else {
            clearBloom()
            return
        }

        let signature = bloomSignature(for: results)
        guard signature != lastBloomSignature else { return }
        lastBloomSignature = signature

        let matches = matchedObservationIDs(for: results)
        guard !matches.isEmpty else {
            clearBloom()
            return
        }
        bloomResetTask?.cancel()

        UIImpactFeedbackGenerator(style: .medium).impactOccurred()

        withAnimation(.smritiSpring) {
            bloomingIDs = matches
            dimmingAll = true
        }

        withAnimation(.easeIn(duration: 0.2)) {
            bloomCaption = "⚡ \(results.count) surprising moments found"
        }

        bloomResetTask = Task {
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                withAnimation(.easeOut(duration: 0.2)) {
                    bloomingIDs = []
                    dimmingAll = false
                    bloomCaption = nil
                }
            }
        }
    }

    private func clearBloom() {
        bloomResetTask?.cancel()
        bloomResetTask = nil
        lastBloomSignature = nil
        withAnimation(.smritiSpring) {
            bloomingIDs = []
            dimmingAll = false
        }
        withAnimation(.easeOut(duration: 0.2)) {
            bloomCaption = nil
        }
    }

    private func bloomSignature(for results: [SmritiRecallItem]) -> String {
        let ids = results.map(\.id).joined(separator: "|")
        let mean = results.map(\.surpriseProxy).reduce(0, +) / Double(results.count)
        return "\(ids)#\(mean.formatted(.number.precision(.fractionLength(3))))"
    }

    private func matchedObservationIDs(for results: [SmritiRecallItem]) -> Set<String> {
        let normalizedResultPaths = Set(
            results.flatMap { result in
                [result.file_path, result.thumbnail_path]
                    .filter { !$0.isEmpty }
                    .map { normalizePath($0) }
            }
        )

        let normalizedResultLabels = Set(
            results.flatMap { result in
                [
                    result.primary_description,
                    result.displaySubtitle,
                    result.primarySetuText
                ]
                .filter { !$0.isEmpty }
                .map { normalizeText($0) }
            }
        )

        return Set(eventStore.observations.compactMap { observation in
            let pathMatches = [observation.image_path, observation.thumbnail_path]
                .map { normalizePath($0) }
                .contains { normalizedResultPaths.contains($0) }

            let textMatches = [
                observation.summary ?? "",
                observation.displayLabel,
                observation.source_query ?? ""
            ]
            .map { normalizeText($0) }
            .contains { normalizedResultLabels.contains($0) }

            return (pathMatches || textMatches) ? observation.id : nil
        })
    }

    private func normalizePath(_ path: String) -> String {
        guard !path.isEmpty else { return "" }
        return URL(fileURLWithPath: path).standardizedFileURL.path.lowercased()
    }

    private func normalizeText(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    private var connectionBadge: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(connectionColor)
                .frame(width: 8, height: 8)
            Text(connectionText)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.white.opacity(0.8))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(
            Capsule(style: .continuous)
                .fill(Color.white.opacity(0.05))
        )
        .overlay(
            Capsule(style: .continuous)
                .stroke(Color.smritiStroke, lineWidth: 0.5)
        )
    }

    private var footer: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("9 most recent observations")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.72))
                Text("New observations push upward with a haptic pulse.")
                    .font(.system(size: 11))
                    .foregroundStyle(.white.opacity(0.42))
            }

            Spacer()
        }
        .padding(.top, 4)
    }

    private var connectionColor: Color {
        switch eventStore.connectionState {
        case .live:
            return .green
        case .connecting:
            return Color.smritiTeal
        case .reconnecting:
            return .orange
        case .failed:
            return .red
        case .idle:
            return .white.opacity(0.35)
        }
    }

    private var connectionText: String {
        switch eventStore.connectionState {
        case .idle:
            return "Idle"
        case .connecting:
            return "Connecting"
        case .live:
            return "Live"
        case .reconnecting(let delay):
            return "Retrying in \(delay)s"
        case .failed:
            return "Offline"
        }
    }
}

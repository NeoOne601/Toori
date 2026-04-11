import AppKit
import QuartzCore
import SwiftUI

struct RecallView: View {
    @EnvironmentObject private var appModel: SmritiAppModel
    @State private var recallTask: Task<Void, Never>?
    @State private var dragTranslation: CGFloat = 0
    @StateObject private var surpriseCoordinator = SurpriseMomentCoordinator()
    @StateObject private var recallEngine = MultilingualRecallEngine()
    @State private var recallNarration: String?
    @State private var detectedLanguageCode: String?
    @State private var showJournal = false
    @State private var insightDismissed = false

    var body: some View {
        ZStack(alignment: .top) {
            SmritiGlassSurface {
                VStack(alignment: .leading, spacing: 18) {
                    Picker("Surface", selection: $appModel.selectedTab) {
                        ForEach(SmritiAppModel.RootTab.allCases) { tab in
                            Text(tab.rawValue).tag(tab)
                        }
                    }
                    .pickerStyle(.segmented)

                    ZStack {
                        if appModel.selectedTab == .recall {
                            recallSurface
                                .transition(
                                    .asymmetric(
                                        insertion: .move(edge: .leading).combined(with: .opacity),
                                        removal: .move(edge: .trailing).combined(with: .opacity)
                                    )
                                )
                        } else {
                            MandalaView()
                                .transition(
                                    .asymmetric(
                                        insertion: .move(edge: .trailing).combined(with: .opacity),
                                        removal: .move(edge: .leading).combined(with: .opacity)
                                    )
                                )
                        }
                    }
                    .animation(.smritiSpring, value: appModel.selectedTab)
                    .gesture(
                        DragGesture(minimumDistance: 20)
                            .onChanged { value in
                                dragTranslation = value.translation.width
                            }
                            .onEnded { value in
                                defer { dragTranslation = 0 }
                                if value.translation.width < -60 {
                                    appModel.selectedTab = .mandala
                                } else if value.translation.width > 60 {
                                    appModel.selectedTab = .recall
                                }
                            }
                    )
                }
            }

            if let banner = surpriseCoordinator.banner {
                SurpriseBannerView(imagePath: banner.thumbnailPath, label: banner.label)
                    .padding(.horizontal, 8)
                    .padding(.top, 8)
                    .zIndex(1)
                    .allowsHitTesting(false)
            }
        }
        .padding(20)
        .onChange(of: appModel.recallQuery) { _, _ in
            scheduleRecall()
        }
        .onChange(of: appModel.selectedTab) { _, newValue in
            if newValue == .mandala {
                Task {
                    await appModel.loadMandalaIfNeeded()
                }
            }
        }
        .onAppear {
            scheduleRecall()
            surpriseCoordinator.start(api: appModel.api)
        }
        .onDisappear {
            surpriseCoordinator.stop()
        }
    }

    private var recallSurface: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(Color.smritiAccent)
                TextField("Search your memory…", text: $appModel.recallQuery)
                    .textFieldStyle(.plain)
                    .font(.system(size: 17))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.white.opacity(0.06))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(Color.smritiGlassStroke, lineWidth: 0.5)
            )

            if let lang = detectedLanguageCode, lang != "en" {
                let displayLanguageName = Locale.current.localizedString(forLanguageCode: lang) ?? "English"
                Label(displayLanguageName, systemImage: "text.bubble")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.smritiAccent.opacity(0.12))
                    .cornerRadius(8)
            }

            Group {
                if appModel.recallQuery.trimmingCharacters(in: .whitespacesAndNewlines).count < 2 {
                    RecallEmptyState(message: "Your memory is building. Show Smriti what matters.")
                } else if appModel.isSearching {
                    ProgressView()
                        .controlSize(.small)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if appModel.recallResults.isEmpty {
                    RecallEmptyState(message: "Your memory is building. Show Smriti what matters.")
                } else {
                    ScrollView(.vertical, showsIndicators: false) {
                        VStack(alignment: .leading, spacing: 12) {
                            if let insight = UserDefaults.standard.string(forKey: "smriti.insight.latest"),
                               let storedWeek = UserDefaults.standard.string(forKey: "smriti.insight.week"),
                               !insightDismissed {
                                
                                let currentWeek = ISO8601DateFormatter().string(from: Date()).prefix(8) // Approx, week calculation logic will be exact in the engine.
                                // Actually let's use the explicit check: 
                                let cal = Calendar.current
                                let week = cal.component(.weekOfYear, from: Date())
                                let year = cal.component(.yearForWeekOfYear, from: Date())
                                if storedWeek == "\(year)-W\(week)" {
                                    AnticipationInsightCard(
                                        insight: insight,
                                        onDismiss: { insightDismissed = true },
                                        onSeePattern: { showJournal = true }
                                    )
                                }
                            }
                            
                            if let narration = recallNarration {
                                Text(narration)
                                    .font(.body)
                                    .italic()
                                    .padding(12)
                                    .background(Color.smritiAccent.opacity(0.08))
                                    .cornerRadius(10)
                                    .animation(.smritiSpring, value: recallNarration != nil)
                            }
                            
                            LazyVStack(spacing: 12) {
                            ForEach(Array(appModel.recallResults.enumerated()), id: \.element.id) { index, item in
                                SmritiRecallCard(item: item, index: index) {
                                    appModel.openDetail(for: item)
                                }
                            }
                        }
                        .padding(.bottom, 4)
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .gesture(DragGesture(minimumDistance: 50).onEnded { value in
                if value.translation.width < 0 {
                    showJournal = true
                }
            })
            .sheet(isPresented: $showJournal) {
                JournalView()
                    .frame(width: 400, height: 500)
            }
        }
    }

    private func scheduleRecall() {
        recallTask?.cancel()
        recallTask = Task {
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            
            let query = appModel.recallQuery.trimmingCharacters(in: .whitespacesAndNewlines)
            guard query.count >= 2 else {
                appModel.recallResults = []
                recallNarration = nil
                detectedLanguageCode = nil
                return
            }
            
            appModel.isSearching = true
            do {
                let result = try await recallEngine.query(query)
                appModel.recallResults = result.items
                recallNarration = result.narration
                detectedLanguageCode = result.detectedLanguageCode
            } catch {
                appModel.recallResults = []
                recallNarration = nil
                detectedLanguageCode = nil
            }
            appModel.isSearching = false
        }
    }
}

@MainActor
private final class SurpriseMomentCoordinator: ObservableObject {
    @Published var banner: SurpriseMomentBanner?
    @Published var bannerVisible = false

    private var listenerTask: Task<Void, Never>?
    private var hideTask: Task<Void, Never>?
    private var didShowFirstSurprise = false

    func start(api: SmritiAPI) {
        guard listenerTask == nil else { return }
        listenerTask = Task {
            defer { listenerTask = nil }
            await monitor(api: api)
        }
    }

    func stop() {
        listenerTask?.cancel()
        listenerTask = nil
        hideTask?.cancel()
        hideTask = nil
    }

    private func monitor(api: SmritiAPI) async {
        let backoffSchedule = [1, 2, 4, 8, 16, 30]
        var backoffIndex = 0

        while !Task.isCancelled {
            do {
                let stream = try await api.eventStream()
                backoffIndex = 0

                for try await event in stream {
                    guard !Task.isCancelled else { return }
                    guard event.type == "observation.created" else { continue }
                    guard let observation = event.payload["observation"]?.decode(SurpriseObservationEvent.self) else {
                        continue
                    }
                    guard observation.surpriseScore > 0.75 else {
                        continue
                    }

                    presentFirstSurprise(observation)
                    return
                }
            } catch {
                guard !Task.isCancelled else { return }
                let delay = backoffSchedule[min(backoffIndex, backoffSchedule.count - 1)]
                backoffIndex += 1
                try? await Task.sleep(for: .seconds(delay))
            }
        }
    }

    @MainActor
    private func presentFirstSurprise(_ observation: SurpriseObservationEvent) {
        guard !didShowFirstSurprise else { return }
        didShowFirstSurprise = true
        hideTask?.cancel()

        banner = SurpriseMomentBanner(
            thumbnailPath: observation.thumbnailPath ?? observation.image_path,
            label: observation.label
        )
        bannerVisible = false

        NSHapticFeedbackManager.defaultPerformer.perform(.generic, performanceTime: .default)
        pulseStatusItemButtonIfAvailable()

        withAnimation(.smritiSpring) {
            bannerVisible = true
        }

        hideTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(3))
            guard !Task.isCancelled else { return }
            guard let self else { return }
            withAnimation(.smritiSpring) {
                self.bannerVisible = false
            }
            self.hideTask = Task { [weak self] in
                try? await Task.sleep(for: .milliseconds(380))
                guard let self else { return }
                self.banner = nil
                self.hideTask = nil
                }
        }
    }

    private func pulseStatusItemButtonIfAvailable() {
        guard let button = statusItemButton() else { return }
        button.wantsLayer = true
        button.layer?.anchorPoint = CGPoint(x: 0.5, y: 0.5)
        button.layer?.removeAnimation(forKey: "smriti.firstSurprisePulse")

        let pulse = CASpringAnimation(keyPath: "transform.scale")
        pulse.fromValue = 1.0
        pulse.toValue = 1.3
        pulse.mass = 0.8
        pulse.stiffness = 240
        pulse.damping = 15
        pulse.initialVelocity = 0
        pulse.duration = 0.3
        pulse.autoreverses = true
        pulse.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        button.layer?.add(pulse, forKey: "smriti.firstSurprisePulse")
    }

    private func statusItemButton() -> NSStatusBarButton? {
        guard let delegate = NSApp.delegate else { return nil }
        let mirror = Mirror(reflecting: delegate)
        for child in mirror.children {
            if let statusItem = child.value as? NSStatusItem, let button = statusItem.button {
                return button
            }
        }
        return nil
    }
}

private struct SurpriseMomentBanner: Identifiable {
    let id = UUID()
    let thumbnailPath: String
    let label: String
}

private struct SurpriseMomentBannerView: View {
    let banner: SurpriseMomentBanner

    var body: some View {
        HStack(spacing: 12) {
            SurpriseMomentThumbnail(path: banner.thumbnailPath)
                .frame(width: 32, height: 32)
                .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))

            VStack(alignment: .leading, spacing: 2) {
                Text(banner.label)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)

                Text("⚡ Surprising moment captured")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.white.opacity(0.10))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color.smritiGlassStroke, lineWidth: 0.5)
        )
        .shadow(color: Color.black.opacity(0.18), radius: 18, x: 0, y: 8)
    }
}

private struct SurpriseMomentThumbnail: View {
    let path: String

    var body: some View {
        Group {
            if !path.isEmpty, let image = NSImage(contentsOfFile: path) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                ZStack {
                    LinearGradient(
                        colors: [Color.smritiAccent.opacity(0.42), Color.smritiTeal.opacity(0.26)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                    Image(systemName: "sparkles")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.85))
                }
            }
        }
    }
}

private struct SurpriseObservationEvent: Decodable {
    let id: String
    let image_path: String
    let thumbnail_path: String
    let summary: String?
    let tags: [String]
    let novelty: Double
    let metadata: [String: JSONValue]?

    var label: String {
        if let summary, !summary.isEmpty {
            return summary
        }
        if let tag = tags.first, !tag.isEmpty {
            return tag
        }
        return "Recent memory"
    }

    var thumbnailPath: String? {
        if !thumbnail_path.isEmpty {
            return thumbnail_path
        }
        if !image_path.isEmpty {
            return image_path
        }
        return nil
    }

    var surpriseScore: Double {
        if
            let metrics = metadata?["world_model"]?.decode(SurpriseWorldModelMetrics.self),
            let surprise = metrics.effectiveSurprise
        {
            return surprise
        }
        return novelty
    }
}

private struct SurpriseWorldModelMetrics: Decodable {
    let effectiveSurprise: Double?

    enum CodingKeys: String, CodingKey {
        case effectiveSurprise = "surprise_score"
    }
}

private extension JSONValue {
    func decode<T: Decodable>(_ type: T.Type) -> T? {
        guard JSONSerialization.isValidJSONObject(foundationValue) else {
            return nil
        }
        guard let data = try? JSONSerialization.data(withJSONObject: foundationValue) else {
            return nil
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(type, from: data)
    }

    var foundationValue: Any {
        switch self {
        case .string(let value):
            return value
        case .number(let value):
            return value
        case .integer(let value):
            return value
        case .bool(let value):
            return value
        case .object(let value):
            return value.mapValues(\.foundationValue)
        case .array(let value):
            return value.map(\.foundationValue)
        case .null:
            return NSNull()
        }
    }
}

private struct RecallEmptyState: View {
    let message: String
    @State private var shimmer = false

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "sparkles")
                .font(.system(size: 32, weight: .medium))
                .foregroundStyle(
                    LinearGradient(
                        colors: [Color.smritiTeal, Color.smritiAccent],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .scaleEffect(shimmer ? 1.08 : 0.94)
                .opacity(shimmer ? 1 : 0.65)
                .animation(.smritiSpring.repeatForever(autoreverses: true), value: shimmer)

            Text(message)
                .font(.system(size: 17))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 240)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            shimmer = true
        }
    }
}

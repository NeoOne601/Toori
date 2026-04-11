import SwiftUI
import AppKit

@MainActor
final class SmritiAppModel: ObservableObject {
    static let shared = SmritiAppModel()

    enum BackendPhase: Equatable {
        case idle
        case checking
        case launching
        case ready
        case failed(String)
    }

    enum RootTab: String, CaseIterable, Identifiable {
        case recall = "Recall"
        case mandala = "Mandala"

        var id: String { rawValue }
    }

    enum OnboardingStep: Int {
        case photos = 1
        case folders = 2
        case ingesting = 3
    }

    let api = SmritiAPI()

    @Published var backendPhase: BackendPhase = .idle
    @Published var selectedTab: RootTab = .recall
    @Published var recallQuery = ""
    @Published var recallResults: [SmritiRecallItem] = []
    @Published var isSearching = false
    @Published var mandalaData: SmritiMandalaData?
    @Published var isMandalaLoading = false
    @Published var selectedRecallItem: SmritiRecallItem?
    @Published var shouldPresentOnboarding = false
    @Published var onboardingStep: OnboardingStep = .photos
    @Published var watchFolders: [WatchFolderStatus] = []
    @Published var ingestionCount = 0
    @Published var onboardingStatus = "Waiting for your first memories."
    @Published var journalStatusMessage: String?

    var detailPresenter: ((SmritiRecallItem) -> Void)?
    var detailDismiss: (() -> Void)?
    var folderPicker: ((URL?) async -> URL?)?
    var sharePresenter: (([Any]) -> Void)?

    private var onboardingEventsTask: Task<Void, Never>?

    private init() {}

    var hasCompletedOnboarding: Bool {
        get { UserDefaults.standard.bool(forKey: "smriti.hasCompletedOnboarding") }
        set { UserDefaults.standard.set(newValue, forKey: "smriti.hasCompletedOnboarding") }
    }

    func openDetail(for item: SmritiRecallItem) {
        selectedRecallItem = item
        detailPresenter?(item)
    }

    func closeDetail() {
        selectedRecallItem = nil
        detailDismiss?()
    }

    func resetOnboardingCounter() {
        ingestionCount = 0
        onboardingStatus = "Listening for new observations."
    }

    func finishOnboarding() {
        hasCompletedOnboarding = true
        shouldPresentOnboarding = false
        onboardingEventsTask?.cancel()
        onboardingEventsTask = nil
    }

    func skipOnboarding() {
        finishOnboarding()
    }

    func maybePresentOnboarding() {
        shouldPresentOnboarding = !hasCompletedOnboarding
    }

    func loadMandalaIfNeeded() async {
        guard mandalaData == nil, !isMandalaLoading else { return }
        await reloadMandala()
    }

    func reloadMandala() async {
        isMandalaLoading = true
        defer { isMandalaLoading = false }
        do {
            mandalaData = try await api.fetchClusters()
        } catch {
            mandalaData = nil
        }
    }

    func runRecall() async {
        let trimmed = recallQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 2 else {
            recallResults = []
            isSearching = false
            return
        }
        isSearching = true
        defer { isSearching = false }
        do {
            let response = try await api.recall(
                SmritiRecallRequest(
                    query: trimmed,
                    session_id: "smriti-macos",
                    top_k: 20,
                    person_filter: nil,
                    location_filter: nil,
                    time_start: nil,
                    time_end: nil,
                    min_confidence: 0
                )
            )
            recallResults = response.results
        } catch {
            recallResults = []
        }
    }

    func refreshWatchFolders() async {
        do {
            watchFolders = try await api.listWatchFolders()
        } catch {
            watchFolders = []
        }
    }

    func pickAndAddWatchFolder(defaultURL: URL?, nextStep: OnboardingStep?) async {
        guard let folder = await folderPicker?(defaultURL) else { return }
        do {
            let status = try await api.addWatchFolder(path: folder.path)
            watchFolders.removeAll { $0.path == status.path }
            watchFolders.append(status)
            watchFolders.sort { $0.path.localizedCaseInsensitiveCompare($1.path) == .orderedAscending }
            if let nextStep {
                onboardingStep = nextStep
            }
            onboardingStatus = "Watching \(folder.lastPathComponent)…"
        } catch {
            onboardingStatus = "Couldn’t add \(folder.lastPathComponent)."
        }
    }

    func startOnboardingEventStreamIfNeeded() {
        guard onboardingEventsTask == nil else { return }
        onboardingEventsTask = Task { [weak self] in
            guard let self else { return }
            do {
                let stream = try await api.eventStream()
                for try await event in stream {
                    guard !Task.isCancelled else { break }
                    if event.type == "observation.created" {
                        await MainActor.run {
                            self.ingestionCount += 1
                            self.onboardingStatus = "\(self.ingestionCount) memories indexed."
                        }
                    }
                }
            } catch {
                await MainActor.run {
                    self.onboardingStatus = "Event stream unavailable. Watching folders anyway."
                }
            }
        }
    }

    func addCurrentItemToJournal() async {
        guard let item = selectedRecallItem else { return }
        if item.person_names.count == 1, let name = item.person_names.first {
            await addToJournal(item: item, personName: name)
        }
    }

    func addToJournal(item: SmritiRecallItem, personName: String) async {
        do {
            _ = try await api.tagPerson(
                request: SmritiTagPersonRequest(
                    media_id: item.media_id,
                    person_name: personName,
                    confirmed: true
                )
            )
            journalStatusMessage = "Added \(personName) to journal."
        } catch {
            journalStatusMessage = "Couldn’t update the journal."
        }
    }

    func share(item: SmritiRecallItem) {
        var shareItems: [Any] = [item.primary_description]
        
        let targetPath = item.file_path
        let summaryText = item.primary_description
        
        if let cardURL = MemoryCardGenerator().generateMemoryCard(imagePath: targetPath, date: item.created_at, summary: summaryText) {
            shareItems.append(cardURL)
        } else {
            shareItems.append(URL(fileURLWithPath: targetPath))
        }
        
        sharePresenter?(shareItems)
    }
}

struct SmritiGlassBackground: NSViewRepresentable {
    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.blendingMode = .behindWindow
        view.material = .hudWindow
        view.state = .active
        view.alphaValue = 0.75
        return view
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {
        nsView.alphaValue = 0.75
    }
}

struct SmritiGlassSurface<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        ZStack {
            SmritiGlassBackground()
            LinearGradient(
                colors: [
                    Color.smritiAccent.opacity(0.14),
                    Color.white.opacity(0.02),
                    Color.black.opacity(0.18)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            content
                .padding(20)
        }
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.white.opacity(0.08), lineWidth: 0.5)
        )
    }
}

struct SmritiLoadingView: View {
    @EnvironmentObject private var appModel: SmritiAppModel
    @State private var ringTrim: CGFloat = 0.22
    @State private var rotation: Double = 0

    private var subtitle: String {
        switch appModel.backendPhase {
        case .idle, .checking:
            return "Checking your local daemon."
        case .launching:
            return "Launching the Smriti runtime."
        case .ready:
            return "Ready."
        case .failed(let message):
            return message
        }
    }

    var body: some View {
        SmritiGlassSurface {
            VStack(spacing: 22) {
                ZStack {
                    Circle()
                        .stroke(Color.white.opacity(0.08), lineWidth: 10)
                        .frame(width: 74, height: 74)
                    Circle()
                        .trim(from: 0.08, to: ringTrim)
                        .stroke(
                            AngularGradient(
                                gradient: Gradient(colors: [.smritiTeal, .smritiAccent]),
                                center: .center
                            ),
                            style: StrokeStyle(lineWidth: 10, lineCap: .round)
                        )
                        .frame(width: 74, height: 74)
                        .rotationEffect(.degrees(rotation))
                }
                VStack(spacing: 8) {
                    Text("Smriti")
                        .font(.system(size: 28, weight: .semibold, design: .default))
                    Text(subtitle)
                        .font(.system(size: 13, weight: .regular, design: .default))
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding(20)
        .onAppear {
            withAnimation(.smritiSpring.repeatForever(autoreverses: false)) {
                ringTrim = 0.92
                rotation = 360
            }
        }
    }
}

struct SmritiRootView: View {
    @EnvironmentObject private var appModel: SmritiAppModel

    var body: some View {
        ZStack {
            switch appModel.backendPhase {
            case .ready:
                RecallView()
            case .failed:
                SmritiLoadingView()
                    .overlay(alignment: .bottom) {
                        Text("Runtime unavailable")
                            .font(.system(size: 11, weight: .medium))
                            .padding(.bottom, 20)
                            .foregroundStyle(.secondary)
                    }
            default:
                SmritiLoadingView()
            }
        }
        .frame(width: 380, height: 520)
        .sheet(isPresented: $appModel.shouldPresentOnboarding) {
            OnboardingSheet()
                .environmentObject(appModel)
                .frame(minWidth: 500, minHeight: 520)
        }
    }
}

@main
struct SmritiApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var appModel = SmritiAppModel.shared

    var body: some Scene {
        Settings {
            EmptyView()
        }
        .environmentObject(appModel)
    }
}

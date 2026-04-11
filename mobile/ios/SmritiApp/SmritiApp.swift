import SwiftUI
import UIKit
import BackgroundTasks
    static let smritiAccent = Color(red: 0.4196, green: 0.3607, blue: 0.9058)
    static let smritiTeal = Color(red: 0.235, green: 0.765, blue: 0.765)
    static let smritiDivider = Color.white.opacity(0.12)
    static let smritiSurface = Color.white.opacity(0.06)
    static let smritiStroke = Color.white.opacity(0.08)
    static let smritiCanvas = Color.black.opacity(0.96)
}

extension Animation {
    static let smritiSpring = Animation.spring(response: 0.38, dampingFraction: 0.72)
}

@MainActor
final class SmritiAppModel: ObservableObject {
    enum RootTab: Hashable {
        case pulse
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
            settingsStatusMessage = nil
        } catch {
            settingsStatusMessage = error.localizedDescription
        }
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
    }
}

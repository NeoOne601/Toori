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


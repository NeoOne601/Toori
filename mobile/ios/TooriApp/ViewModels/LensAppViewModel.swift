import Foundation
import SwiftUI
import Combine

@MainActor
final class LensAppViewModel: ObservableObject {
    @Published var settings: RuntimeSettings?
    @Published var health: [ProviderHealth] = []
    @Published var observations: [Observation] = []
    @Published var latestAnswer: Answer?
    @Published var latestHits: [SearchHit] = []
    @Published var searchHits: [SearchHit] = []
    @Published var searchAnswer: Answer?
    @Published var groundedSummary: String?
    @Published var confidenceLabel: String?
    @Published var prompt = ""
    @Published var searchText = ""
    @Published var status = "Idle"
    @Published var isRuntimeConnected = false

    // B1 — Compare mode state
    @Published var compareMode = false
    @Published var imageA: UIImage?
    @Published var imageB: UIImage?

    // D1 — Calibration state
    @Published var isCalibrated = false
    @Published var calibrationMessage: String?
    @Published var showCalibrationSheet = false

    // Persistent session ID — survives app backgrounding
    @Published var sessionId: String

    let api = TooriAPIClient()
    let camera = CameraService()
    private let discovery = DiscoveryService()
    private var discoveryCancellable: AnyCancellable?

    private static let sessionIdKey = "toori.sessionId"

    init() {
        if let saved = UserDefaults.standard.string(forKey: Self.sessionIdKey) {
            sessionId = saved
        } else {
            let newId = UUID().uuidString
            sessionId = newId
            UserDefaults.standard.set(newId, forKey: Self.sessionIdKey)
        }
    }

    func bootstrap() async {
        status = "Searching for backend..."
        discovery.startBrowsing()

        discoveryCancellable = discovery.$discoveredEndpoint.sink { [weak self] url in
            if let url = url {
                self?.api.baseURL = url
                self?.status = "Discovered \(url.host ?? "backend")"
                Task {
                    await self?.refresh()
                }
            }
        }

        await refresh()
    }

    func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }

    func refresh() async {
        do {
            async let settingsTask = api.fetchSettings()
            async let healthTask = api.fetchHealth()
            async let observationTask = api.fetchObservations(sessionId: sessionId)
            settings = try await settingsTask
            health = try await healthTask
            observations = try await observationTask
            status = "Connected to runtime"
            isRuntimeConnected = true
        } catch {
            status = error.localizedDescription
            isRuntimeConnected = false
        }
    }

    func captureAndAnalyze() async {
        hideKeyboard()
        do {
            status = "Capturing frame..."
            let rawData = try await camera.capturePhoto()

            guard let image = UIImage(data: rawData) else {
                status = "Error: Invalid image data"
                return
            }

            status = "Optimizing for iMac..."
            guard let optimizedData = image.toOptimizedData() else {
                status = "Error: Optimization failed"
                return
            }

            status = "Uploading to iMac..."
            let response = try await api.analyze(imageData: optimizedData, sessionId: sessionId, prompt: prompt.isEmpty ? nil : prompt)

            self.latestAnswer = response.answer
            self.groundedSummary = response.grounded_summary
            self.confidenceLabel = response.confidence_label
            self.latestHits = response.hits
            self.health = response.provider_health
            self.status = "Analyzed \(response.observation.id)"

            try? await Task.sleep(nanoseconds: 1_000_000_000)
            await refresh()
        } catch {
            let nsError = error as NSError
            if nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorNetworkConnectionLost {
                status = "iMac Memory Overflow. Backend is restarting (8GB Limit)."
            } else if nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorTimedOut {
                status = "iMac is heavy thinking (Swap Thrash). Request timed out."
            } else {
                status = error.localizedDescription
            }
            print("❌ [Analysis] Error: \(error)")
        }
    }

    // MARK: - Multi-turn text follow-up

    func sendTextQuery(_ text: String) async -> AnalyzeResponse? {
        do {
            return try await api.analyzeText(query: text, sessionId: sessionId)
        } catch {
            status = error.localizedDescription
            return nil
        }
    }

    // MARK: - B1 Compare

    func compareImages(query: String?) async -> CompareResponse? {
        guard let a = imageA, let b = imageB,
              let jpegA = a.jpegData(compressionQuality: 0.75),
              let jpegB = b.jpegData(compressionQuality: 0.75) else { return nil }
        do {
            return try await api.compare(imageA: jpegA, imageB: jpegB, sessionId: sessionId, query: query)
        } catch {
            status = "Compare failed: \(error.localizedDescription)"
            return nil
        }
    }

    // MARK: - D1 Calibrate

    func calibrate(preset: KnownObjectPreset) async {
        do {
            let response = try await api.calibrate(
                sessionId: sessionId,
                label: preset.rawValue,
                realWidthCm: preset.realWidthCm,
                bbox: nil
            )
            isCalibrated = true
            calibrationMessage = response.message
        } catch {
            calibrationMessage = "Calibration failed: \(error.localizedDescription)"
        }
        showCalibrationSheet = false
    }

    // MARK: - New conversation thread

    func newConversation() {
        let newId = UUID().uuidString
        sessionId = newId
        UserDefaults.standard.set(newId, forKey: Self.sessionIdKey)
        groundedSummary = nil
        confidenceLabel = nil
        latestAnswer = nil
        latestHits = []
        imageA = nil
        imageB = nil
        compareMode = false
        isCalibrated = false
        calibrationMessage = nil
    }

    func runSearch() async {
        hideKeyboard()
        guard let settings else { return }
        do {
            let response = try await api.search(query: searchText, sessionId: sessionId, topK: settings.top_k)
            searchHits = response.hits
            searchAnswer = response.answer
        } catch {
            status = error.localizedDescription
        }
    }

    func saveSettings() async {
        hideKeyboard()
        guard let settings else { return }
        do {
            self.settings = try await api.updateSettings(settings)
            status = "Settings saved"
        } catch {
            status = error.localizedDescription
        }
    }
}


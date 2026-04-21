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
    @Published var sessionId = "ios-live"
    @Published var prompt = ""
    @Published var searchText = ""
    @Published var status = "Idle"
    @Published var isRuntimeConnected = false

    let api = TooriAPIClient()
    let camera = CameraService()
    private let discovery = DiscoveryService()
    private var discoveryCancellable: AnyCancellable?

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
            
            // Dispatch updates to MainActor
            self.latestAnswer = response.answer
            self.groundedSummary = response.grounded_summary
            self.confidenceLabel = response.confidence_label
            self.latestHits = response.hits
            self.health = response.provider_health
            self.status = "Analyzed \(response.observation.id)"
            
            // Wait a moment so the UI can show the Analyze result before refresh clears status
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

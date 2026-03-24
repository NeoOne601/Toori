import Foundation
import SwiftUI

@MainActor
final class LensAppViewModel: ObservableObject {
    @Published var settings: RuntimeSettings?
    @Published var health: [ProviderHealth] = []
    @Published var observations: [Observation] = []
    @Published var latestAnswer: Answer?
    @Published var latestHits: [SearchHit] = []
    @Published var searchHits: [SearchHit] = []
    @Published var searchAnswer: Answer?
    @Published var sessionId = "ios-live"
    @Published var prompt = ""
    @Published var searchText = ""
    @Published var status = "Idle"

    let api = TooriAPIClient()
    let camera = CameraService()

    func bootstrap() async {
        await refresh()
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
        } catch {
            status = error.localizedDescription
        }
    }

    func captureAndAnalyze() async {
        do {
            status = "Capturing frame"
            let imageData = try await camera.capturePhoto()
            let response = try await api.analyze(imageData: imageData, sessionId: sessionId, prompt: prompt.isEmpty ? nil : prompt)
            latestAnswer = response.answer
            latestHits = response.hits
            health = response.provider_health
            status = "Analyzed \(response.observation.id)"
            await refresh()
        } catch {
            status = error.localizedDescription
        }
    }

    func runSearch() async {
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
        guard let settings else { return }
        do {
            self.settings = try await api.updateSettings(settings)
            status = "Settings saved"
        } catch {
            status = error.localizedDescription
        }
    }
}

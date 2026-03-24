import Foundation

struct ProviderConfig: Codable, Hashable {
    var name: String
    var enabled: Bool
    var base_url: String?
    var model: String?
    var model_path: String?
}

struct RuntimeSettings: Codable {
    var runtime_profile: String
    var sampling_fps: Double
    var top_k: Int
    var retention_days: Int
    var primary_perception_provider: String
    var reasoning_backend: String
    var local_reasoning_disabled: Bool
    var providers: [String: ProviderConfig]
}

struct Observation: Codable, Identifiable, Hashable {
    let id: String
    let session_id: String
    let created_at: String
    let image_path: String
    let thumbnail_path: String
    let summary: String?
    let source_query: String?
    let confidence: Double
    let novelty: Double
    let providers: [String]
}

struct SearchHit: Codable, Identifiable, Hashable {
    var id: String { observation_id }
    let observation_id: String
    let score: Double
    let summary: String?
    let thumbnail_path: String
    let session_id: String
    let created_at: String
}

struct Answer: Codable, Hashable {
    let text: String
    let provider: String
    let confidence: Double
}

struct ProviderHealth: Codable, Hashable {
    let name: String
    let role: String
    let enabled: Bool
    let healthy: Bool
    let message: String
}

struct AnalyzeResponse: Codable {
    let observation: Observation
    let hits: [SearchHit]
    let answer: Answer?
    let provider_health: [ProviderHealth]
}

struct QueryResponse: Codable {
    let hits: [SearchHit]
    let answer: Answer?
    let provider_health: [ProviderHealth]
}

struct ObservationsResponse: Codable {
    let observations: [Observation]
}

struct ProviderHealthResponse: Codable {
    let providers: [ProviderHealth]
}

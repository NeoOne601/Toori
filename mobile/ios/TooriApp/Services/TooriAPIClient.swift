import Foundation

final class TooriAPIClient {
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()
    private let baseURL: URL

    init(baseURL: URL = URL(string: ProcessInfo.processInfo.environment["TOORI_RUNTIME_URL"] ?? "http://127.0.0.1:7777")!) {
        self.baseURL = baseURL
    }

    func fetchSettings() async throws -> RuntimeSettings {
        try await request(path: "/v1/settings", method: "GET", body: Optional<String>.none)
    }

    func updateSettings(_ settings: RuntimeSettings) async throws -> RuntimeSettings {
        try await request(path: "/v1/settings", method: "PUT", body: settings)
    }

    func fetchHealth() async throws -> [ProviderHealth] {
        let response: ProviderHealthResponse = try await request(path: "/v1/providers/health", method: "GET", body: Optional<String>.none)
        return response.providers
    }

    func fetchObservations(sessionId: String) async throws -> [Observation] {
        let response: ObservationsResponse = try await request(
            path: "/v1/observations?session_id=\(sessionId.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? sessionId)&limit=48",
            method: "GET",
            body: Optional<String>.none
        )
        return response.observations
    }

    func analyze(imageData: Data, sessionId: String, prompt: String?) async throws -> AnalyzeResponse {
        struct AnalyzePayload: Codable {
            let image_base64: String
            let session_id: String
            let query: String?
            let decode_mode: String
        }

        return try await request(
            path: "/v1/analyze",
            method: "POST",
            body: AnalyzePayload(
                image_base64: imageData.base64EncodedString(),
                session_id: sessionId,
                query: prompt,
                decode_mode: "auto"
            )
        )
    }

    func search(query: String, sessionId: String, topK: Int) async throws -> QueryResponse {
        struct QueryPayload: Codable {
            let query: String
            let session_id: String
            let top_k: Int
        }

        return try await request(
            path: "/v1/query",
            method: "POST",
            body: QueryPayload(query: query, session_id: sessionId, top_k: topK)
        )
    }

    private func request<Response: Decodable, Body: Encodable>(
        path: String,
        method: String,
        body: Body?
    ) async throws -> Response {
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw URLError(.badURL)
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let body {
            request.httpBody = try encoder.encode(body)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse, (200..<300).contains(httpResponse.statusCode) else {
            throw URLError(.badServerResponse)
        }
        return try decoder.decode(Response.self, from: data)
    }
}

import Foundation

final class TooriAPIClient {
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()
    var baseURL: URL

    init(baseURL: URL = URL(string: ProcessInfo.processInfo.environment["TOORI_RUNTIME_URL"] ?? "http://192.168.0.174:7777")!) {
        self.baseURL = baseURL
    }
    
    func ping() async -> Bool {
        do {
            let _: [ProviderHealth] = try await fetchHealth()
            return true
        } catch {
            return false
        }
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
            ),
            timeout: 120.0
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

    // MARK: - Compare (Sprint B1)

    func compare(imageA: Data, imageB: Data, sessionId: String, query: String?) async throws -> CompareResponse {
        struct ComparePayload: Codable {
            let image_base64_a: String
            let image_base64_b: String
            let session_id: String
            let query: String?
            let decode_mode: String
        }
        return try await request(
            path: "/v1/analyze/compare",
            method: "POST",
            body: ComparePayload(
                image_base64_a: imageA.base64EncodedString(),
                image_base64_b: imageB.base64EncodedString(),
                session_id: sessionId,
                query: query,
                decode_mode: query != nil ? "auto" : "off"
            ),
            timeout: 180.0
        )
    }

    // MARK: - Calibrate (Sprint D1)

    func calibrate(sessionId: String, label: String, realWidthCm: Double?, bbox: [String: Double]?) async throws -> CalibrationResponse {
        return try await request(
            path: "/v1/calibrate",
            method: "POST",
            body: CalibratePayload(session_id: sessionId, label: label, real_width_cm: realWidthCm, bbox: bbox),
            timeout: 10.0
        )
    }

    // MARK: - Text-only follow-up (multi-turn)

    func analyzeText(query: String, sessionId: String) async throws -> AnalyzeResponse {
        return try await request(
            path: "/v1/analyze",
            method: "POST",
            body: TextAnalyzePayload(session_id: sessionId, query: query, decode_mode: "force"),
            timeout: 60.0
        )
    }

    private func request<Response: Decodable, Body: Encodable>(
        path: String,
        method: String,
        body: Body?,
        timeout: TimeInterval = 90.0
    ) async throws -> Response {
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw URLError(.badURL)
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = timeout
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

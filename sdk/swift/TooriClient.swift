import Foundation

public final class TooriClient {
    private let baseURL: URL
    private let apiKey: String?

    public init(baseURL: URL = URL(string: "http://127.0.0.1:7777")!, apiKey: String? = nil) {
        self.baseURL = baseURL
        self.apiKey = apiKey
    }

    public func settings() async throws -> Data {
        try await request(path: "/v1/settings", method: "GET", body: nil)
    }

    public func query(text: String, sessionId: String = "default", topK: Int = 6) async throws -> Data {
        let payload = ["query": text, "session_id": sessionId, "top_k": topK] as [String : Any]
        return try await request(path: "/v1/query", method: "POST", body: payload)
    }

    private func request(path: String, method: String, body: [String: Any]?) async throws -> Data {
        var request = URLRequest(url: URL(string: path, relativeTo: baseURL)!)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let apiKey {
            request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")
        }
        if let body {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        return data
    }
}

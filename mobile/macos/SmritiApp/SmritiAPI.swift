import Foundation

enum SmritiAPIError: LocalizedError {
    case badResponse
    case nonLocalURL

    var errorDescription: String? {
        switch self {
        case .badResponse:
            return "The local runtime returned an unexpected response."
        case .nonLocalURL:
            return "Smriti only talks to localhost."
        }
    }
}

struct SmritiAPI: Sendable {
    private let baseURL: URL

    init(baseURL: URL = URL(string: "http://127.0.0.1:7777")!) {
        guard ["127.0.0.1", "localhost"].contains(baseURL.host() ?? "") else {
            fatalError(SmritiAPIError.nonLocalURL.localizedDescription)
        }

        self.baseURL = baseURL
    }

    func healthCheck() async throws -> Bool {
        struct Healthz: Decodable { let status: String }
        let response: Healthz = try await request(path: "/healthz", method: "GET", body: Optional<String>.none)
        return response.status == "ok"
    }

    func recall(_ request: SmritiRecallRequest) async throws -> SmritiRecallResponse {
        try await self.request(path: "/v1/smriti/recall", method: "POST", body: request)
    }

    func fetchClusters() async throws -> SmritiMandalaData {
        try await request(path: "/v1/smriti/clusters", method: "GET", body: Optional<String>.none)
    }

    func fetchPersonJournal(name: String) async throws -> SmritiPersonJournal {
        try await request(
            path: "/v1/smriti/person/\(name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? name)/journal",
            method: "GET",
            body: Optional<String>.none
        )
    }

    func fetchStorageUsage() async throws -> StorageUsageReport {
        try await request(path: "/v1/smriti/storage/usage", method: "GET", body: Optional<String>.none)
    }

    func listWatchFolders() async throws -> [WatchFolderStatus] {
        try await request(path: "/v1/smriti/watch-folders", method: "GET", body: Optional<String>.none)
    }

    func addWatchFolder(path: String) async throws -> WatchFolderStatus {
        struct Request: Codable { let path: String }
        return try await request(path: "/v1/smriti/watch-folders", method: "POST", body: Request(path: path))
    }

    func fetchRuntimeSnapshot(sessionID: String) async throws -> RuntimeSnapshotResponse {
        try await request(
            path: "/v1/runtime/snapshot?session_id=\(sessionID.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? sessionID)",
            method: "GET",
            body: Optional<String>.none
        )
    }

    func livingLensTick(_ request: LivingLensTickRequest) async throws -> LivingLensTickResponse {
        try await self.request(path: "/v1/living-lens/tick", method: "POST", body: request)
    }

    func ingestFolder(path: String) async throws -> SmritiIngestResponse {
        struct Request: Codable { let folder_path: String }
        return try await request(path: "/v1/smriti/ingest", method: "POST", body: Request(folder_path: path))
    }

    func tagPerson(request: SmritiTagPersonRequest) async throws -> SmritiTagPersonResponse {
        try await self.request(path: "/v1/smriti/tag/person", method: "POST", body: request)
    }

    func eventStream() async throws -> AsyncThrowingStream<EventMessage, Error> {
        let url = URL(string: "ws://127.0.0.1:7777/v1/events")!
        let task = URLSession.shared.webSocketTask(with: url)
        let decoder = Self.makeDecoder()
        task.resume()

        return AsyncThrowingStream { continuation in
            @Sendable func receiveNext() {
                task.receive { result in
                    switch result {
                    case .failure(let error):
                        continuation.finish(throwing: error)
                    case .success(let message):
                        switch message {
                        case .string(let text):
                            do {
                                let data = Data(text.utf8)
                                let event = try decoder.decode(EventMessage.self, from: data)
                                continuation.yield(event)
                                receiveNext()
                            } catch {
                                continuation.finish(throwing: error)
                            }
                        case .data(let data):
                            do {
                                let event = try decoder.decode(EventMessage.self, from: data)
                                continuation.yield(event)
                                receiveNext()
                            } catch {
                                continuation.finish(throwing: error)
                            }
                        @unknown default:
                            continuation.finish(throwing: SmritiAPIError.badResponse)
                        }
                    }
                }
            }

            receiveNext()

            continuation.onTermination = { _ in
                task.cancel(with: .goingAway, reason: nil)
            }
        }
    }

    private func request<Response: Decodable, Body: Encodable>(
        path: String,
        method: String,
        body: Body?
    ) async throws -> Response {
        let encoder = Self.makeEncoder()
        let decoder = Self.makeDecoder()
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw URLError(.badURL)
        }
        guard ["127.0.0.1", "localhost"].contains(url.host() ?? baseURL.host() ?? "") else {
            throw SmritiAPIError.nonLocalURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let body {
            request.httpBody = try encoder.encode(body)
        }

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw SmritiAPIError.badResponse
        }
        return try decoder.decode(Response.self, from: data)
    }

    private static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    private static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let value = try container.decode(String.self)
            let primary = ISO8601DateFormatter()
            primary.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            if let date = primary.date(from: value) {
                return date
            }
            let fallback = ISO8601DateFormatter()
            fallback.formatOptions = [.withInternetDateTime]
            if let date = fallback.date(from: value) {
                return date
            }
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Invalid ISO-8601 date: \(value)")
        }
        return decoder
    }
}

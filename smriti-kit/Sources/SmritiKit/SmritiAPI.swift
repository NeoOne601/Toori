import Foundation

public enum SmritiAPIError: LocalizedError {
    case badResponse
    case invalidHost
    case disallowedHost

    public var errorDescription: String? {
        switch self {
        case .badResponse:
            return "The local runtime returned an unexpected response."
        case .invalidHost:
            return "Smriti needs a valid local host."
        case .disallowedHost:
            return "Smriti only talks to localhost or a private local network host."
        }
    }
}

public struct SmritiAPI: Sendable {
    public let baseURL: URL

    public init() {
        self.baseURL = URL(string: "http://127.0.0.1:7777")!
    }

    public init(baseURL: URL) throws {
        guard let host = baseURL.host, Self.isAllowedHostString(host) else {
            throw SmritiAPIError.disallowedHost
        }
        self.baseURL = baseURL
    }

    public init(host: String) throws {
        let url = try Self.resolveBaseURL(from: host)
        guard let resolvedHost = url.host, Self.isAllowedHostString(resolvedHost) else {
            throw SmritiAPIError.disallowedHost
        }
        self.baseURL = url
    }

    public var websocketURL: URL {
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false) ?? URLComponents()
        components.scheme = baseURL.scheme == "https" ? "wss" : "ws"
        return components.url ?? URL(string: "ws://127.0.0.1:7777/v1/events")!
    }

    public func healthCheck() async throws -> Bool {
        struct Healthz: Decodable { let status: String }
        let response: Healthz = try await request(path: "/healthz", method: "GET", body: Optional<String>.none)
        return response.status == "ok"
    }

    public func listObservations(limit: Int = 9, summaryOnly: Bool = true) async throws -> ObservationSummariesResponse {
        let query = URLQueryItem(name: "limit", value: String(limit))
        let summaryOnlyItem = URLQueryItem(name: "summary_only", value: summaryOnly ? "true" : "false")
        return try await request(
            path: "/v1/observations?limit=\(query.value ?? "9")&summary_only=\(summaryOnlyItem.value ?? "true")",
            method: "GET",
            body: Optional<String>.none
        )
    }

    public func recall(_ request: SmritiRecallRequest) async throws -> SmritiRecallResponse {
        try await self.request(path: "/v1/smriti/recall", method: "POST", body: request)
    }

    public func fetchClusters() async throws -> SmritiMandalaData {
        try await request(path: "/v1/smriti/clusters", method: "GET", body: Optional<String>.none)
    }

    public func fetchPersonJournal(name: String) async throws -> SmritiPersonJournal {
        try await request(
            path: "/v1/smriti/person/\(name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? name)/journal",
            method: "GET",
            body: Optional<String>.none
        )
    }

    public func fetchStorageUsage() async throws -> StorageUsageReport {
        try await request(path: "/v1/smriti/storage/usage", method: "GET", body: Optional<String>.none)
    }

    public func listWatchFolders() async throws -> [WatchFolderStatus] {
        try await request(path: "/v1/smriti/watch-folders", method: "GET", body: Optional<String>.none)
    }

    public func addWatchFolder(path: String) async throws -> WatchFolderStatus {
        struct Request: Codable { let path: String }
        return try await request(path: "/v1/smriti/watch-folders", method: "POST", body: Request(path: path))
    }

    public func fetchRuntimeSnapshot(sessionID: String) async throws -> RuntimeSnapshotResponse {
        try await request(
            path: "/v1/runtime/snapshot?session_id=\(sessionID.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? sessionID)",
            method: "GET",
            body: Optional<String>.none
        )
    }

    public func livingLensTick(_ request: LivingLensTickRequest) async throws -> LivingLensTickResponse {
        try await self.request(path: "/v1/living-lens/tick", method: "POST", body: request)
    }

    public func ingestFolder(path: String) async throws -> SmritiIngestResponse {
        struct Request: Codable { let folder_path: String }
        return try await request(path: "/v1/smriti/ingest", method: "POST", body: Request(folder_path: path))
    }

    public func ingestFile(path: String) async throws -> SmritiIngestResponse {
        struct Request: Codable { let file_path: String }
        return try await request(path: "/v1/smriti/ingest", method: "POST", body: Request(file_path: path))
    }

    public func tagPerson(request: SmritiTagPersonRequest) async throws -> SmritiTagPersonResponse {
        try await self.request(path: "/v1/smriti/tag/person", method: "POST", body: request)
    }

    public func audioQuery(_ request: AudioQueryRequest) async throws -> AudioQueryResponse {
        try await self.request(path: "/v1/audio/query", method: "POST", body: request)
    }

    public func analyze(_ request: AnalyzeRequest) async throws -> AnalyzeResponse {
        try await self.request(path: "/v1/analyze", method: "POST", body: request)
    }

    public func eventStream() async throws -> AsyncThrowingStream<EventMessage, Error> {
        let task = URLSession.shared.webSocketTask(with: websocketURL)
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

    public static func isAllowedHostString(_ host: String) -> Bool {
        let normalized = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return false }
        if normalized == "localhost" || normalized == "127.0.0.1" || normalized == "::1" {
            return true
        }
        if normalized.hasSuffix(".local") {
            return true
        }
        return isPrivateIPv4(normalized)
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
        guard let host = url.host, Self.isAllowedHostString(host) else {
            throw SmritiAPIError.disallowedHost
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

    private static func resolveBaseURL(from host: String) throws -> URL {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw SmritiAPIError.invalidHost
        }

        if let url = URLComponents(string: trimmed)?.url,
           let hostname = URLComponents(string: trimmed)?.host,
           isAllowedHostString(hostname)
        {
            return url
        }

        let prefixed = trimmed.hasPrefix("http://") || trimmed.hasPrefix("https://") ? trimmed : "http://\(trimmed)"
        guard let url = URL(string: prefixed), let hostname = url.host, isAllowedHostString(hostname) else {
            throw SmritiAPIError.disallowedHost
        }
        return url
    }

    static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    static func makeDecoder() -> JSONDecoder {
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

    private static func isPrivateIPv4(_ host: String) -> Bool {
        let parts = host.split(separator: ".").compactMap { Int($0) }
        guard parts.count == 4, parts.allSatisfy({ 0...255 ~= $0 }) else { return false }
        if parts[0] == 10 {
            return true
        }
        if parts[0] == 172 {
            return (16...31).contains(parts[1])
        }
        if parts[0] == 192 {
            return parts[1] == 168
        }
        if parts[0] == 169 {
            return parts[1] == 254
        }
        return false
    }
}

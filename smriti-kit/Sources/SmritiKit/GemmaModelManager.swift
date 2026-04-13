import Foundation

public enum DeviceTier: String, Codable {
    case base
    case standard
    case enhanced
}

@MainActor
public final class GemmaModelManager: ObservableObject {

    public static let shared = GemmaModelManager()

    @Published public var downloadState: DownloadState = .idle
    @Published public var downloadProgress: Double = 0.0

    public enum DownloadState: Equatable { case idle, downloading, ready, error(String) }

    private var daemonProcess: Process?

    public init() {}

    public func detectTier() -> DeviceTier {
        if let override = UserDefaults.standard.string(forKey: "smriti.gemma.tier_override"),
           let tier = DeviceTier(rawValue: override) {
            return tier
        }
        
        let memory = ProcessInfo.processInfo.physicalMemory
        let tier: DeviceTier
        if memory < 6_000_000_000 {
            tier = .base
        } else if memory < 10_000_000_000 {
            tier = .standard
        } else {
            tier = .enhanced
        }
        
        // Store result in UserDefaults "smriti.device.tier" on first run
        if UserDefaults.standard.string(forKey: "smriti.device.tier") == nil {
            UserDefaults.standard.set(tier.rawValue, forKey: "smriti.device.tier")
        }
        return tier
    }

    public func modelDirectory(for variant: String) -> URL {
        #if os(macOS)
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("Smriti").appendingPathComponent("models").appendingPathComponent(variant)
        #elseif os(iOS)
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        var dir = docs.appendingPathComponent("smriti-models").appendingPathComponent(variant)
        var resourceValues = URLResourceValues()
        resourceValues.isExcludedFromBackup = true
        try? dir.setResourceValues(resourceValues)
        return dir
        #else
        return URL(fileURLWithPath: NSTemporaryDirectory())
        #endif
    }

    public func isModelPresent() -> Bool {
        let tier = detectTier()
        if tier == .base { return true }
        let dir = modelDirectory(for: selectedVariant())
        return FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path)
    }

    public func selectedVariant() -> String {
        let tier = detectTier()
        switch tier {
        case .base: return "Essentials (via backend)"
        case .standard: return "gemma-4-e2b-it-4bit"
        case .enhanced: return "gemma-4-e4b-it-4bit"
        }
    }

    public func markModelReady() {
        downloadState = .ready
    }

    private struct MLXReasonerInput: Codable {
        let prompt: String
        let image_base64: String?
        let max_tokens: Int
    }
    
    private struct MLXReasonerOutput: Codable {
        let text: String?
        let tokens_generated: Int?
        let error: String?
    }

    public func generate(prompt: String, maxTokens: Int) async throws -> String {
        let tier = detectTier()
        #if os(macOS)
        if tier == .standard || tier == .enhanced {
            return try await generateLocal(prompt: prompt, maxTokens: maxTokens)
        }
        #endif
        
        let api = SmritiAPI()
        struct QueryRequest: Codable { let query: String; let top_k: Int }
        struct QueryResponse: Codable { struct Answer: Codable { let text: String }; let answer: Answer? }
        
        var request = URLRequest(url: api.baseURL.appendingPathComponent("/v1/query"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(QueryRequest(query: prompt, top_k: 1))
        
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        let res = try JSONDecoder().decode(QueryResponse.self, from: data)
        return res.answer?.text ?? ""
    }
    
    #if os(macOS)
    private func generateLocal(prompt: String, maxTokens: Int) async throws -> String {
        if daemonProcess == nil || daemonProcess?.isRunning == false {
            let process = Process()
            let python311 = "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11"
            let resolvedPython = FileManager.default.fileExists(atPath: python311)
                ? python311
                : "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
            process.executableURL = URL(fileURLWithPath: resolvedPython)
            // The scripts folder is at the repo root. In apps, currentDirectoryPath is repo root or build folder.
            process.arguments = ["scripts/mlx_reasoner.py"]
            process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            
            var env = ProcessInfo.processInfo.environment
            env["TOORI_DATA_DIR"] = ".toori"
            process.environment = env
            
            let pipeIn = Pipe()
            let pipeOut = Pipe()
            process.standardInput = pipeIn
            process.standardOutput = pipeOut
            
            try process.run()
            daemonProcess = process
        }
        
        guard let process = daemonProcess,
              let pipeIn = process.standardInput as? Pipe,
              let pipeOut = process.standardOutput as? Pipe else {
            throw URLError(.cannotConnectToHost)
        }
        
        let input = MLXReasonerInput(prompt: prompt, image_base64: nil, max_tokens: maxTokens)
        let data = try JSONEncoder().encode(input)
        
        let fh = pipeIn.fileHandleForWriting
        try fh.write(contentsOf: data)
        try fh.write(contentsOf: "\n".data(using: .utf8)!)
        
        var receivedData = Data()
        let fhOut = pipeOut.fileHandleForReading
        while let chunk = try fhOut.read(upToCount: 1), let byte = chunk.first {
            receivedData.append(byte)
            if byte == 10 { break } // newline
        }
        
        guard !receivedData.isEmpty, let line = String(data: receivedData, encoding: .utf8) else {
            throw URLError(.badServerResponse)
        }
        
        let result = try JSONDecoder().decode(MLXReasonerOutput.self, from: line.data(using: .utf8)!)
        if let err = result.error { throw NSError(domain: "GemmaError", code: 1, userInfo: [NSLocalizedDescriptionKey: err]) }
        return result.text ?? ""
    }
    #endif

    public func isAvailable() -> Bool {
        return detectTier() == .base || isModelPresent()
    }
}

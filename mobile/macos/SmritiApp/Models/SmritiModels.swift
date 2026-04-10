import Foundation

struct SmritiRecallRequest: Codable {
    let query: String
    let session_id: String
    let top_k: Int
    let person_filter: String?
    let location_filter: String?
    let time_start: Date?
    let time_end: Date?
    let min_confidence: Double
}

struct SmritiRecallResponse: Codable {
    let query: String
    let results: [SmritiRecallItem]
    let total_searched: Int
    let setu_ms: Double
}

struct SmritiRecallItem: Codable, Identifiable, Hashable {
    var id: String { media_id }

    let media_id: String
    let file_path: String
    let thumbnail_path: String
    let setu_score: Double
    let hybrid_score: Double
    let primary_description: String
    let anchor_basis: String
    let depth_stratum: String
    let hallucination_risk: Double
    let created_at: Date
    let person_names: [String]
    let location_name: String?
    let depth_strata_data: SmritiDepthStrata?
    let anchor_matches: [SmritiAnchorMatch]
    let setu_descriptions: [SmritiSetuDescriptionRecord]

    var surpriseProxy: Double { setu_score }
    var thumbnailURL: URL? { thumbnail_path.isEmpty ? nil : URL(fileURLWithPath: thumbnail_path) }
    var fileURL: URL { URL(fileURLWithPath: file_path) }
    var displaySubtitle: String {
        if let location_name, !location_name.isEmpty {
            return location_name
        }
        if let person = person_names.first {
            return person
        }
        return anchor_basis
    }
    var primarySetuText: String {
        setu_descriptions.first?.description.text ?? primary_description
    }
}

struct SmritiDepthStrata: Codable, Hashable {
    let depth_proxy: [[Double]]?
    let foreground_mask: [[Bool]]?
    let midground_mask: [[Bool]]?
    let background_mask: [[Bool]]?
    let confidence: Double?
    let strata_entropy: Double?
}

struct SmritiAnchorMatch: Codable, Hashable {
    let template_name: String
    let confidence: Double
    let patch_indices: [Int]
    let depth_stratum: String
    let open_vocab_label: String?
}

struct SmritiSetuDescriptionRecord: Codable, Hashable {
    let description: SmritiSetuDescription
}

struct SmritiSetuDescription: Codable, Hashable {
    let text: String
    let confidence: Double
    let anchor_basis: String?
    let hallucination_risk: Double?
}

struct SmritiMandalaData: Codable, Hashable {
    let nodes: [SmritiClusterNode]
    let edges: [SmritiClusterEdge]
    let generated_at: Date
}

struct SmritiClusterNode: Codable, Hashable, Identifiable {
    let id: Int
    let label: String
    let media_count: Int
    let centroid: [Double]
    let dominant_depth_stratum: String?
    let temporal_span_days: Double?
}

struct SmritiClusterEdge: Codable, Hashable {
    let source: Int
    let target: Int
    let similarity: Double
}

struct SmritiPersonJournal: Codable, Hashable {
    let person_name: String
    let entries: [SmritiPersonJournalEntry]
    let count: Int
}

struct SmritiPersonJournalEntry: Codable, Hashable {
    let media_id: String
    let file_path: String
    let ingested_at: Date
}

struct StorageUsageReport: Codable, Hashable {
    let smriti_data_dir: String
    let total_media_count: Int
    let indexed_count: Int
    let pending_count: Int
    let failed_count: Int
    let frames_bytes: Int
    let thumbs_bytes: Int
    let smriti_db_bytes: Int
    let faiss_index_bytes: Int
    let templates_bytes: Int
    let total_bytes: Int
    let total_human: String
    let max_storage_gb: Double
    let budget_pct: Double
    let budget_warning: Bool
    let budget_critical: Bool
}

struct WatchFolderStatus: Codable, Hashable, Identifiable {
    var id: String { path }

    let path: String
    let exists: Bool
    let is_accessible: Bool
    let media_count_total: Int
    let media_count_indexed: Int
    let media_count_pending: Int
    let watchdog_active: Bool
    let last_event_at: Date?
    let error: String?
}

struct RuntimeSnapshotResponse: Codable, Hashable {
    let session_id: String
    let current: SceneState?
    let entity_tracks: [EntityTrack]
    let latest_jepa_tick: JEPATickPayload?
    let world_model_status: WorldModelStatus
    let observations: [ObservationSummary]
    let observation_count: Int
}

struct SceneState: Codable, Hashable {
    let id: String
    let session_id: String
    let created_at: Date
    let observation_id: String
}

struct ObservationSummary: Codable, Hashable, Identifiable {
    let id: String
    let session_id: String
    let created_at: Date
    let world_state_id: String?
    let observation_kind: String
    let image_path: String
    let thumbnail_path: String
    let width: Int
    let height: Int
    let summary: String?
    let source_query: String?
    let tags: [String]
    let confidence: Double
    let novelty: Double
    let providers: [String]
}

struct EntityTrack: Codable, Hashable, Identifiable {
    let id: String
    let session_id: String
    let label: String
    let status: String
    let first_seen_at: Date
    let last_seen_at: Date
    let first_observation_id: String
    let last_observation_id: String
    let observations: [String]
    let visibility_streak: Int
    let occlusion_count: Int
    let reidentification_count: Int
    let persistence_confidence: Double
    let continuity_score: Double
    let last_similarity: Double
    let status_history: [String]
}

struct JEPATickPayload: Codable, Hashable {
    let timestamp_ms: Int
    let mean_energy: Double
    let energy_std: Double
    let surprise_score: Double?
}

struct WorldModelStatus: Codable, Hashable {
    let encoder_type: String
    let model_id: String
    let model_loaded: Bool
    let device: String
    let n_frames: Int
    let test_mode: Bool
    let total_ticks: Int
    let mean_prediction_error: Double?
    let mean_surprise_score: Double?
    let configured_encoder: String
    let last_tick_encoder_type: String
    let degraded: Bool
    let degrade_reason: String?
    let degrade_stage: String?
    let active_backend: String
    let native_ready: Bool
    let preflight_status: String
    let last_failure_at: Date?
    let crash_fingerprint: String?
    let native_process_state: String
    let last_native_exit_code: Int?
    let last_native_signal: Int?
    let retryable_native_failure: Bool
    let telescope_test: String
}

struct EventMessage: Codable, Hashable, Identifiable {
    var id: String { "\(type)-\(timestamp.timeIntervalSince1970)" }

    let type: String
    let timestamp: Date
    let payload: [String: JSONValue]
}

struct SmritiIngestResponse: Codable, Hashable {
    let queued: Int
    let status: String
}

struct SmritiTagPersonRequest: Codable {
    let media_id: String
    let person_name: String
    let confirmed: Bool
}

struct SmritiTagPersonResponse: Codable, Hashable {
    let person_id: String
    let propagated_to: Int
}

struct LivingLensTickRequest: Codable {
    let image_base64: String?
    let file_path: String?
    let session_id: String
    let query: String?
    let decode_mode: String
    let top_k: Int?
    let time_window_s: Int?
    let tags: [String]
    let proof_mode: String
}

struct LivingLensTickResponse: Codable, Hashable {
    let scene_state: SceneState
    let entity_tracks: [EntityTrack]
}

enum JSONValue: Codable, Hashable {
    case string(String)
    case number(Double)
    case integer(Int)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .integer(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else {
            self = .array(try container.decode([JSONValue].self))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .integer(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }
}

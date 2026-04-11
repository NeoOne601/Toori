import Foundation

public struct SmritiRecallRequest: Codable, Hashable {
    public let query: String
    public let session_id: String
    public let top_k: Int
    public let person_filter: String?
    public let location_filter: String?
    public let time_start: Date?
    public let time_end: Date?
    public let min_confidence: Double

    public init(
        query: String,
        session_id: String,
        top_k: Int,
        person_filter: String?,
        location_filter: String?,
        time_start: Date?,
        time_end: Date?,
        min_confidence: Double
    ) {
        self.query = query
        self.session_id = session_id
        self.top_k = top_k
        self.person_filter = person_filter
        self.location_filter = location_filter
        self.time_start = time_start
        self.time_end = time_end
        self.min_confidence = min_confidence
    }
}

public struct SmritiRecallResponse: Codable, Hashable {
    public let query: String
    public let results: [SmritiRecallItem]
    public let total_searched: Int
    public let setu_ms: Double
}

public struct SmritiRecallItem: Codable, Identifiable, Hashable {
    public var id: String { media_id }

    public let media_id: String
    public let file_path: String
    public let thumbnail_path: String
    public let setu_score: Double
    public let hybrid_score: Double
    public let primary_description: String
    public let anchor_basis: String
    public let depth_stratum: String
    public let hallucination_risk: Double
    public let created_at: Date
    public let person_names: [String]
    public let location_name: String?
    public let depth_strata_data: SmritiDepthStrata?
    public let anchor_matches: [SmritiAnchorMatch]
    public let setu_descriptions: [SmritiSetuDescriptionRecord]

    public init(
        media_id: String,
        file_path: String,
        thumbnail_path: String,
        setu_score: Double,
        hybrid_score: Double,
        primary_description: String,
        anchor_basis: String,
        depth_stratum: String,
        hallucination_risk: Double,
        created_at: Date,
        person_names: [String],
        location_name: String?,
        depth_strata_data: SmritiDepthStrata?,
        anchor_matches: [SmritiAnchorMatch],
        setu_descriptions: [SmritiSetuDescriptionRecord]
    ) {
        self.media_id = media_id
        self.file_path = file_path
        self.thumbnail_path = thumbnail_path
        self.setu_score = setu_score
        self.hybrid_score = hybrid_score
        self.primary_description = primary_description
        self.anchor_basis = anchor_basis
        self.depth_stratum = depth_stratum
        self.hallucination_risk = hallucination_risk
        self.created_at = created_at
        self.person_names = person_names
        self.location_name = location_name
        self.depth_strata_data = depth_strata_data
        self.anchor_matches = anchor_matches
        self.setu_descriptions = setu_descriptions
    }

    public var effectiveSurpriseScore: Double {
        setu_score
    }

    public var surpriseProxy: Double { setu_score }
    public var thumbnailURL: URL? { thumbnail_path.isEmpty ? nil : URL(fileURLWithPath: thumbnail_path) }
    public var fileURL: URL { URL(fileURLWithPath: file_path) }

    public var displaySubtitle: String {
        if let location_name, !location_name.isEmpty {
            return location_name
        }
        if let person = person_names.first {
            return person
        }
        return anchor_basis
    }

    public var primarySetuText: String {
        setu_descriptions.first?.description.text ?? primary_description
    }
}

public struct SmritiDepthStrata: Codable, Hashable {
    public let depth_proxy: [[Double]]?
    public let foreground_mask: [[Bool]]?
    public let midground_mask: [[Bool]]?
    public let background_mask: [[Bool]]?
    public let confidence: Double?
    public let strata_entropy: Double?
}

public struct SmritiAnchorMatch: Codable, Hashable {
    public let template_name: String
    public let confidence: Double
    public let patch_indices: [Int]
    public let depth_stratum: String
    public let open_vocab_label: String?
}

public struct SmritiSetuDescriptionRecord: Codable, Hashable {
    public let description: SmritiSetuDescription
}

public struct SmritiSetuDescription: Codable, Hashable {
    public let text: String
    public let confidence: Double
    public let anchor_basis: String?
    public let hallucination_risk: Double?
}

public struct SmritiMandalaData: Codable, Hashable {
    public let nodes: [SmritiClusterNode]
    public let edges: [SmritiClusterEdge]
    public let generated_at: Date

    public init(nodes: [SmritiClusterNode], edges: [SmritiClusterEdge], generated_at: Date) {
        self.nodes = nodes
        self.edges = edges
        self.generated_at = generated_at
    }
}

public struct SmritiClusterNode: Codable, Hashable, Identifiable {
    public let id: Int
    public let label: String
    public let media_count: Int
    public let centroid: [Double]
    public let dominant_depth_stratum: String?
    public let temporal_span_days: Double?

    public init(
        id: Int,
        label: String,
        media_count: Int,
        centroid: [Double],
        dominant_depth_stratum: String?,
        temporal_span_days: Double?
    ) {
        self.id = id
        self.label = label
        self.media_count = media_count
        self.centroid = centroid
        self.dominant_depth_stratum = dominant_depth_stratum
        self.temporal_span_days = temporal_span_days
    }
}

public struct SmritiClusterEdge: Codable, Hashable {
    public let source: Int
    public let target: Int
    public let similarity: Double

    public init(source: Int, target: Int, similarity: Double) {
        self.source = source
        self.target = target
        self.similarity = similarity
    }
}

public struct SmritiPersonJournal: Codable, Hashable {
    public let person_name: String
    public let entries: [SmritiPersonJournalEntry]
    public let count: Int
}

public struct SmritiPersonJournalEntry: Codable, Hashable {
    public let media_id: String
    public let file_path: String
    public let ingested_at: Date
}

public struct StorageUsageReport: Codable, Hashable {
    public let smriti_data_dir: String
    public let total_media_count: Int
    public let indexed_count: Int
    public let pending_count: Int
    public let failed_count: Int
    public let frames_bytes: Int
    public let thumbs_bytes: Int
    public let smriti_db_bytes: Int
    public let faiss_index_bytes: Int
    public let templates_bytes: Int
    public let total_bytes: Int
    public let total_human: String
    public let max_storage_gb: Double
    public let budget_pct: Double
    public let budget_warning: Bool
    public let budget_critical: Bool
}

public struct WatchFolderStatus: Codable, Hashable, Identifiable {
    public var id: String { path }

    public let path: String
    public let exists: Bool
    public let is_accessible: Bool
    public let media_count_total: Int
    public let media_count_indexed: Int
    public let media_count_pending: Int
    public let watchdog_active: Bool
    public let last_event_at: Date?
    public let error: String?
}

public struct RuntimeSnapshotResponse: Codable, Hashable {
    public let session_id: String
    public let current: SceneState?
    public let entity_tracks: [EntityTrack]
    public let latest_jepa_tick: JEPATickPayload?
    public let world_model_status: WorldModelStatus
    public let observations: [ObservationSummary]
    public let observation_count: Int
    public let scene_graph: SceneGraphPayload?
}

public struct SceneState: Codable, Hashable {
    public let id: String
    public let session_id: String
    public let created_at: Date
    public let observation_id: String
}

public struct ObservationSummary: Codable, Hashable, Identifiable, Sendable {
    public let id: String
    public let session_id: String
    public let created_at: Date
    public let world_state_id: String?
    public let observation_kind: String
    public let image_path: String
    public let thumbnail_path: String
    public let width: Int
    public let height: Int
    public let summary: String?
    public let source_query: String?
    public let tags: [String]
    public let confidence: Double
    public let novelty: Double
    public let providers: [String]
    public let metadata: [String: JSONValue]?

    public init(
        id: String,
        session_id: String,
        created_at: Date,
        world_state_id: String?,
        observation_kind: String,
        image_path: String,
        thumbnail_path: String,
        width: Int,
        height: Int,
        summary: String?,
        source_query: String?,
        tags: [String],
        confidence: Double,
        novelty: Double,
        providers: [String],
        metadata: [String: JSONValue]? = nil
    ) {
        self.id = id
        self.session_id = session_id
        self.created_at = created_at
        self.world_state_id = world_state_id
        self.observation_kind = observation_kind
        self.image_path = image_path
        self.thumbnail_path = thumbnail_path
        self.width = width
        self.height = height
        self.summary = summary
        self.source_query = source_query
        self.tags = tags
        self.confidence = confidence
        self.novelty = novelty
        self.providers = providers
        self.metadata = metadata
    }

    public var imageURL: URL? { image_path.isEmpty ? nil : URL(fileURLWithPath: image_path) }
    public var thumbnailURL: URL? { thumbnail_path.isEmpty ? nil : URL(fileURLWithPath: thumbnail_path) }

    public var worldModelMetrics: ObservationWorldModelMetrics? {
        metadata?["world_model"]?.decode(ObservationWorldModelMetrics.self)
    }

    public var effectiveSurpriseScore: Double {
        novelty
    }

    public var surpriseProxy: Double {
        worldModelMetrics?.surprise_score ?? novelty
    }

    public var displayLabel: String {
        if let summary, !summary.isEmpty {
            return summary
        }
        if let tag = tags.first, !tag.isEmpty {
            return tag
        }
        if let source_query, !source_query.isEmpty {
            return source_query
        }
        return "Recent memory"
    }

    public static func fromEventPayload(_ payload: [String: JSONValue]) -> ObservationSummary? {
        guard let rawObservation = payload["observation"] else {
            return nil
        }
        return rawObservation.decode(ObservationSummary.self)
    }
}

public struct ObservationSummariesResponse: Codable, Hashable {
    public let observations: [ObservationSummary]
}

public struct ObservationWorldModelMetrics: Codable, Hashable {
    public let prediction_consistency: Double?
    public let surprise_score: Double?
    public let energy_activation_score: Double?
    public let temporal_continuity_score: Double?
    public let persistence_confidence: Double?
}

public struct EntityTrack: Codable, Hashable, Identifiable {
    public let id: String
    public let session_id: String
    public let label: String
    public let status: String
    public let first_seen_at: Date
    public let last_seen_at: Date
    public let first_observation_id: String
    public let last_observation_id: String
    public let observations: [String]
    public let visibility_streak: Int
    public let occlusion_count: Int
    public let reidentification_count: Int
    public let persistence_confidence: Double
    public let continuity_score: Double
    public let last_similarity: Double
    public let status_history: [String]
}

public struct JEPATickPayload: Codable, Hashable {
    public let timestamp_ms: Int
    public let mean_energy: Double
    public let energy_std: Double
    public let surprise_score: Double?
}

public struct WorldModelStatus: Codable, Hashable {
    public let encoder_type: String
    public let model_id: String
    public let model_loaded: Bool
    public let device: String
    public let n_frames: Int
    public let test_mode: Bool
    public let total_ticks: Int
    public let mean_prediction_error: Double?
    public let mean_surprise_score: Double?
    public let configured_encoder: String
    public let last_tick_encoder_type: String
    public let degraded: Bool
    public let degrade_reason: String?
    public let degrade_stage: String?
    public let active_backend: String
    public let native_ready: Bool
    public let preflight_status: String
    public let last_failure_at: Date?
    public let crash_fingerprint: String?
    public let native_process_state: String
    public let last_native_exit_code: Int?
    public let last_native_signal: Int?
    public let retryable_native_failure: Bool
    public let telescope_test: String
}

public struct SceneGraphPayload: Codable, Hashable {
    public let nodes: [SceneGraphNode]
    public let edges: [SceneGraphEdge]
}

public struct SceneGraphNode: Codable, Hashable, Identifiable {
    public let id: String
    public let label: String
    public let status: String?
    public let depth_confidence: Double?
    public let label_source: String?
    public let label_evidence: String?
}

public struct SceneGraphEdge: Codable, Hashable {
    public let source: String
    public let target: String
    public let label: String?
}

public struct EventMessage: Codable, Hashable, Identifiable, Sendable {
    public var id: String { "\(type)-\(timestamp.timeIntervalSince1970)" }

    public let type: String
    public let timestamp: Date
    public let payload: [String: JSONValue]
}

public struct SmritiIngestResponse: Codable, Hashable {
    public let queued: Int
    public let status: String
}

public struct SmritiTagPersonRequest: Codable, Hashable {
    public let media_id: String
    public let person_name: String
    public let confirmed: Bool

    public init(media_id: String, person_name: String, confirmed: Bool) {
        self.media_id = media_id
        self.person_name = person_name
        self.confirmed = confirmed
    }
}

public struct SmritiTagPersonResponse: Codable, Hashable {
    public let person_id: String
    public let propagated_to: Int
}

public struct LivingLensTickRequest: Codable, Hashable {
    public let image_base64: String?
    public let file_path: String?
    public let session_id: String
    public let query: String?
    public let decode_mode: String
    public let top_k: Int?
    public let time_window_s: Int?
    public let tags: [String]
    public let proof_mode: String

    public init(
        image_base64: String?,
        file_path: String?,
        session_id: String,
        query: String?,
        decode_mode: String,
        top_k: Int?,
        time_window_s: Int?,
        tags: [String],
        proof_mode: String
    ) {
        self.image_base64 = image_base64
        self.file_path = file_path
        self.session_id = session_id
        self.query = query
        self.decode_mode = decode_mode
        self.top_k = top_k
        self.time_window_s = time_window_s
        self.tags = tags
        self.proof_mode = proof_mode
    }
}

public struct LivingLensTickResponse: Codable, Hashable {
    public let scene_state: SceneState
    public let entity_tracks: [EntityTrack]
}

public struct AudioQueryRequest: Codable, Hashable {
    public let audio_base64: String
    public let sample_rate: Int
    public let top_k: Int
    public let session_id: String?
    public let depth_stratum: String?
    public let person_filter: String?
    public let confidence_min: Double
    public let cross_modal: Bool

    public init(
        audio_base64: String,
        sample_rate: Int,
        top_k: Int,
        session_id: String?,
        depth_stratum: String?,
        person_filter: String?,
        confidence_min: Double,
        cross_modal: Bool
    ) {
        self.audio_base64 = audio_base64
        self.sample_rate = sample_rate
        self.top_k = top_k
        self.session_id = session_id
        self.depth_stratum = depth_stratum
        self.person_filter = person_filter
        self.confidence_min = confidence_min
        self.cross_modal = cross_modal
    }
}

public struct AudioQueryResponse: Codable, Hashable {
    public let results: [AudioQueryResult]
    public let query_audio_energy: Double
    public let index_size: Int
    public let latency_ms: Double
    public let encoder: String
}

public struct AudioQueryResult: Codable, Hashable, Identifiable {
    public var id: String { media_id }

    public let media_id: String
    public let audio_score: Double
    public let rank: Int
    public let thumbnail_path: String?
    public let setu_descriptions: [SmritiSetuDescriptionRecord]
    public let audio_energy: Double?
    public let audio_duration_seconds: Double?
    public let gemma4_narration: String?

    public var thumbnailURL: URL? {
        guard let thumbnail_path, !thumbnail_path.isEmpty else { return nil }
        return URL(fileURLWithPath: thumbnail_path)
    }

    public var primaryText: String {
        if let gemma4_narration, !gemma4_narration.isEmpty {
            return gemma4_narration
        }
        return setu_descriptions.first?.description.text ?? "Matched by sound"
    }
}

public struct SmritiPhotoIngestProgress: Hashable, Sendable {
    public let importedCount: Int
}

public struct DetailMetricTriplet: Hashable {
    public let prediction_consistency: Double
    public let temporal_continuity_score: Double
    public let surprise_score: Double
}

public struct DetailEntityPill: Identifiable, Hashable {
    public var id: String { "\(systemImage)-\(label)" }
    public let label: String
    public let systemImage: String
}

public enum SelectedMemory: Identifiable, Hashable {
    case observation(ObservationSummary)
    case recall(SmritiRecallItem)
    case audio(AudioQueryResult)

    public var id: String {
        switch self {
        case .observation(let observation):
            return "observation-\(observation.id)"
        case .recall(let item):
            return "recall-\(item.id)"
        case .audio(let result):
            return "audio-\(result.id)"
        }
    }

    public var title: String {
        switch self {
        case .observation(let observation):
            return observation.displayLabel
        case .recall(let item):
            return item.primary_description
        case .audio(let result):
            return result.primaryText
        }
    }

    public var subtitle: String {
        switch self {
        case .observation(let observation):
            return observation.created_at.formatted(date: .abbreviated, time: .shortened)
        case .recall(let item):
            return item.displaySubtitle
        case .audio(let result):
            return "Matched by sound • \(result.audio_score.formatted(.number.precision(.fractionLength(2))))"
        }
    }

    public var descriptionText: String {
        switch self {
        case .observation(let observation):
            return observation.summary ?? "Smriti stored this observation from the live stream."
        case .recall(let item):
            return item.primarySetuText
        case .audio(let result):
            return result.primaryText
        }
    }

    public var heroPath: String? {
        switch self {
        case .observation(let observation):
            return observation.image_path
        case .recall(let item):
            return item.file_path
        case .audio(let result):
            return result.thumbnail_path
        }
    }

    public var fallbackThumbnailPath: String? {
        switch self {
        case .observation(let observation):
            return observation.thumbnail_path
        case .recall(let item):
            return item.thumbnail_path
        case .audio(let result):
            return result.thumbnail_path
        }
    }

    public var metricTriplet: DetailMetricTriplet {
        switch self {
        case .observation(let observation):
            let metrics = observation.worldModelMetrics
            return DetailMetricTriplet(
                prediction_consistency: metrics?.prediction_consistency ?? observation.confidence,
                temporal_continuity_score: metrics?.temporal_continuity_score ?? max(0.18, 1.0 - observation.novelty),
                surprise_score: metrics?.surprise_score ?? observation.novelty
            )
        case .recall(let item):
            return DetailMetricTriplet(
                prediction_consistency: min(max(item.hybrid_score, 0), 1),
                temporal_continuity_score: min(max(1.0 - item.hallucination_risk, 0), 1),
                surprise_score: min(max(item.surpriseProxy, 0), 1)
            )
        case .audio(let result):
            let score = min(max(result.audio_score, 0), 1)
            return DetailMetricTriplet(
                prediction_consistency: score,
                temporal_continuity_score: max(0.12, score * 0.82),
                surprise_score: score
            )
        }
    }

    public var entityPills: [DetailEntityPill] {
        switch self {
        case .observation(let observation):
            return observation.tags.map { DetailEntityPill(label: $0, systemImage: "tag") }
        case .recall(let item):
            return item.person_names.map { DetailEntityPill(label: $0, systemImage: "eye") }
                + item.anchor_matches.map {
                    DetailEntityPill(label: $0.open_vocab_label ?? $0.template_name, systemImage: "questionmark")
                }
        case .audio:
            return []
        }
    }

    public var surpriseValue: Double {
        switch self {
        case .observation(let observation):
            return observation.surpriseProxy
        case .recall(let item):
            return item.surpriseProxy
        case .audio(let result):
            return result.audio_score
        }
    }

    public var shareText: String {
        switch self {
        case .observation(let observation):
            return observation.displayLabel
        case .recall(let item):
            return item.primarySetuText
        case .audio(let result):
            return result.primaryText
        }
    }
}

public enum PulseOrbSlot: Identifiable, Hashable {
    case observation(ObservationSummary)
    case placeholder(Int)

    public var id: String {
        switch self {
        case .observation(let observation):
            return observation.id
        case .placeholder(let index):
            return "placeholder-\(index)"
        }
    }
}

public enum JSONValue: Codable, Hashable, Sendable {
    case string(String)
    case number(Double)
    case integer(Int)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    public init(from decoder: Decoder) throws {
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

    public func encode(to encoder: Encoder) throws {
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

    public func decode<T: Decodable>(_ type: T.Type) -> T? {
        guard JSONSerialization.isValidJSONObject(foundationValue) else {
            return nil
        }
        guard let data = try? JSONSerialization.data(withJSONObject: foundationValue) else {
            return nil
        }
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
        return try? decoder.decode(type, from: data)
    }

    public var foundationValue: Any {
        switch self {
        case .string(let value):
            return value
        case .number(let value):
            return value
        case .integer(let value):
            return value
        case .bool(let value):
            return value
        case .object(let value):
            return value.mapValues(\.foundationValue)
        case .array(let value):
            return value.map(\.foundationValue)
        case .null:
            return NSNull()
        }
    }
}

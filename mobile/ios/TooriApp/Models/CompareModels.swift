import Foundation

// MARK: - Compare (Sprint B1)

struct CompareRequest: Codable {
    let image_base64_a: String
    let image_base64_b: String
    let session_id: String
    let query: String?
    let decode_mode: String

    init(imageA: Data, imageB: Data, sessionId: String, query: String?) {
        self.image_base64_a = imageA.base64EncodedString()
        self.image_base64_b = imageB.base64EncodedString()
        self.session_id = sessionId
        self.query = query
        self.decode_mode = query != nil ? "auto" : "off"
    }
}

struct CompareResponse: Codable {
    let observation_a: Observation
    let observation_b: Observation
    let semantic_distance: Double
    let changed_regions: [ChangedRegion]
    let grounded_diff: String
    let confidence_label: String
    let similarity_pct: Int
    let change_summary: String
}

struct ChangedRegion: Codable, Identifiable {
    var id: String { label }
    let label: String
    let bbox: [String: Double]?
    let depth_stratum: String
    let is_novel: Bool
}

// MARK: - Calibration (Sprint D1)

struct CalibratePayload: Codable {
    let session_id: String
    let label: String
    let real_width_cm: Double?
    let bbox: [String: Double]?
}

struct CalibrationResponse: Codable {
    let session_id: String
    let scale_px_per_cm: Double
    let calibrated_at: String
    let anchor_label: String
    let message: String
}

// MARK: - Known object size presets (mirrors STANDARD_OBJECT_SIZES_CM in service.py)

enum KnownObjectPreset: String, CaseIterable, Identifiable {
    case door          = "door"
    case a4Sheet       = "a4_sheet"
    case creditCard    = "credit_card"
    case deskSurface   = "desk_surface"
    case laptop        = "laptop"
    case chairSeat     = "chair_seat"
    case iphone        = "iphone"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .door:         return "Standard Door (80 cm)"
        case .a4Sheet:      return "A4 Paper (21 cm)"
        case .creditCard:   return "Credit Card (8.5 cm)"
        case .deskSurface:  return "Desk Surface (120 cm)"
        case .laptop:       return "Laptop (33 cm)"
        case .chairSeat:    return "Chair Seat (45 cm)"
        case .iphone:       return "iPhone (7.1 cm)"
        }
    }

    var realWidthCm: Double {
        switch self {
        case .door:         return 80.0
        case .a4Sheet:      return 21.0
        case .creditCard:   return 8.56
        case .deskSurface:  return 120.0
        case .laptop:       return 33.0
        case .chairSeat:    return 45.0
        case .iphone:       return 7.1
        }
    }
}

// MARK: - Text-only analyze payload (for multi-turn follow-ups)

struct TextAnalyzePayload: Codable {
    let session_id: String
    let query: String
    let decode_mode: String
}

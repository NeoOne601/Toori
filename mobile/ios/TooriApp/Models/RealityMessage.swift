import Foundation
import SwiftUI

// MARK: - RealityMessage
/// A single turn in the Reality Intelligence conversational interface.
/// Carries either a user query or a full JEPA-grounded response.
struct RealityMessage: Identifiable, Equatable {
    let id = UUID().uuidString
    let role: Role
    let text: String?
    let image: UIImage?

    // ECGD-grounded fields (populated only on assistant messages with analysis)
    var groundedSummary: String?
    var confidence: Double?
    var confidenceLabel: String?
    var entities: [String] = []
    var similarScenes: [String] = []

    // TPDS mask data — if available from the backend
    var depthStrata: RealityDepthStrata?
    var isError: Bool = false

    enum Role: String {
        case user
        case assistant
    }

    static func == (lhs: RealityMessage, rhs: RealityMessage) -> Bool {
        lhs.id == rhs.id
    }
}

// MARK: - RealityDepthStrata
/// Tri-Planar Depth Separation masks, decoded from the backend's depth payload.
struct RealityDepthStrata {
    let foreground_mask: [[Bool]]?
    let midground_mask:  [[Bool]]?
    let background_mask: [[Bool]]?
    let confidence: Double?
}

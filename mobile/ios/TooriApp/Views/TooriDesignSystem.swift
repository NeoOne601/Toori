import SwiftUI

// MARK: - Toori Color Palette
// Deep, cinematic dark mode — Reality Intelligence visual language
extension Color {
    /// Warm amber — the signature color of JEPA-grounded Reality Intelligence
    static let tooriAmber   = Color(red: 0.92, green: 0.68, blue: 0.28)
    /// Soft indigo — the secondary accent for spatial / depth UI elements
    static let tooriIndigo  = Color(red: 0.45, green: 0.38, blue: 0.82)
    /// Deep cosmic navy — the primary background canvas
    static let tooriCanvas  = Color(red: 0.03, green: 0.05, blue: 0.10)
    /// Subtle separator
    static let tooriStroke  = Color.white.opacity(0.08)
    /// Surface glass overlay
    static let tooriSurface = Color.white.opacity(0.05)
    /// Grounded green — high-confidence ECGD result
    static let tooriGrounded = Color(red: 0.28, green: 0.82, blue: 0.46)
    /// Warning red — low-confidence or uncertain result
    static let tooriUncertain = Color(red: 0.88, green: 0.36, blue: 0.36)
}

// MARK: - Toori Animation Suite
extension Animation {
    /// The signature spring: snappy, never bouncy
    static let tooriSpring = Animation.spring(response: 0.38, dampingFraction: 0.72)
    /// Gentle reveal for content
    static let tooriReveal = Animation.easeOut(duration: 0.45)
    /// Micro-interaction tap spring
    static let tooriTap = Animation.spring(response: 0.25, dampingFraction: 0.65)
}

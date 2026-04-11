import SwiftUI

public extension Color {
    static let smritiAccent = Color(red: 0.4196, green: 0.3607, blue: 0.9058)
    static let smritiTeal = Color(red: 0.235, green: 0.765, blue: 0.765)
    static let smritiDivider = Color.white.opacity(0.12)
    static let smritiSurface = Color.white.opacity(0.06)
    static let smritiStroke = Color.white.opacity(0.08)
    static let smritiCanvas = Color.black.opacity(0.96)
    static let smritiBadgeFill = Color.smritiAccent.opacity(0.15)
    static let smritiShimmerBase = Color.primary.opacity(0.08)
    static let smritiShimmerHighlight = Color.primary.opacity(0.18)
}

public extension Animation {
    static let smritiSpring = Animation.spring(response: 0.38, dampingFraction: 0.72)

    static func smritiStagger(index: Int) -> Double {
        Double(index) * 0.04
    }
}

public func surpriseColor(_ score: Double) -> Color {
    Color(hue: 0.47 - (score * 0.15),
          saturation: 0.70 + (score * 0.15),
          brightness: 0.75)
}

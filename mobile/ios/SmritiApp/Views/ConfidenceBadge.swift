import SwiftUI

/// A reusable badge displaying ECGD confidence level.
/// Shows color-coded confidence from the epistemic confidence gate.
struct ConfidenceBadge: View {
    let confidence: Double
    let label: String?

    init(confidence: Double, label: String? = nil) {
        self.confidence = confidence
        self.label = label
    }

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(badgeColor)
                .frame(width: 8, height: 8)
                .shadow(color: badgeColor.opacity(0.6), radius: 4)

            Text(displayLabel)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.white.opacity(0.92))

            Text(confidenceText)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.white.opacity(0.6))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            Capsule(style: .continuous)
                .fill(badgeColor.opacity(0.15))
        )
        .overlay(
            Capsule(style: .continuous)
                .stroke(badgeColor.opacity(0.4), lineWidth: 0.8)
        )
    }

    private var badgeColor: Color {
        if confidence > 0.8 {
            return Color(red: 0.28, green: 0.82, blue: 0.46) // green — grounded
        } else if confidence > 0.5 {
            return Color(red: 0.92, green: 0.68, blue: 0.28) // amber — likely
        } else {
            return Color(red: 0.88, green: 0.36, blue: 0.36) // red — uncertain
        }
    }

    private var displayLabel: String {
        if let label { return label }
        if confidence > 0.8 { return "Grounded" }
        if confidence > 0.5 { return "Likely" }
        return "Uncertain"
    }

    private var confidenceText: String {
        "\(Int(confidence * 100))%"
    }
}

/// Compact inline confidence indicator for list items.
struct ConfidenceDot: View {
    let confidence: Double

    var body: some View {
        Circle()
            .fill(dotColor)
            .frame(width: 6, height: 6)
            .shadow(color: dotColor.opacity(0.5), radius: 2)
    }

    private var dotColor: Color {
        if confidence > 0.8 {
            return Color(red: 0.28, green: 0.82, blue: 0.46)
        } else if confidence > 0.5 {
            return Color(red: 0.92, green: 0.68, blue: 0.28)
        } else {
            return Color(red: 0.88, green: 0.36, blue: 0.36)
        }
    }
}

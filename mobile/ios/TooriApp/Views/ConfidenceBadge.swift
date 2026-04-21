import SwiftUI

// MARK: - ConfidenceBadge
/// ECGD confidence badge. Color-coded from the Epistemic Confidence Gate output.
/// Grounded (>80%) → green | Likely (>50%) → amber | Uncertain → red
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
                .frame(width: 7, height: 7)
                .shadow(color: badgeColor.opacity(0.7), radius: 5)

            Text(displayLabel)
                .font(.system(size: 11, weight: .bold))
                .foregroundStyle(.white.opacity(0.95))

            Text(confidenceText)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.white.opacity(0.55))
        }
        .padding(.horizontal, 11)
        .padding(.vertical, 7)
        .background(
            Capsule(style: .continuous)
                .fill(badgeColor.opacity(0.13))
        )
        .overlay(
            Capsule(style: .continuous)
                .stroke(badgeColor.opacity(0.35), lineWidth: 0.8)
        )
    }

    private var badgeColor: Color {
        if confidence > 0.8  { return .tooriGrounded }
        if confidence > 0.5  { return .tooriAmber }
        return .tooriUncertain
    }

    private var displayLabel: String {
        if let label { return label }
        if confidence > 0.8  { return "Grounded" }
        if confidence > 0.5  { return "Likely" }
        return "Uncertain"
    }

    private var confidenceText: String { "\(Int(confidence * 100))%" }
}

// MARK: - ConfidenceDot
/// Compact inline dot indicator for list views.
struct ConfidenceDot: View {
    let confidence: Double

    var body: some View {
        Circle()
            .fill(dotColor)
            .frame(width: 6, height: 6)
            .shadow(color: dotColor.opacity(0.55), radius: 3)
    }

    private var dotColor: Color {
        if confidence > 0.8 { return .tooriGrounded }
        if confidence > 0.5 { return .tooriAmber }
        return .tooriUncertain
    }
}

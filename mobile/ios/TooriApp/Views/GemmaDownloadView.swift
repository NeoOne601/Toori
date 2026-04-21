import SwiftUI

// MARK: - GemmaModelInfo
/// Describes a local on-device Gemma variant available through the MLX reasoning backend.
struct GemmaModelInfo: Identifiable {
    let id: String          // HuggingFace model ID used by runtime settings
    let displayName: String
    let parameterSize: String
    let ramRequirement: String
    let isDefault: Bool
}

let gemmaModels: [GemmaModelInfo] = [
    GemmaModelInfo(
        id: "mlx-community/gemma-4-e2b-it-4bit",
        displayName: "Gemma 4 (2B)",
        parameterSize: "2 Billion",
        ramRequirement: "~1.8 GB",
        isDefault: true
    ),
    GemmaModelInfo(
        id: "mlx-community/gemma-4-e4b-it-4bit",
        displayName: "Gemma 4 (4B)",
        parameterSize: "4 Billion",
        ramRequirement: "~3.4 GB",
        isDefault: false
    )
]

// MARK: - GemmaDownloadView
/// On-device Gemma model selection and download status view.
/// Shown when the selected MLX model is not yet present on the iMac backend.
public struct GemmaDownloadView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var selectedModelId = "mlx-community/gemma-4-e2b-it-4bit"

    public init() {}

    public var body: some View {
        ZStack {
            Color.tooriCanvas.ignoresSafeArea()
            RadialGradient(
                colors: [Color.tooriAmber.opacity(0.06), .clear],
                center: .top, startRadius: 30, endRadius: 400
            ).ignoresSafeArea()

            VStack(spacing: 32) {
                // Header
                VStack(spacing: 12) {
                    ZStack {
                        Circle().fill(Color.tooriAmber.opacity(0.10))
                            .frame(width: 80, height: 80).blur(radius: 10)
                        Image(systemName: "brain.filled.head.profile")
                            .font(.system(size: 38))
                            .foregroundStyle(Color.tooriAmber)
                    }

                    Text("On-Device Reasoning")
                        .font(.system(size: 22, weight: .bold))
                        .foregroundStyle(.white)

                    Text("Gemma 4 runs entirely on your iMac.\nNo data leaves your local network.")
                        .font(.system(size: 14))
                        .foregroundStyle(.white.opacity(0.5))
                        .multilineTextAlignment(.center)
                        .lineSpacing(3)
                }

                // Model selection cards
                VStack(spacing: 10) {
                    ForEach(gemmaModels) { model in
                        ModelSelectionCard(
                            model: model,
                            isSelected: selectedModelId == model.id
                        ) {
                            withAnimation(.tooriTap) { selectedModelId = model.id }
                        }
                    }
                }

                // Info note
                HStack(spacing: 8) {
                    Image(systemName: "info.circle")
                        .font(.system(size: 12))
                        .foregroundStyle(Color.tooriAmber.opacity(0.7))
                    Text("Download runs on your iMac via the Toori runtime.")
                        .font(.system(size: 12))
                        .foregroundStyle(.white.opacity(0.45))
                }

                // CTA
                Button { dismiss() } label: {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                        Text("Continue with \(gemmaModels.first(where: { $0.id == selectedModelId })?.displayName ?? "Selected Model")")
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(Color.tooriAmber)
                    .foregroundStyle(Color.tooriCanvas)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                }
            }
            .padding(28)
        }
        .preferredColorScheme(.dark)
    }
}

// MARK: - ModelSelectionCard

private struct ModelSelectionCard: View {
    let model: GemmaModelInfo
    let isSelected: Bool
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 14) {
                // Radio indicator
                ZStack {
                    Circle()
                        .stroke(isSelected ? Color.tooriAmber : Color.tooriStroke, lineWidth: 1.5)
                        .frame(width: 22, height: 22)
                    if isSelected {
                        Circle()
                            .fill(Color.tooriAmber)
                            .frame(width: 12, height: 12)
                    }
                }

                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 8) {
                        Text(model.displayName)
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(.white)
                        if model.isDefault {
                            Text("DEFAULT")
                                .font(.system(size: 9, weight: .black))
                                .foregroundStyle(Color.tooriAmber)
                                .padding(.horizontal, 6).padding(.vertical, 2)
                                .background(
                                    Capsule(style: .continuous)
                                        .fill(Color.tooriAmber.opacity(0.12))
                                )
                        }
                    }
                    Text("\(model.parameterSize) · \(model.ramRequirement) RAM")
                        .font(.system(size: 12))
                        .foregroundStyle(.white.opacity(0.45))
                }

                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(isSelected
                          ? Color.tooriAmber.opacity(0.07)
                          : Color.white.opacity(0.03))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(isSelected
                            ? Color.tooriAmber.opacity(0.4)
                            : Color.tooriStroke, lineWidth: 0.8)
            )
        }
    }
}

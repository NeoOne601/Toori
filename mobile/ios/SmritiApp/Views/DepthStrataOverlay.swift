import SwiftUI
import SmritiKit

/// Visual overlay that renders depth stratum zones on an image.
/// Uses the TPDS (Tri-Planar Depth Separation) data when available,
/// or falls back to a stylized gradient simulation.
struct DepthStrataOverlay: View {
    let strataData: SmritiDepthStrata?
    let imageSize: CGSize

    var body: some View {
        ZStack {
            if let strataData, strataData.confidence ?? 0 > 0.3 {
                // Real strata overlays from JEPA depth separation
                if let _ = strataData.foreground_mask {
                    stratumLayer(color: Color.tooriAmber.opacity(0.18), label: "Foreground")
                        .mask(strataMask(mask: strataData.foreground_mask, size: imageSize))
                }
                if let _ = strataData.midground_mask {
                    stratumLayer(color: Color.white.opacity(0.08), label: "Midground")
                        .mask(strataMask(mask: strataData.midground_mask, size: imageSize))
                }
                if let _ = strataData.background_mask {
                    stratumLayer(color: Color.tooriIndigo.opacity(0.18), label: "Background")
                        .mask(strataMask(mask: strataData.background_mask, size: imageSize))
                }
            } else {
                // Stylized depth gradient when no real data is available
                stylizedDepthOverlay
            }
        }
    }

    private func stratumLayer(color: Color, label: String) -> some View {
        Rectangle()
            .fill(color)
            .overlay(alignment: .bottomLeading) {
                Text(label)
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(.white.opacity(0.5))
                    .padding(4)
            }
    }

    private func strataMask(mask: [[Bool]]?, size: CGSize) -> some View {
        Canvas { context, canvasSize in
            guard let mask, !mask.isEmpty else { return }
            let rows = mask.count
            let cols = mask.first?.count ?? 0
            guard rows > 0, cols > 0 else { return }

            let cellWidth = canvasSize.width / CGFloat(cols)
            let cellHeight = canvasSize.height / CGFloat(rows)

            for row in 0..<rows {
                for col in 0..<cols {
                    if mask[row][col] {
                        let rect = CGRect(
                            x: CGFloat(col) * cellWidth,
                            y: CGFloat(row) * cellHeight,
                            width: cellWidth + 0.5,
                            height: cellHeight + 0.5
                        )
                        context.fill(Path(rect), with: .color(.white))
                    }
                }
            }
        }
        .frame(width: size.width, height: size.height)
    }

    private var stylizedDepthOverlay: some View {
        VStack(spacing: 0) {
            // Background stratum (top)
            Rectangle()
                .fill(
                    LinearGradient(
                        colors: [Color.tooriIndigo.opacity(0.12), Color.clear],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(maxHeight: .infinity)
                .overlay(alignment: .topTrailing) {
                    stratumTag("Background", color: .tooriIndigo)
                        .padding(8)
                }

            // Midground stratum (center)
            Rectangle()
                .fill(Color.white.opacity(0.02))
                .frame(maxHeight: .infinity)

            // Foreground stratum (bottom)
            Rectangle()
                .fill(
                    LinearGradient(
                        colors: [Color.clear, Color.tooriAmber.opacity(0.12)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(maxHeight: .infinity)
                .overlay(alignment: .bottomLeading) {
                    stratumTag("Foreground", color: .tooriAmber)
                        .padding(8)
                }
        }
    }

    private func stratumTag(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .bold))
            .foregroundStyle(.white.opacity(0.8))
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(
                Capsule(style: .continuous)
                    .fill(color.opacity(0.4))
            )
    }
}

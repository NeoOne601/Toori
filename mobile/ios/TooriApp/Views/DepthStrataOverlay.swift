import SwiftUI

// MARK: - DepthStrataOverlay
/// TPDS (Tri-Planar Depth Separation) visual overlay.
/// Renders foreground / midground / background strata on top of an image.
/// Falls back to a stylized gradient simulation when server masks are unavailable.
struct DepthStrataOverlay: View {
    let strataData: RealityDepthStrata?
    let imageSize: CGSize

    var body: some View {
        ZStack {
            if let strata = strataData, (strata.confidence ?? 0) > 0.3 {
                // Real TPDS masks from the JEPA pipeline
                if strata.foreground_mask != nil {
                    stratumLayer(color: Color.tooriAmber.opacity(0.18), label: "FG")
                        .mask(strataMask(mask: strata.foreground_mask, size: imageSize))
                }
                if strata.midground_mask != nil {
                    stratumLayer(color: Color.white.opacity(0.07), label: "MG")
                        .mask(strataMask(mask: strata.midground_mask, size: imageSize))
                }
                if strata.background_mask != nil {
                    stratumLayer(color: Color.tooriIndigo.opacity(0.18), label: "BG")
                        .mask(strataMask(mask: strata.background_mask, size: imageSize))
                }
            } else {
                stylizedDepthGradient
            }
        }
    }

    // MARK: - Helpers

    private func stratumLayer(color: Color, label: String) -> some View {
        Rectangle()
            .fill(color)
            .overlay(alignment: .bottomLeading) {
                Text(label)
                    .font(.system(size: 8, weight: .black))
                    .foregroundStyle(.white.opacity(0.45))
                    .padding(4)
            }
    }

    private func strataMask(mask: [[Bool]]?, size: CGSize) -> some View {
        Canvas { ctx, canvasSize in
            guard let mask, !mask.isEmpty else { return }
            let rows = mask.count
            let cols = mask.first?.count ?? 0
            guard rows > 0, cols > 0 else { return }
            let cw = canvasSize.width  / CGFloat(cols)
            let ch = canvasSize.height / CGFloat(rows)
            for r in 0..<rows {
                for c in 0..<cols where mask[r][c] {
                    ctx.fill(
                        Path(CGRect(x: CGFloat(c)*cw, y: CGFloat(r)*ch,
                                    width: cw + 0.5, height: ch + 0.5)),
                        with: .color(.white)
                    )
                }
            }
        }
        .frame(width: size.width, height: size.height)
    }

    /// Elegant fallback gradient when TPDS data is unavailable
    private var stylizedDepthGradient: some View {
        VStack(spacing: 0) {
            // Background zone — top
            ZStack(alignment: .topTrailing) {
                LinearGradient(
                    colors: [Color.tooriIndigo.opacity(0.14), .clear],
                    startPoint: .top, endPoint: .bottom
                )
                stratumTag("Background", color: .tooriIndigo).padding(8)
            }
            .frame(maxHeight: .infinity)

            // Midground zone
            Color.white.opacity(0.02)
                .frame(maxHeight: .infinity)

            // Foreground zone — bottom
            ZStack(alignment: .bottomLeading) {
                LinearGradient(
                    colors: [.clear, Color.tooriAmber.opacity(0.14)],
                    startPoint: .top, endPoint: .bottom
                )
                stratumTag("Foreground", color: .tooriAmber).padding(8)
            }
            .frame(maxHeight: .infinity)
        }
    }

    private func stratumTag(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 8, weight: .black))
            .foregroundStyle(.white.opacity(0.75))
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(
                Capsule(style: .continuous).fill(color.opacity(0.35))
            )
    }
}

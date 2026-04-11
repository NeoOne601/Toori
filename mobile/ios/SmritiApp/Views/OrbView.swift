import SwiftUI
import UIKit

struct OrbView: View {
    let slot: PulseOrbSlot
    let index: Int
    let onTap: () -> Void

    @State private var revealLabel = false

    var body: some View {
        ZStack(alignment: .bottom) {
            Circle()
                .fill(Color.white.opacity(isPlaceholder ? 0.035 : 0.06))
                .overlay(orbImage)
                .overlay(ringOverlay)
                .frame(width: 108, height: 108)
                .scaleEffect(breathingScale)
                .shadow(color: ringColor.opacity(isPlaceholder ? 0 : 0.32), radius: 22, y: 8)
                .contentShape(Circle())
                .onTapGesture {
                    guard !isPlaceholder else { return }
                    onTap()
                }
                .onLongPressGesture(minimumDuration: 0.35, maximumDistance: 24) {
                } onPressingChanged: { pressing in
                    guard !isPlaceholder else { return }
                    if pressing && !revealLabel {
                        UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    }
                    withAnimation(.smritiSpring) {
                        revealLabel = pressing
                    }
                }

            if revealLabel, let label = orbLabel {
                Text(label)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.white)
                    .lineLimit(2)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(
                        Capsule(style: .continuous)
                            .fill(Color.black.opacity(0.72))
                    )
                    .offset(y: 28)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .frame(width: 120, height: 144)
    }

    private var isPlaceholder: Bool {
        if case .placeholder = slot {
            return true
        }
        return false
    }

    private var orbImage: some View {
        Group {
            switch slot {
            case .observation(let observation):
                if let image = loadImage(path: observation.thumbnail_path) ?? loadImage(path: observation.image_path) {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                } else {
                    placeholderIcon
                }
            case .placeholder:
                placeholderIcon
            }
        }
        .clipShape(Circle())
    }

    private var placeholderIcon: some View {
        ZStack {
            Circle()
                .fill(Color.white.opacity(0.03))
            Image(systemName: "brain.head.profile.fill")
                .font(.system(size: 34, weight: .medium))
                .foregroundStyle(Color.smritiAccent.opacity(0.6))
        }
    }

    private var ringOverlay: some View {
        Circle()
            .stroke(
                ringColor.opacity(isPlaceholder ? 0.12 : 0.92),
                style: StrokeStyle(lineWidth: ringLineWidth)
            )
            .padding(2)
    }

    private var orbLabel: String? {
        switch slot {
        case .observation(let observation):
            return observation.displayLabel
        case .placeholder:
            return nil
        }
    }

    private var ringColor: Color {
        switch slot {
        case .observation(let observation):
            let surprise = observation.surpriseProxy
            return Color(
                red: min(0.92, 0.22 + surprise * 0.46),
                green: 0.36,
                blue: min(1.0, 0.78 + surprise * 0.14)
            )
        case .placeholder:
            return .white
        }
    }

    private var ringLineWidth: CGFloat {
        switch slot {
        case .observation(let observation):
            return 2.0 + CGFloat(min(max(observation.surpriseProxy, 0), 1)) * 6.0
        case .placeholder:
            return 1.2
        }
    }

    private var breathingScale: CGFloat {
        let time = Date().timeIntervalSinceReferenceDate
        let stagger = Double(index) * 0.6
        let phase = (time + stagger).truncatingRemainder(dividingBy: 4) / 4
        let sine = sin(phase * .pi * 2)
        return 0.985 + CGFloat((sine + 1) / 2) * 0.015
    }

    private func loadImage(path: String?) -> UIImage? {
        guard let path, !path.isEmpty else { return nil }
        return UIImage(contentsOfFile: path)
    }
}

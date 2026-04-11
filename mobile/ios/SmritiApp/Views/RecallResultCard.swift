import SwiftUI
import UIKit

struct RecallResultCard: View {
    let thumbnailPath: String?
    let title: String
    let subtitle: String
    let surprise: Double?
    let matchedBySound: Bool
    var onTap: (() -> Void)?

    var body: some View {
        Button {
            onTap?()
        } label: {
            HStack(spacing: 14) {
                thumbnail

                VStack(alignment: .leading, spacing: 6) {
                    Text(title)
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.leading)
                        .lineLimit(2)
                    Text(subtitle)
                        .font(.system(size: 12))
                        .foregroundStyle(.white.opacity(0.6))
                        .lineLimit(2)

                    HStack(spacing: 8) {
                        if let surprise, surprise > 0.6 {
                            badge(title: "⚡ \(surprise.formatted(.number.precision(.fractionLength(2))))", fill: Color.smritiAccent.opacity(0.18))
                        }
                        if matchedBySound {
                            badge(title: "matched by sound", fill: Color.smritiTeal.opacity(0.18))
                        }
                    }
                }

                Spacer(minLength: 0)
            }
            .padding(14)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(Color.white.opacity(0.06))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(Color.smritiStroke, lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
    }

    private var thumbnail: some View {
        Group {
            if let thumbnailPath, let image = UIImage(contentsOfFile: thumbnailPath) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Color.white.opacity(0.06))
                    Image(systemName: "photo")
                        .font(.system(size: 18, weight: .medium))
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
        }
        .frame(width: 64, height: 48)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private func badge(title: String, fill: Color) -> some View {
        Text(title)
            .font(.system(size: 10, weight: .semibold))
            .foregroundStyle(.white.opacity(0.9))
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(
                Capsule(style: .continuous)
                    .fill(fill)
            )
    }
}

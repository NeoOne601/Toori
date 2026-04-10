import SwiftUI

struct SmritiRecallCard: View {
    let item: SmritiRecallItem
    let index: Int
    let onSelect: () -> Void

    @State private var isVisible = false

    var body: some View {
        Button(action: onSelect) {
            HStack(alignment: .top, spacing: 14) {
                RecallThumbnailView(path: item.thumbnail_path, fallbackPath: item.file_path)
                    .frame(width: 80, height: 60)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))

                VStack(alignment: .leading, spacing: 8) {
                    HStack(alignment: .top, spacing: 12) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(item.primary_description)
                                .font(.system(size: 17, weight: .medium))
                                .foregroundStyle(.primary)
                                .multilineTextAlignment(.leading)

                            Text(item.displaySubtitle)
                                .font(.system(size: 13))
                                .foregroundStyle(.secondary)
                                .multilineTextAlignment(.leading)
                        }

                        Spacer(minLength: 0)

                        if item.surpriseProxy > 0.6 {
                            Text("⚡ \(item.surpriseProxy.formatted(.number.precision(.fractionLength(2))))")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(Color.smritiAccent)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(
                                    Capsule(style: .continuous)
                                        .fill(Color.smritiBadgeFill)
                                )
                        }
                    }

                    Divider()
                        .overlay(Color.smritiDivider)
                }
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.white.opacity(0.05))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(Color.smritiGlassStroke, lineWidth: 0.5)
            )
            .contentShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .offset(y: isVisible ? 0 : 24)
            .opacity(isVisible ? 1 : 0)
            .animation(.smritiStagger(index: index), value: isVisible)
        }
        .buttonStyle(.plain)
        .onAppear {
            isVisible = true
        }
    }
}

private struct RecallThumbnailView: View {
    let path: String
    let fallbackPath: String

    var body: some View {
        Group {
            if let image = NSImage(contentsOfFile: path), !path.isEmpty {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
            } else if let image = NSImage(contentsOfFile: fallbackPath) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                ZStack {
                    LinearGradient(
                        colors: [Color.smritiAccent.opacity(0.3), Color.smritiTeal.opacity(0.2)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                    Image(systemName: "photo")
                        .font(.system(size: 18, weight: .medium))
                        .foregroundStyle(.white.opacity(0.7))
                }
            }
        }
    }
}

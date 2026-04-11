import SwiftUI

struct SurpriseBannerView: View {
    let imagePath: String?
    let label: String

    @State var isVisible: Bool = false

    var body: some View {
        HStack(spacing: 12) {
            if let imagePath, !imagePath.isEmpty, let nsImage = NSImage(contentsOfFile: imagePath) {
                Image(nsImage: nsImage)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 32, height: 32)
                    .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
            } else {
                fallbackThumbnail
                    .frame(width: 32, height: 32)
                    .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)

                Text("⚡ Surprising moment captured")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 0)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.smritiAccent.opacity(0.12))
        )
        .offset(y: isVisible ? 0 : -44)
        .opacity(isVisible ? 1.0 : 0.0)
        .animation(.smritiSpring, value: isVisible)
        .onAppear {
            withAnimation(.smritiSpring) {
                isVisible = true
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                withAnimation(.smritiSpring) {
                    isVisible = false
                }
            }
        }
    }

    private var fallbackThumbnail: some View {
        ZStack {
            LinearGradient(
                colors: [Color.smritiAccent.opacity(0.42), Color.smritiTeal.opacity(0.26)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            Image(systemName: "sparkles")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.white.opacity(0.85))
        }
    }
}

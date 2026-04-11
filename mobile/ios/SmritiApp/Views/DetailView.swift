import SwiftUI
import UIKit

struct DetailView: View {
    @EnvironmentObject private var appModel: SmritiAppModel

    let memory: SelectedMemory

    var body: some View {
        NavigationStack {
            DetailContent(memory: memory, showsCloseButton: true) {
                appModel.dismissDetail()
            }
            .toolbar(.hidden, for: .navigationBar)
        }
    }
}

struct DetailContent: View {
    let memory: SelectedMemory
    let showsCloseButton: Bool
    var onClose: (() -> Void)?

    @State private var sharePresented = false
    @State private var meterProgress = 0.0

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Color.smritiCanvas.ignoresSafeArea()

            ScrollView(showsIndicators: false) {
                VStack(alignment: .leading, spacing: 22) {
                    heroSection
                    titleBlock
                    surpriseMeter
                    metricRow
                    entityRow
                    if !memory.entityPills.isEmpty {
                        peopleOrbitRow
                    }
                    descriptionBlock
                    actionRow
                }
                .padding(.bottom, 28)
            }

            if showsCloseButton {
                Button {
                    onClose?()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(.white.opacity(0.9))
                        .frame(width: 36, height: 36)
                        .background(Circle().fill(Color.black.opacity(0.44)))
                }
                .padding(.top, 12)
                .padding(.trailing, 18)
            }
        }
        .sheet(isPresented: $sharePresented) {
            ActivityShareView(items: shareItems)
        }
        .onAppear {
            withAnimation(.smritiSpring.delay(0.08)) {
                meterProgress = memory.surpriseValue
            }
        }
    }

    private var heroSection: some View {
        GeometryReader { proxy in
            let minY = proxy.frame(in: .global).minY

            Group {
                if let image = detailImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                } else {
                    ZStack {
                        LinearGradient(
                            colors: [Color.smritiAccent.opacity(0.34), Color.smritiTeal.opacity(0.18)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                        Image(systemName: "brain.head.profile.fill")
                            .font(.system(size: 56, weight: .medium))
                            .foregroundStyle(.white.opacity(0.85))
                    }
                }
            }
            .frame(width: proxy.size.width, height: proxy.size.height + max(minY, 0))
            .clipped()
            .offset(y: minY > 0 ? -minY * 0.4 : minY * 0.4)
        }
        .frame(height: 340)
        .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .stroke(Color.smritiStroke, lineWidth: 0.5)
        )
        .padding(.horizontal, 18)
        .padding(.top, 18)
    }

    private var titleBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(memory.title)
                .font(.system(size: 28, weight: .semibold))
                .foregroundStyle(.white)
            Text(memory.subtitle)
                .font(.system(size: 13))
                .foregroundStyle(.white.opacity(0.62))
        }
        .padding(.horizontal, 22)
    }

    private var surpriseMeter: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Surprise")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.82))
                Spacer()
                Text(memory.surpriseValue.formatted(.number.precision(.fractionLength(2))))
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Color.smritiAccent)
            }

            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule(style: .continuous)
                        .fill(Color.white.opacity(0.08))

                    Capsule(style: .continuous)
                        .fill(surpriseColor(memory.surpriseValue))
                        .frame(width: proxy.size.width * max(0, min(meterProgress, 1)))
                }
            }
            .frame(height: 10)
        }
        .padding(18)
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(Color.smritiSurface)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .stroke(Color.smritiStroke, lineWidth: 0.5)
        )
        .padding(.horizontal, 18)
    }

    private var metricRow: some View {
        HStack(spacing: 14) {
            MetricsArcGauge(title: "Prediction", value: memory.metricTriplet.prediction_consistency)
            MetricsArcGauge(title: "Continuity", value: memory.metricTriplet.temporal_continuity_score)
            MetricsArcGauge(title: "Surprise", value: memory.surpriseValue)
        }
        .padding(.horizontal, 18)
    }

    private var entityRow: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Entity tracks")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.white.opacity(0.68))

            if memory.entityPills.isEmpty {
                Text("No tracked entities surfaced for this memory.")
                    .font(.system(size: 13))
                    .foregroundStyle(.white.opacity(0.42))
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 10) {
                        ForEach(memory.entityPills) { pill in
                            HStack(spacing: 7) {
                                Image(systemName: pill.systemImage)
                                Text(pill.label)
                            }
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.white.opacity(0.92))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(
                                Capsule(style: .continuous)
                                    .fill(Color.white.opacity(0.06))
                            )
                            .overlay(
                                Capsule(style: .continuous)
                                    .stroke(Color.smritiStroke, lineWidth: 0.5)
                            )
                        }
                    }
                }
            }
        }
        .padding(.horizontal, 22)
    }

    private var peopleOrbitRow: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Social orbit")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.white.opacity(0.68))

            PeopleOrbitView(orbits: PeopleOrbitEngine().generateOrbit(for: memory.entityPills.map(\.label)))
                .frame(height: 220)
                .background(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(Color.smritiSurface)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .stroke(Color.smritiStroke, lineWidth: 0.5)
                )
        }
        .padding(.horizontal, 22)
    }

    private var descriptionBlock: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Setu-2")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.white.opacity(0.68))
            WordRevealText(text: memory.descriptionText)
        }
        .padding(18)
        .background(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(Color.smritiSurface)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .stroke(Color.smritiStroke, lineWidth: 0.5)
        )
        .padding(.horizontal, 18)
    }

    private var actionRow: some View {
        HStack {
            Button {
                sharePresented = true
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "square.and.arrow.up")
                    Text("Share")
                }
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 18)
                .padding(.vertical, 14)
                .background(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(Color.smritiAccent)
                )
            }

            Spacer()
        }
        .padding(.horizontal, 18)
    }

    private var detailImage: UIImage? {
        if let heroPath = memory.heroPath, let image = UIImage(contentsOfFile: heroPath) {
            return image
        }
        if let fallbackPath = memory.fallbackThumbnailPath {
            return UIImage(contentsOfFile: fallbackPath)
        }
        return nil
    }

    private var shareItems: [Any] {
        var items: [Any] = [memory.shareText]
        let generator = MemoryCardGenerator()
        if let targetPath = memory.heroPath ?? memory.fallbackThumbnailPath {
            if let cardURL = generator.generateMemoryCard(imagePath: targetPath, date: memory.creationDate, summary: memory.descriptionText) {
                items.append(cardURL)
            } else {
                items.append(URL(fileURLWithPath: targetPath))
            }
        }
        return items
    }
}

private struct MetricsArcGauge: View {
    let title: String
    let value: Double

    @State private var progress = 0.0

    var body: some View {
        VStack(spacing: 12) {
            Canvas { context, size in
                let rect = CGRect(origin: .zero, size: size).insetBy(dx: 8, dy: 8)
                let start = Angle(degrees: 135)
                let end = Angle(degrees: 405)

                var background = Path()
                background.addArc(center: CGPoint(x: rect.midX, y: rect.midY), radius: rect.width / 2, startAngle: start, endAngle: end, clockwise: false)
                context.stroke(background, with: .color(Color.white.opacity(0.08)), style: .init(lineWidth: 10, lineCap: .round))

                var foreground = Path()
                foreground.addArc(
                    center: CGPoint(x: rect.midX, y: rect.midY),
                    radius: rect.width / 2,
                    startAngle: start,
                    endAngle: Angle(degrees: 135 + 270 * progress),
                    clockwise: false
                )
                context.stroke(
                    foreground,
                    with: .linearGradient(
                        Gradient(colors: [Color.smritiTeal, Color.smritiAccent]),
                        startPoint: CGPoint(x: rect.minX, y: rect.midY),
                        endPoint: CGPoint(x: rect.maxX, y: rect.midY)
                    ),
                    style: .init(lineWidth: 10, lineCap: .round)
                )
            }
            .frame(width: 96, height: 96)

            VStack(spacing: 4) {
                Text(title)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.65))
                Text(value.formatted(.percent.precision(.fractionLength(0))))
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(.white)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 18)
        .background(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(Color.smritiSurface)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .stroke(Color.smritiStroke, lineWidth: 0.5)
        )
        .onAppear {
            withAnimation(.smritiSpring.delay(0.06)) {
                progress = min(max(value, 0), 1)
            }
        }
    }
}

private struct WordRevealText: View {
    let text: String

    @State private var reveal = false

    private var words: [String] {
        text.split(separator: " ").map(String.init)
    }

    var body: some View {
        FlowWrap(spacing: 6) {
            ForEach(Array(words.enumerated()), id: \.offset) { index, word in
                Text(word)
                    .font(.system(size: 17))
                    .italic()
                    .foregroundStyle(.white.opacity(0.92))
                    .opacity(reveal ? 1 : 0)
                    .offset(y: reveal ? 0 : 4)
                    .animation(.smritiSpring.delay(Double(index) * 0.03), value: reveal)
            }
        }
        .onAppear {
            reveal = true
        }
    }
}

private struct FlowWrap: Layout {
    let spacing: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? UIScreen.main.bounds.width - 64
        var x: CGFloat = 0
        var y: CGFloat = 0
        var lineHeight: CGFloat = 0

        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x + size.width > width, x > 0 {
                x = 0
                y += lineHeight + spacing
                lineHeight = 0
            }
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }

        return CGSize(width: width, height: y + lineHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX
        var y = bounds.minY
        var lineHeight: CGFloat = 0

        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x + size.width > bounds.maxX, x > bounds.minX {
                x = bounds.minX
                y += lineHeight + spacing
                lineHeight = 0
            }
            view.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}

private struct ActivityShareView: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

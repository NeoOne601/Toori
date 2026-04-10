import SwiftUI

struct DetailSheet: View {
    @EnvironmentObject private var appModel: SmritiAppModel

    let item: SmritiRecallItem

    @State private var barProgress = 0.0

    var body: some View {
        ZStack {
            SmritiGlassBackground()
            ScrollView(.vertical, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 20) {
                    header
                    heroImage
                    surpriseMeter
                    entitySection
                    descriptionSection
                    actionRow
                    if let journalStatusMessage = appModel.journalStatusMessage {
                        Text(journalStatusMessage)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(24)
            }
        }
        .frame(minWidth: 720, minHeight: 820)
        .onAppear {
            withAnimation(.smritiSpring.delay(0.08)) {
                barProgress = item.surpriseProxy
            }
        }
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Memory Detail")
                    .font(.system(size: 28, weight: .semibold))
                Text(item.created_at.formatted(date: .abbreviated, time: .shortened))
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button {
                appModel.closeDetail()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(.secondary)
                    .frame(width: 30, height: 30)
                    .background(Circle().fill(Color.white.opacity(0.06)))
            }
            .buttonStyle(.plain)
        }
    }

    private var heroImage: some View {
        Group {
            if let image = NSImage(contentsOfFile: item.file_path) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
            } else if let image = NSImage(contentsOfFile: item.thumbnail_path) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
            } else {
                ZStack {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color.white.opacity(0.05))
                    Image(systemName: "photo.artframe")
                        .font(.system(size: 40, weight: .medium))
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Color.smritiGlassStroke, lineWidth: 0.5)
        )
    }

    private var surpriseMeter: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Surprise score")
                    .font(.system(size: 13, weight: .semibold))
                Spacer()
                Text(item.surpriseProxy.formatted(.number.precision(.fractionLength(2))))
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Color.smritiAccent)
            }

            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule(style: .continuous)
                        .fill(Color.white.opacity(0.08))

                    Capsule(style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [Color.smritiTeal, Color.smritiAccent],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: geometry.size.width * barProgress)
                }
            }
            .frame(height: 10)
        }
    }

    private var entitySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Entity tracks")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.secondary)

            FlowLayout(spacing: 10) {
                ForEach(item.person_names, id: \.self) { name in
                    EntityPill(label: name, systemImage: "eye")
                }
                ForEach(item.anchor_matches, id: \.self) { match in
                    EntityPill(label: match.open_vocab_label ?? match.template_name, systemImage: "questionmark")
                }
            }
        }
    }

    private var descriptionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Setu-2")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.secondary)
            Text(item.primarySetuText)
                .font(.system(size: 17))
                .italic()
                .multilineTextAlignment(.leading)
        }
    }

    private var actionRow: some View {
        HStack(spacing: 12) {
            if item.person_names.count > 1 {
                Menu {
                    ForEach(item.person_names, id: \.self) { name in
                        Button(name) {
                            Task {
                                await appModel.addToJournal(item: item, personName: name)
                            }
                        }
                    }
                } label: {
                    actionLabel(title: "Add to journal", systemImage: "book.closed")
                }
                .menuStyle(.borderlessButton)
                .disabled(item.person_names.isEmpty)
            } else {
                Button {
                    Task {
                        await appModel.addCurrentItemToJournal()
                    }
                } label: {
                    actionLabel(title: "Add to journal", systemImage: "book.closed")
                }
                .buttonStyle(.plain)
                .disabled(item.person_names.isEmpty)
            }

            Button {
                appModel.share(item: item)
            } label: {
                actionLabel(title: "Share", systemImage: "square.and.arrow.up")
            }
            .buttonStyle(.plain)
        }
    }

    private func actionLabel(title: String, systemImage: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: systemImage)
            Text(title)
        }
        .font(.system(size: 13, weight: .semibold))
        .foregroundStyle(.primary)
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.white.opacity(0.06))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Color.smritiGlassStroke, lineWidth: 0.5)
        )
    }
}

private struct EntityPill: View {
    let label: String
    let systemImage: String

    var body: some View {
        HStack(spacing: 7) {
            Image(systemName: systemImage)
            Text(label)
        }
        .font(.system(size: 11, weight: .medium))
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(
            Capsule(style: .continuous)
                .fill(Color.white.opacity(0.06))
        )
    }
}

private struct FlowLayout: Layout {
    let spacing: CGFloat

    init(spacing: CGFloat = 8) {
        self.spacing = spacing
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 600
        var currentX: CGFloat = 0
        var currentY: CGFloat = 0
        var lineHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if currentX + size.width > width, currentX > 0 {
                currentX = 0
                currentY += lineHeight + spacing
                lineHeight = 0
            }
            currentX += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }

        return CGSize(width: width, height: currentY + lineHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var currentX = bounds.minX
        var currentY = bounds.minY
        var lineHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if currentX + size.width > bounds.maxX, currentX > bounds.minX {
                currentX = bounds.minX
                currentY += lineHeight + spacing
                lineHeight = 0
            }
            subview.place(
                at: CGPoint(x: currentX, y: currentY),
                proposal: ProposedViewSize(size)
            )
            currentX += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}

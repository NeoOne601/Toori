import SwiftUI
import OSLog

private let mandalaLogger = Logger(subsystem: "com.toori.smriti", category: "Mandala")

struct MandalaView: View {
    @EnvironmentObject private var appModel: SmritiAppModel
    @StateObject private var simulation = MandalaSimulation()
    @GestureState private var dragOffset: CGSize = .zero
    @State private var committedOffset: CGSize = .zero
    @State private var committedZoom: CGFloat = 1
    @State private var lastLoggedClusterCount: Int?

    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    var body: some View {
        Group {
            if UIDevice.current.userInterfaceIdiom == .pad && horizontalSizeClass == .regular {
                NavigationSplitView {
                    graphSurface
                } detail: {
                    if let selectedMemory = appModel.selectedMemory {
                        DetailContent(memory: selectedMemory, showsCloseButton: false)
                    } else {
                        detailPlaceholder
                    }
                }
            } else {
                graphSurface
            }
        }
        .task {
            await appModel.loadMandalaIfNeeded()
        }
    }

    private var graphSurface: some View {
        GeometryReader { geometry in
            ZStack {
                Color.smritiCanvas.ignoresSafeArea()

                if appModel.isMandalaLoading {
                    ProgressView()
                        .tint(Color.smritiAccent)
                } else if let data = appModel.mandalaData, !data.nodes.isEmpty {
                    let displayData = truncatedMandalaData(from: data)
                    TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { _ in
                        Canvas { context, size in
                            simulation.updateData(displayData, in: size)
                            simulation.zoom = min(max(committedZoom, 0.5), 3)
                            simulation.offset = CGSize(
                                width: committedOffset.width + dragOffset.width,
                                height: committedOffset.height + dragOffset.height
                            )
                            simulation.step()
                            drawEdges(context: &context, in: size, data: displayData)
                            drawNodes(context: &context, in: size, data: displayData)
                        }
                    }
                    .contentShape(Rectangle())
                    .gesture(dragGesture)
                    .simultaneousGesture(magnificationGesture)
                    .simultaneousGesture(
                        SpatialTapGesture().onEnded { value in
                            simulation.selectNode(at: value.location, size: geometry.size)
                        }
                    )
                    .overlay(alignment: .topLeading) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Mandala")
                                .font(.system(size: 28, weight: .semibold))
                                .foregroundStyle(.white)
                            Text("Clusters and their co-occurrence edges.")
                                .font(.system(size: 13))
                                .foregroundStyle(.white.opacity(0.58))
                        }
                        .padding(24)
                    }
                    .overlay {
                        if let selected = simulation.selectedNode(in: displayData) {
                            orbitOverlay(for: selected, size: geometry.size)
                        }
                    }
                    .task(id: data.nodes.count) {
                        guard data.nodes.count > 120, lastLoggedClusterCount != data.nodes.count else { return }
                        mandalaLogger.warning("Truncating mandala cluster graph from \(data.nodes.count, privacy: .public) nodes to 120 plus a meta-node.")
                        lastLoggedClusterCount = data.nodes.count
                    }
                } else {
                    VStack(spacing: 14) {
                        Image(systemName: "point.3.connected.trianglepath.dotted")
                            .font(.system(size: 34, weight: .medium))
                            .foregroundStyle(Color.smritiAccent)
                        Text("Mandala appears once Smriti has enough memories to cluster.")
                            .font(.system(size: 15))
                            .foregroundStyle(.white.opacity(0.62))
                            .multilineTextAlignment(.center)
                    }
                    .padding(32)
                }
            }
        }
    }

    private var detailPlaceholder: some View {
        ZStack {
            Color.smritiCanvas.ignoresSafeArea()
            VStack(spacing: 14) {
                Image(systemName: "sparkles")
                    .font(.system(size: 30, weight: .medium))
                    .foregroundStyle(Color.smritiAccent)
                Text("Select a memory from Pulse to see it here while you explore Mandala.")
                    .font(.system(size: 15))
                    .foregroundStyle(.white.opacity(0.6))
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 320)
            }
        }
    }

    private var dragGesture: some Gesture {
        DragGesture()
            .updating($dragOffset) { value, state, _ in
                state = value.translation
            }
            .onEnded { value in
                committedOffset.width += value.translation.width
                committedOffset.height += value.translation.height
            }
    }

    private var magnificationGesture: some Gesture {
        MagnificationGesture()
            .onChanged { value in
                simulation.zoom = min(max(committedZoom * value, 0.5), 3)
            }
            .onEnded { value in
                committedZoom = min(max(committedZoom * value, 0.5), 3)
                simulation.zoom = committedZoom
            }
    }

    private func drawEdges(context: inout GraphicsContext, in size: CGSize, data: SmritiMandalaData) {
        for edge in data.edges {
            guard
                let start = simulation.projectedPosition(for: edge.source, size: size),
                let end = simulation.projectedPosition(for: edge.target, size: size)
            else { continue }

            var path = Path()
            path.move(to: start)
            path.addLine(to: end)
            context.stroke(
                path,
                with: .color(Color.smritiAccent.opacity(max(0.08, edge.similarity * 0.42))),
                lineWidth: 0.5
            )
        }
    }

    private func drawNodes(context: inout GraphicsContext, in size: CGSize, data: SmritiMandalaData) {
        let minCount = data.nodes.map(\.media_count).min() ?? 1
        let maxCount = max(data.nodes.map(\.media_count).max() ?? 1, minCount + 1)

        for node in data.nodes {
            let diameter = simulation.nodeDiameter(for: node, minCount: minCount, maxCount: maxCount)
            let opacity = simulation.nodeOpacity(for: node, minCount: minCount, maxCount: maxCount)
            let position = simulation.projectedPosition(for: node.id, size: size) ?? CGPoint(x: size.width / 2, y: size.height / 2)
            let rect = CGRect(x: position.x - diameter / 2, y: position.y - diameter / 2, width: diameter, height: diameter)

            context.fill(Path(ellipseIn: rect), with: .color(Color.smritiAccent.opacity(opacity)))

            if simulation.selectedNodeID == node.id {
                context.stroke(Path(ellipseIn: rect.insetBy(dx: -5, dy: -5)), with: .color(.white.opacity(0.55)), lineWidth: 1)
            }
        }
    }

    private func orbitOverlay(for node: SmritiClusterNode, size: CGSize) -> some View {
        let point = simulation.projectedPosition(for: node.id, size: size) ?? CGPoint(x: size.width / 2, y: size.height / 2)
        return ZStack {
            ForEach(0..<3, id: \.self) { index in
                let angle = CGFloat(index) / 3 * .pi * 2 - .pi / 2
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [Color.smritiAccent.opacity(0.58), Color.smritiTeal.opacity(0.32)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 28, height: 28)
                    .overlay(
                        Image(systemName: "photo")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.82))
                    )
                    .position(
                        x: point.x + cos(angle) * 44,
                        y: point.y + sin(angle) * 44
                    )
                    .transition(.scale.combined(with: .opacity))
            }

            Text(node.label)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(
                    Capsule(style: .continuous)
                        .fill(Color.black.opacity(0.58))
                )
                .position(x: point.x, y: point.y + 44)
        }
        .animation(.smritiSpring, value: simulation.selectedNodeID)
    }

    private func truncatedMandalaData(from data: SmritiMandalaData) -> SmritiMandalaData {
        guard data.nodes.count > 120 else { return data }

        let sortedNodes = data.nodes.sorted { lhs, rhs in
            if lhs.media_count == rhs.media_count {
                return lhs.id < rhs.id
            }
            return lhs.media_count > rhs.media_count
        }
        let keptNodes = Array(sortedNodes.prefix(120))
        let removedNodes = Array(sortedNodes.dropFirst(120))
        guard !removedNodes.isEmpty else { return data }

        let keptIDs = Set(keptNodes.map(\.id))
        let removedIDs = Set(removedNodes.map(\.id))
        let moreNodeID = (keptNodes.map(\.id).min() ?? 0) - 1
        let aggregatedCount = removedNodes.reduce(0) { $0 + $1.media_count }

        var edgeTotals: [Int: Double] = [:]
        var keptEdges: [SmritiClusterEdge] = []

        for edge in data.edges {
            let sourceKept = keptIDs.contains(edge.source)
            let targetKept = keptIDs.contains(edge.target)
            let sourceRemoved = removedIDs.contains(edge.source)
            let targetRemoved = removedIDs.contains(edge.target)

            switch (sourceKept, targetKept, sourceRemoved, targetRemoved) {
            case (true, true, _, _):
                keptEdges.append(edge)
            case (true, false, _, true):
                edgeTotals[edge.source, default: 0] += edge.similarity
            case (false, true, true, _):
                edgeTotals[edge.target, default: 0] += edge.similarity
            default:
                continue
            }
        }

        let moreNode = SmritiClusterNode(
            id: moreNodeID,
            label: "more…",
            media_count: aggregatedCount,
            centroid: [0, 0],
            dominant_depth_stratum: nil,
            temporal_span_days: nil
        )

        let moreEdges = edgeTotals
            .sorted { $0.key < $1.key }
            .map { keptID, similarity in
                SmritiClusterEdge(source: moreNodeID, target: keptID, similarity: similarity)
            }

        return SmritiMandalaData(
            nodes: keptNodes + [moreNode],
            edges: keptEdges + moreEdges,
            generated_at: data.generated_at
        )
    }
}

@MainActor
final class MandalaSimulation: ObservableObject {
    struct NodeState {
        var point: CGPoint
        var velocity: CGVector = .zero
    }

    private var data: SmritiMandalaData?
    private var states: [Int: NodeState] = [:]

    var zoom: CGFloat = 1
    var offset: CGSize = .zero
    var selectedNodeID: Int?

    func updateData(_ data: SmritiMandalaData, in size: CGSize) {
        self.data = data
        let nodeIDs = Set(data.nodes.map(\.id))
        if states.count == data.nodes.count, Set(states.keys) == nodeIDs { return }

        var nextStates: [Int: NodeState] = [:]
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        for (index, node) in data.nodes.enumerated() {
            let angle = Double(index) / Double(max(data.nodes.count, 1)) * .pi * 2
            let radius = min(size.width, size.height) * 0.22
            nextStates[node.id] = NodeState(
                point: CGPoint(
                    x: center.x + CGFloat(cos(angle)) * radius,
                    y: center.y + CGFloat(sin(angle)) * radius
                )
            )
        }
        states = nextStates
    }

    func step() {
        guard let data else { return }
        let repulsion: CGFloat = 80
        let attraction: CGFloat = 0.12
        let damping: CGFloat = 0.85

        for source in data.nodes {
            guard var sourceState = states[source.id] else { continue }

            var force = CGVector.zero
            for target in data.nodes where target.id != source.id {
                guard let targetState = states[target.id] else { continue }
                let dx = sourceState.point.x - targetState.point.x
                let dy = sourceState.point.y - targetState.point.y
                let distance = max(sqrt(dx * dx + dy * dy), 1)
                let magnitude = repulsion / distance
                force.dx += (dx / distance) * magnitude
                force.dy += (dy / distance) * magnitude
            }

            for edge in data.edges where edge.source == source.id || edge.target == source.id {
                let otherID = edge.source == source.id ? edge.target : edge.source
                guard let otherState = states[otherID] else { continue }
                let dx = otherState.point.x - sourceState.point.x
                let dy = otherState.point.y - sourceState.point.y
                force.dx += dx * attraction * CGFloat(edge.similarity)
                force.dy += dy * attraction * CGFloat(edge.similarity)
            }

            sourceState.velocity.dx = (sourceState.velocity.dx + force.dx * 0.016) * damping
            sourceState.velocity.dy = (sourceState.velocity.dy + force.dy * 0.016) * damping
            sourceState.point.x += sourceState.velocity.dx
            sourceState.point.y += sourceState.velocity.dy
            states[source.id] = sourceState
        }
    }

    func nodeDiameter(for node: SmritiClusterNode, minCount: Int, maxCount: Int) -> CGFloat {
        let normalized = CGFloat(node.media_count - minCount) / CGFloat(max(1, maxCount - minCount))
        return 12 + normalized * 20
    }

    func nodeOpacity(for node: SmritiClusterNode, minCount: Int, maxCount: Int) -> Double {
        let normalized = Double(node.media_count - minCount) / Double(max(1, maxCount - minCount))
        return 0.28 + normalized * 0.52
    }

    func projectedPosition(for id: Int, size: CGSize) -> CGPoint? {
        guard let state = states[id] else { return nil }
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        return CGPoint(
            x: center.x + (state.point.x - center.x) * zoom + offset.width,
            y: center.y + (state.point.y - center.y) * zoom + offset.height
        )
    }

    func selectNode(at location: CGPoint, size: CGSize) {
        guard let data else { return }
        selectedNodeID = data.nodes.min(by: { lhs, rhs in
            let l = distance(from: location, to: projectedPosition(for: lhs.id, size: size))
            let r = distance(from: location, to: projectedPosition(for: rhs.id, size: size))
            return l < r
        })?.id
    }

    func selectedNode(in data: SmritiMandalaData) -> SmritiClusterNode? {
        guard let selectedNodeID else { return nil }
        return data.nodes.first { $0.id == selectedNodeID }
    }

    private func distance(from point: CGPoint, to target: CGPoint?) -> CGFloat {
        guard let target else { return .greatestFiniteMagnitude }
        let dx = target.x - point.x
        let dy = target.y - point.y
        return sqrt(dx * dx + dy * dy)
    }
}

import SwiftUI
import CoreVideo
import OSLog

private let mandalaLogger = Logger(subsystem: "com.toori.smriti", category: "Mandala")

struct MandalaView: View {
    @EnvironmentObject private var appModel: SmritiAppModel
    @StateObject private var simulation = MandalaSimulation()
    @State private var gestureScale: CGFloat = 1
    @State private var gestureOffset: CGSize = .zero
    @State private var lastLoggedClusterCount: Int?

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.white.opacity(0.03))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .stroke(Color.smritiGlassStroke, lineWidth: 0.5)
                    )

                if appModel.isMandalaLoading {
                    ProgressView()
                        .controlSize(.small)
                } else if let data = appModel.mandalaData, !data.nodes.isEmpty {
                    let displayData = truncatedMandalaData(from: data)
                    Canvas { context, size in
                        simulation.update(data: displayData, canvasSize: size)
                        drawEdges(context: &context, data: displayData, size: size)
                        drawNodes(context: &context, data: displayData, size: size)
                    }
                    .contentShape(Rectangle())
                    .gesture(dragGesture)
                    .simultaneousGesture(magnificationGesture)
                    .simultaneousGesture(tapGesture(in: geometry.size))
                    .overlay(alignment: .topLeading) {
                        Text("Mandala")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(.secondary)
                            .padding(14)
                    }
                    .overlay {
                        if let selected = simulation.selectedNode(in: displayData) {
                            MandalaOrbitOverlay(
                                node: selected,
                                point: simulation.displayPoint(for: selected.id, in: geometry.size)
                            )
                        }
                    }
                    .onAppear {
                        simulation.start()
                    }
                    .onDisappear {
                        simulation.stop()
                    }
                    .task(id: data.nodes.count) {
                        guard data.nodes.count > 120, lastLoggedClusterCount != data.nodes.count else { return }
                        mandalaLogger.warning("Truncating mandala cluster graph from \(data.nodes.count, privacy: .public) nodes to 120 plus a meta-node.")
                        lastLoggedClusterCount = data.nodes.count
                    }
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "point.3.connected.trianglepath.dotted")
                            .font(.system(size: 26, weight: .medium))
                            .foregroundStyle(Color.smritiAccent)
                        Text("Clusters appear after Smriti stores enough memories.")
                            .font(.system(size: 13))
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: 220)
                    }
                }
            }
            .onAppear {
                Task {
                    await appModel.loadMandalaIfNeeded()
                }
            }
        }
    }

    private var dragGesture: some Gesture {
        DragGesture(minimumDistance: 1)
            .onChanged { value in
                simulation.additionalOffset = CGSize(
                    width: gestureOffset.width + value.translation.width,
                    height: gestureOffset.height + value.translation.height
                )
            }
            .onEnded { value in
                gestureOffset.width += value.translation.width
                gestureOffset.height += value.translation.height
                simulation.additionalOffset = gestureOffset
            }
    }

    private var magnificationGesture: some Gesture {
        MagnificationGesture()
            .onChanged { value in
                simulation.zoomScale = min(max(gestureScale * value, 0.5), 3)
            }
            .onEnded { value in
                gestureScale = min(max(gestureScale * value, 0.5), 3)
                simulation.zoomScale = gestureScale
            }
    }

    private func tapGesture(in size: CGSize) -> some Gesture {
        SpatialTapGesture()
            .onEnded { value in
                simulation.selectNode(at: value.location, in: size)
            }
    }

    private func drawEdges(context: inout GraphicsContext, data: SmritiMandalaData, size: CGSize) {
        for edge in data.edges {
            guard
                let start = simulation.displayPointOptional(for: edge.source, in: size),
                let end = simulation.displayPointOptional(for: edge.target, in: size)
            else { continue }

            var path = Path()
            path.move(to: start)
            path.addLine(to: end)
            context.stroke(
                path,
                with: .color(Color.smritiAccent.opacity(max(0.12, edge.similarity * 0.45))),
                lineWidth: 0.5
            )
        }
    }

    private func drawNodes(context: inout GraphicsContext, data: SmritiMandalaData, size: CGSize) {
        let counts = data.nodes.map(\.media_count)
        let minCount = counts.min() ?? 1
        let maxCount = max(counts.max() ?? 1, minCount + 1)

        for node in data.nodes {
            let point = simulation.displayPoint(for: node.id, in: size)
            let diameter = simulation.nodeDiameter(for: node, minCount: minCount, maxCount: maxCount)
            let opacity = simulation.nodeOpacity(for: node, minCount: minCount, maxCount: maxCount)
            let rect = CGRect(
                x: point.x - (diameter / 2),
                y: point.y - (diameter / 2),
                width: diameter,
                height: diameter
            )

            context.fill(Path(ellipseIn: rect), with: .color(Color.smritiAccent.opacity(opacity)))

            if simulation.selectedNodeID == node.id {
                context.stroke(
                    Path(ellipseIn: rect.insetBy(dx: -5, dy: -5)),
                    with: .color(Color.white.opacity(0.4)),
                    lineWidth: 1
                )
            }
        }
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

private struct MandalaOrbitOverlay: View {
    let node: SmritiClusterNode
    let point: CGPoint
    @State private var expanded = false

    var body: some View {
        ZStack {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [Color.smritiAccent.opacity(0.5), Color.smritiTeal.opacity(0.28)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 30, height: 30)
                    .overlay(
                        Image(systemName: "photo")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.8))
                    )
                    .position(orbitPosition(index: index))
                    .scaleEffect(expanded ? 1 : 0.2)
                    .opacity(expanded ? 1 : 0)
                    .animation(.smritiSpring.delay(Double(index) * 0.04), value: expanded)
            }

            Text(node.label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.white.opacity(0.85))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(
                    Capsule(style: .continuous)
                        .fill(Color.black.opacity(0.25))
                )
                .position(x: point.x, y: point.y + 42)
                .opacity(expanded ? 1 : 0)
                .animation(.smritiSpring.delay(0.08), value: expanded)
        }
        .onAppear {
            expanded = true
        }
    }

    private func orbitPosition(index: Int) -> CGPoint {
        let angle = (Double(index) / 3.0) * (.pi * 2) - (.pi / 2)
        let radius: CGFloat = 44
        return CGPoint(
            x: point.x + cos(angle) * radius,
            y: point.y + sin(angle) * radius
        )
    }
}

@MainActor
final class MandalaSimulation: ObservableObject {
    struct NodeState {
        var point: CGPoint
        var velocity: CGVector = .zero
    }

    private(set) var states: [Int: NodeState] = [:]
    private var displayLink: CVDisplayLink?
    private var latestData: SmritiMandalaData?
    private var canvasSize: CGSize = .zero

    var zoomScale: CGFloat = 1
    var additionalOffset: CGSize = .zero
    var selectedNodeID: Int?

    init() {
        var link: CVDisplayLink?
        CVDisplayLinkCreateWithActiveCGDisplays(&link)
        displayLink = link
        if let displayLink {
            CVDisplayLinkSetOutputCallback(displayLink, { _, _, _, _, _, userInfo in
                guard let userInfo else { return kCVReturnSuccess }
                let simulation = Unmanaged<MandalaSimulation>.fromOpaque(userInfo).takeUnretainedValue()
                Task { @MainActor in
                    simulation.step()
                }
                return kCVReturnSuccess
            }, Unmanaged.passUnretained(self).toOpaque())
        }
    }

    func start() {
        if let displayLink, !CVDisplayLinkIsRunning(displayLink) {
            CVDisplayLinkStart(displayLink)
        }
    }

    func stop() {
        if let displayLink, CVDisplayLinkIsRunning(displayLink) {
            CVDisplayLinkStop(displayLink)
        }
    }

    func update(data: SmritiMandalaData, canvasSize: CGSize) {
        latestData = data
        self.canvasSize = canvasSize
        let nodeIDs = Set(data.nodes.map(\.id))
        if states.count == data.nodes.count, Set(states.keys) == nodeIDs {
            return
        }
        var seeded: [Int: NodeState] = [:]
        let center = CGPoint(x: canvasSize.width / 2, y: canvasSize.height / 2)
        for (index, node) in data.nodes.enumerated() {
            let angle = (Double(index) / Double(max(data.nodes.count, 1))) * (.pi * 2)
            let radius = min(canvasSize.width, canvasSize.height) * 0.22
            seeded[node.id] = NodeState(
                point: CGPoint(
                    x: center.x + cos(angle) * radius,
                    y: center.y + sin(angle) * radius
                )
            )
        }
        states = seeded
    }

    func step() {
        guard let data = latestData, !states.isEmpty else { return }

        let repulsion: CGFloat = 80
        let attraction: CGFloat = 0.12
        let damping: CGFloat = 0.85

        var nextStates = states

        for left in data.nodes {
            guard var leftState = nextStates[left.id] else { continue }

            for right in data.nodes where right.id != left.id {
                guard let rightState = nextStates[right.id] else { continue }
                let dx = leftState.point.x - rightState.point.x
                let dy = leftState.point.y - rightState.point.y
                let distance = max(sqrt((dx * dx) + (dy * dy)), 1)
                let force = repulsion / distance
                leftState.velocity.dx += (dx / distance) * force * 0.0025
                leftState.velocity.dy += (dy / distance) * force * 0.0025
            }

            for edge in data.edges where edge.source == left.id || edge.target == left.id {
                let otherID = edge.source == left.id ? edge.target : edge.source
                guard let otherState = nextStates[otherID] else { continue }
                let dx = otherState.point.x - leftState.point.x
                let dy = otherState.point.y - leftState.point.y
                leftState.velocity.dx += dx * attraction * 0.0009
                leftState.velocity.dy += dy * attraction * 0.0009
            }

            leftState.velocity.dx *= damping
            leftState.velocity.dy *= damping
            leftState.point.x += leftState.velocity.dx
            leftState.point.y += leftState.velocity.dy

            nextStates[left.id] = leftState
        }

        states = nextStates
        objectWillChange.send()
    }

    func nodeDiameter(for node: SmritiClusterNode, minCount: Int, maxCount: Int) -> CGFloat {
        let normalized = CGFloat(node.media_count - minCount) / CGFloat(max(maxCount - minCount, 1))
        return 12 + (normalized * 20)
    }

    func nodeOpacity(for node: SmritiClusterNode, minCount: Int, maxCount: Int) -> Double {
        let normalized = Double(node.media_count - minCount) / Double(max(maxCount - minCount, 1))
        return 0.32 + (normalized * 0.56)
    }

    func displayPoint(for nodeID: Int, in size: CGSize) -> CGPoint {
        let raw = states[nodeID]?.point ?? CGPoint(x: size.width / 2, y: size.height / 2)
        return CGPoint(
            x: (raw.x * zoomScale) + additionalOffset.width,
            y: (raw.y * zoomScale) + additionalOffset.height
        )
    }

    func displayPointOptional(for nodeID: Int, in size: CGSize) -> CGPoint? {
        guard states[nodeID] != nil else { return nil }
        return displayPoint(for: nodeID, in: size)
    }

    func selectNode(at location: CGPoint, in size: CGSize) {
        guard let data = latestData else { return }
        let counts = data.nodes.map(\.media_count)
        let minCount = counts.min() ?? 1
        let maxCount = max(counts.max() ?? 1, minCount + 1)

        selectedNodeID = data.nodes.first(where: { node in
            let point = displayPoint(for: node.id, in: size)
            let radius = nodeDiameter(for: node, minCount: minCount, maxCount: maxCount) / 2
            let dx = point.x - location.x
            let dy = point.y - location.y
            return sqrt((dx * dx) + (dy * dy)) <= radius + 12
        })?.id

        objectWillChange.send()
    }

    func selectedNode(in data: SmritiMandalaData) -> SmritiClusterNode? {
        guard let selectedNodeID else { return nil }
        return data.nodes.first(where: { $0.id == selectedNodeID })
    }
}

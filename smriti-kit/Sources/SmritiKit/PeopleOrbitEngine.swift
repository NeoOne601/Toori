import SwiftUI

public struct PersonOrbit: Identifiable, Hashable {
    public let id = UUID()
    public let name: String
    public let theta: Double
    public let radius: Double
    public let size: Double
}

public class PeopleOrbitEngine {
    public init() {}
    
    public func generateOrbit(for peopleTracker: [String]) -> [PersonOrbit] {
        return peopleTracker.enumerated().map { index, name in
            // Deterministic hashing for stable coordinates
            let hash = abs(name.hashValue)
            
            // Map hash to theta (0 to 2π)
            let theta = Double(hash % 360) * .pi / 180.0
            
            // Map hash to radius (0.3 to 1.0)
            let radius = 0.3 + Double((hash / 360) % 70) / 100.0
            
            // Size mapping
            let size = 0.5 + Double((hash / 720) % 50) / 100.0
            
            return PersonOrbit(name: name, theta: theta, radius: radius, size: size)
        }
    }
}

public struct PeopleOrbitView: View {
    let orbits: [PersonOrbit]
    
    public init(orbits: [PersonOrbit]) {
        self.orbits = orbits
    }
    
    public var body: some View {
        TimelineView(.animation(minimumInterval: 1/60, paused: false)) { timeline in
            Canvas { context, size in
                let center = CGPoint(x: size.width / 2, y: size.height / 2)
                let maxRadius = min(size.width, size.height) / 2.2
                
                // Draw gravitational center
                let centerRect = CGRect(x: center.x - 10, y: center.y - 10, width: 20, height: 20)
                context.fill(Path(ellipseIn: centerRect), with: .color(Color(red: 0.4196, green: 0.3607, blue: 0.9058)))
                
                let t = timeline.date.timeIntervalSinceReferenceDate * 0.1
                
                for orbit in orbits {
                    let r = orbit.radius * maxRadius
                    let currentTheta = orbit.theta + t * (1.1 - orbit.radius) // Closer slightly faster differential
                    
                    let x = center.x + r * cos(currentTheta)
                    let y = center.y + r * sin(currentTheta)
                    
                    let nodeSize = orbit.size * 20
                    let rect = CGRect(x: x - nodeSize / 2, y: y - nodeSize / 2, width: nodeSize, height: nodeSize)
                    
                    context.fill(Path(ellipseIn: rect), with: .color(Color.white.opacity(0.8)))
                    
                    let font = Font.system(size: 10, weight: .semibold)
                    var resolvedText = Text(orbit.name).font(font)
                    context.draw(resolvedText, at: CGPoint(x: x, y: y + nodeSize + 4))
                    
                    var path = Path()
                    path.addArc(center: center, radius: r, startAngle: .zero, endAngle: .radians(2 * .pi), clockwise: false)
                    context.stroke(path, with: .color(Color.white.opacity(0.1)), lineWidth: 1)
                    
                    var linkPath = Path()
                    linkPath.move(to: center)
                    linkPath.addLine(to: CGPoint(x: x, y: y))
                    context.stroke(linkPath, with: .color(Color.white.opacity(0.05)), lineWidth: 1)
                }
            }
        }
    }
}

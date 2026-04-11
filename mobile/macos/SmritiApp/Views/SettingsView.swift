import SwiftUI

public struct SettingsView: View {
    @State private var showGemmaDownload = false

    public init() {}

    public var body: some View {
        Form {
            Section(header: Text("Memory intelligence")) {
                let manager = GemmaModelManager.shared
                let currentTier = manager.detectTier()
                let tierDisplayName: String = {
                    switch currentTier {
                    case .base: return "Essentials"
                    case .standard: return "Standard"
                    case .enhanced: return "Enhanced"
                    }
                }()
                
                Label(tierDisplayName, systemImage: "memorychip")
                
                Text(manager.selectedVariant())
                    .foregroundColor(.secondary)
                
                let features: [(String, DeviceTier)] = [
                    ("Multilingual recall", .base),
                    ("Gemma narration", .standard),
                    ("Silent journal", .standard),
                    ("Scene archaeology", .enhanced),
                    ("People orbit", .standard),
                    ("Hum to find", .base)
                ]
                let priorities: [String: Int] = ["base": 0, "standard": 1, "enhanced": 2]
                
                ForEach(features, id: \.0) { featureName, minimumTier in
                    let currentPrio = priorities[currentTier.rawValue] ?? 0
                    let minPrio = priorities[minimumTier.rawValue] ?? 0
                    let isUnlocked = currentPrio >= minPrio
                    
                    Label(featureName, systemImage: isUnlocked ? "checkmark.circle.fill" : "circle")
                        .foregroundColor(isUnlocked ? Color.smritiAccent : .secondary)
                }
                
                if (priorities[currentTier.rawValue] ?? 0) < 2 && ProcessInfo.processInfo.physicalMemory > 10_000_000_000 {
                    Button("Upgrade to Enhanced") {
                        showGemmaDownload = true
                    }
                }
            }
        }
        .sheet(isPresented: $showGemmaDownload) {
            GemmaDownloadView()
        }
        .padding()
    }
}

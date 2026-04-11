import SwiftUI
#if os(iOS)
import UIKit
#endif
#if os(macOS)
import AppKit
#endif

// Removed DownloadDelegate

public struct GemmaDownloadView: View {
    @ObservedObject var manager = GemmaModelManager.shared
    @Environment(\.dismiss) var dismiss
    
    @State private var errorMessage: String?
    @State private var downloadTimer: Timer?
    
    public init() {}
    
    public var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "brain")
                .font(.system(size: 48))
                .foregroundColor(Color.smritiAccent)
            
            Text(manager.selectedVariant() + (manager.detectTier() == .standard ? " ~1.8 GB" : (manager.detectTier() == .enhanced ? " ~3.4 GB" : "")))
                .font(.headline)
            
            Text("Gemma 4 runs entirely on your device.\nYour memories and conversations never leave your Mac/iPhone.")
                .multilineTextAlignment(.center)
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            if manager.downloadState == .downloading {
                ZStack {
                    Canvas { context, size in
                        let center = CGPoint(x: size.width / 2, y: size.height / 2)
                        let rect = CGRect(origin: CGPoint(x: center.x - 60, y: center.y - 60), size: CGSize(width: 120, height: 120))
                        var path = Path()
                        path.addArc(center: center, radius: 60, startAngle: .zero, endAngle: .radians(2 * .pi * manager.downloadProgress), clockwise: false)
                        context.stroke(path, with: .color(Color.smritiAccent), lineWidth: 4)
                    }
                    .frame(width: 140, height: 140)
                    
                    Text("\(Int(manager.downloadProgress * 100))%")
                        .font(.title2.bold())
                }
                
                let etaMinutes = Int((1.0 - manager.downloadProgress) * (manager.detectTier() == .enhanced ? 12 : 7))
                Text("about \(etaMinutes) minutes on Wi-Fi")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Button(role: .cancel) {
                    downloadTimer?.invalidate()
                    manager.downloadState = .idle
                } label: {
                    Text("Cancel")
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                
            } else if case .error(let msg) = manager.downloadState {
                Text(msg)
                    .foregroundColor(.red)
                    .font(.caption)
                
                Button {
                    startDownload()
                } label: {
                    Text("Retry")
                        .frame(maxWidth: .infinity, minHeight: 56)
                }
                .buttonStyle(.borderedProminent)
                .tint(Color.smritiAccent)
                .clipShape(RoundedRectangle(cornerRadius: 16))
            } else {
                Button {
                    startDownload()
                } label: {
                    Text("Download")
                        .frame(maxWidth: .infinity, alignment: .center)
                        // .frame(height: 56) in parent wrapper
                }
                .buttonStyle(.borderedProminent)
                .tint(Color.smritiAccent)
                .frame(maxWidth: .infinity, minHeight: 56)
                .background(Color.smritiAccent)
                .clipShape(RoundedRectangle(cornerRadius: 16))
            }
        }
        .padding(32)
        .frame(width: 400)
    }
    
    private func startDownload() {
        manager.downloadState = .downloading
        manager.downloadProgress = 0.0
        
        #if os(macOS)
        let targetDir = manager.modelDirectory(for: manager.selectedVariant())
        try? FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)
        
        let expectedSize: Double = manager.detectTier() == .enhanced ? 3_600_000_000 : 1_900_000_000
        
        downloadTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in
            let size = (try? FileManager.default.contentsOfDirectory(at: targetDir, includingPropertiesForKeys: [.fileSizeKey])
                .compactMap { try $0.resourceValues(forKeys: [.fileSizeKey]).fileSize }
                .reduce(0, +)) ?? 0
            
            DispatchQueue.main.async {
                manager.downloadProgress = min(1.0, Double(size) / expectedSize)
            }
        }
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        let script = "export PATH=\"/opt/homebrew/bin:/usr/local/bin:$PATH\"; huggingface-cli download mlx-community/\(manager.selectedVariant()) --local-dir \"\(targetDir.path)\""
        process.arguments = ["-c", script]
        
        process.terminationHandler = { p in
            DispatchQueue.main.async {
                self.downloadTimer?.invalidate()
                self.downloadTimer = nil
                if p.terminationStatus == 0 {
                    let configURL = targetDir.appendingPathComponent("config.json")
                    if FileManager.default.fileExists(atPath: configURL.path) {
                        self.manager.markModelReady()
                        self.dismiss()
                    } else {
                        self.manager.downloadState = .error("Download finished but config.json missing.")
                    }
                } else {
                    self.manager.downloadState = .error("Download failed. Check that huggingface-cli is installed.")
                }
            }
        }
        
        do {
            try process.run()
        } catch {
            downloadTimer?.invalidate()
            manager.downloadState = .error("Failed to start huggingface-cli.")
        }
        #else
        manager.downloadState = .error("Local model download is not supported on this device.")
        #endif
    }
}

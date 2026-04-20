import SwiftUI
import SmritiKit

struct SettingsView: View {
    @EnvironmentObject private var appModel: SmritiAppModel

    @Binding var backendHost: String
    @Binding var hasCompletedOnboarding: Bool

    @State private var draftHost = ""
    @State private var showGemmaDownload = false
    @State private var selectedModel = "auto"

    var body: some View {
        NavigationStack {
            ScrollView(showsIndicators: false) {
                VStack(alignment: .leading, spacing: 22) {
                    backendSection
                    modelSection
                    storageSection
                    watchFolderSection
                    capabilitySection
                    aboutSection
                }
                .padding(20)
            }
            .background(Color.smritiCanvas.ignoresSafeArea())
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .task {
                draftHost = backendHost
                await appModel.loadSettingsData()
            }
            .onTapGesture {
                appModel.hideKeyboard()
            }
        }
        .preferredColorScheme(.dark)
    }

    // MARK: - Backend Connection

    private var backendSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: "network")
                    .font(.system(size: 15))
                    .foregroundStyle(Color.smritiAccent)
                Text("Runtime Connection")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(.white)
            }

            TextField("192.168.0.x:7777", text: $draftHost)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)
                .font(.system(size: 15, weight: .medium, design: .monospaced))
                .foregroundStyle(.white)
                .padding(14)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.white.opacity(0.06))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(Color.smritiStroke, lineWidth: 0.5)
                )

            Text("Enter your Mac's local IP address and port. Toori stays local — only private LAN addresses are allowed.")
                .font(.system(size: 12))
                .foregroundStyle(.white.opacity(0.56))

            HStack(spacing: 12) {
                Button {
                    appModel.hideKeyboard()
                    if SmritiAPI.isAllowedHostString(draftHost) {
                        backendHost = draftHost
                        appModel.configureHost(draftHost)
                    }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 13))
                        Text("Save & Connect")
                    }
                }
                .buttonStyle(SettingsPillButtonStyle(fill: Color.smritiAccent))
                .disabled(!SmritiAPI.isAllowedHostString(draftHost))

                Button {
                    hasCompletedOnboarding = false
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.system(size: 13))
                        Text("Replay Onboarding")
                    }
                }
                .buttonStyle(SettingsPillButtonStyle(fill: Color.white.opacity(0.08)))
            }
        }
        .sectionCard()
    }

    // MARK: - Model Selection

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: "cpu")
                    .font(.system(size: 15))
                    .foregroundStyle(Color.smritiAccent)
                Text("Reasoning Model")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(.white)
            }

            VStack(spacing: 10) {
                modelOption(
                    id: "auto",
                    title: "Auto (Recommended)",
                    subtitle: "Selects the best model for your hardware",
                    icon: "sparkles"
                )
                modelOption(
                    id: "gemma-4-e2b-it-4bit",
                    title: "Gemma 4 e2b",
                    subtitle: "Faster, lower memory — ideal for 8GB devices",
                    icon: "hare"
                )
                modelOption(
                    id: "gemma-4-e4b-it-4bit",
                    title: "Gemma 4 e4b",
                    subtitle: "Higher quality — requires 16GB+ RAM",
                    icon: "brain.head.profile"
                )
            }

            Text("If Gemma returns empty results, Toori automatically restarts the model daemon and retries.")
                .font(.system(size: 12))
                .foregroundStyle(.white.opacity(0.44))
        }
        .sectionCard()
    }

    private func modelOption(id: String, title: String, subtitle: String, icon: String) -> some View {
        Button {
            withAnimation(.smritiSpring) {
                selectedModel = id
            }
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
        } label: {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.system(size: 16))
                    .foregroundStyle(selectedModel == id ? Color.smritiAccent : .white.opacity(0.4))
                    .frame(width: 28)

                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.white)
                    Text(subtitle)
                        .font(.system(size: 11))
                        .foregroundStyle(.white.opacity(0.5))
                }

                Spacer()

                ZStack {
                    Circle()
                        .stroke(selectedModel == id ? Color.smritiAccent : Color.white.opacity(0.2), lineWidth: 2)
                        .frame(width: 20, height: 20)
                    if selectedModel == id {
                        Circle()
                            .fill(Color.smritiAccent)
                            .frame(width: 12, height: 12)
                    }
                }
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(selectedModel == id ? Color.smritiAccent.opacity(0.08) : Color.white.opacity(0.03))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(selectedModel == id ? Color.smritiAccent.opacity(0.3) : Color.smritiStroke, lineWidth: 0.5)
            )
        }
    }

    // MARK: - Storage

    private var storageSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: "internaldrive")
                    .font(.system(size: 15))
                    .foregroundStyle(Color.smritiAccent)
                Text("Storage")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(.white)
            }

            HStack(spacing: 20) {
                StorageRingChart(usage: appModel.storageUsage)
                    .frame(width: 120, height: 120)

                VStack(alignment: .leading, spacing: 8) {
                    Text(appModel.storageUsage?.total_human ?? "Connecting...")
                        .font(.system(size: 22, weight: .semibold))
                        .foregroundStyle(.white)
                    Text("Every observation is stored locally with full geometry metadata for future recall.")
                        .font(.system(size: 13))
                        .foregroundStyle(.white.opacity(0.58))
                }
            }
        }
        .sectionCard()
    }

    // MARK: - Watch Folders

    private var watchFolderSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: "folder.badge.gearshape")
                    .font(.system(size: 15))
                    .foregroundStyle(Color.smritiAccent)
                Text("Watch Folders")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(.white)
            }

            Text("Managed from the macOS Toori app. Showing current status.")
                .font(.system(size: 13))
                .foregroundStyle(.white.opacity(0.58))

            if appModel.watchFolders.isEmpty {
                HStack(spacing: 8) {
                    Image(systemName: "folder")
                        .foregroundStyle(.white.opacity(0.3))
                    Text("No watch folders configured.")
                        .font(.system(size: 13))
                        .foregroundStyle(.white.opacity(0.44))
                }
                .padding(.vertical, 4)
            } else {
                ForEach(appModel.watchFolders) { folder in
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 8) {
                            Image(systemName: "folder.fill")
                                .font(.system(size: 12))
                                .foregroundStyle(Color.smritiAccent.opacity(0.7))
                            Text(folder.path)
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(.white)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                        HStack(spacing: 12) {
                            Label("\(folder.media_count_indexed) indexed", systemImage: "checkmark.circle")
                                .font(.system(size: 12))
                                .foregroundStyle(.white.opacity(0.56))
                            Label("\(folder.media_count_pending) pending", systemImage: "clock")
                                .font(.system(size: 12))
                                .foregroundStyle(.white.opacity(0.56))
                        }
                    }
                    .padding(.vertical, 6)
                    if folder.id != appModel.watchFolders.last?.id {
                        Divider().overlay(Color.smritiDivider)
                    }
                }
            }
        }
        .sectionCard()
    }

    // MARK: - Capabilities

    private var capabilitySection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: "eye.circle")
                    .font(.system(size: 15))
                    .foregroundStyle(Color.smritiAccent)
                Text("Reality Intelligence")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(.white)
            }

            let manager = GemmaModelManager.shared
            let currentTier = manager.detectTier()
            let tierDisplayName: String = {
                switch currentTier {
                case .base: return "Essentials"
                case .standard: return "Standard"
                case .enhanced: return "Enhanced"
                }
            }()

            HStack(spacing: 10) {
                Image(systemName: "memorychip")
                    .font(.system(size: 14))
                    .foregroundStyle(Color.smritiAccent)
                Text(tierDisplayName)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white)
                Spacer()
                Text(manager.selectedVariant())
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))
            }
            .padding(10)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color.smritiAccent.opacity(0.08))
            )

            let items: [(String, String, DeviceTier, Binding<Bool>)] = [
                ("Grounded scene analysis", "scope", .base, $appModel.openVocabEnabled),
                ("Gemma narration", "text.bubble", .standard, .constant(true)),
                ("Silent journal", "book.closed", .standard, .constant(true)),
                ("Scene archaeology", "clock.arrow.circlepath", .enhanced, .constant(true)),
                ("People orbit", "person.2", .standard, .constant(true)),
                ("Audio search", "waveform", .base, $appModel.tvlcEnabled),
            ]
            let priorities: [String: Int] = ["base": 0, "standard": 1, "enhanced": 2]

            ForEach(items, id: \.0) { featureName, icon, minimumTier, binding in
                let currentPrio = priorities[currentTier.rawValue] ?? 0
                let minPrio = priorities[minimumTier.rawValue] ?? 0
                let isUnlocked = currentPrio >= minPrio

                HStack(spacing: 12) {
                    Image(systemName: icon)
                        .font(.system(size: 13))
                        .foregroundStyle(isUnlocked ? Color.smritiAccent : .white.opacity(0.3))
                        .frame(width: 22)
                    
                    Text(featureName)
                        .font(.system(size: 14))
                        .foregroundStyle(isUnlocked ? .white.opacity(0.9) : .white.opacity(0.4))
                    
                    Spacer()
                    
                    if isUnlocked {
                        Toggle("", isOn: binding)
                            .toggleStyle(SwitchToggleStyle(tint: Color.smritiAccent))
                            .labelsHidden()
                            .scaleEffect(0.8)
                            .frame(width: 40)
                    } else {
                        Image(systemName: "lock.fill")
                            .font(.system(size: 12))
                            .foregroundStyle(.white.opacity(0.2))
                    }
                }
                .padding(.vertical, 2)
            }

            if (priorities[currentTier.rawValue] ?? 0) < 2 && ProcessInfo.processInfo.physicalMemory > 10_000_000_000 {
                Button {
                    showGemmaDownload = true
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "arrow.down.circle.fill")
                            .font(.system(size: 13))
                        Text("Upgrade to Enhanced")
                    }
                }
                .buttonStyle(SettingsPillButtonStyle(fill: Color.smritiAccent))
            }
        }
        .sectionCard()
        .sheet(isPresented: $showGemmaDownload) {
            GemmaDownloadView()
        }
    }

    // MARK: - About

    private var aboutSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "info.circle")
                    .font(.system(size: 15))
                    .foregroundStyle(Color.smritiAccent)
                Text("About Toori")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(.white)
            }

            HStack {
                Text("Version")
                    .font(.system(size: 13))
                    .foregroundStyle(.white.opacity(0.56))
                Spacer()
                Text(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0")
                    .font(.system(size: 13, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.7))
            }

            if let settingsStatusMessage = appModel.settingsStatusMessage {
                Text(settingsStatusMessage)
                    .font(.system(size: 12))
                    .foregroundStyle(.white.opacity(0.44))
            }
        }
        .sectionCard()
    }
}

// MARK: - Storage Ring Chart

private struct StorageRingChart: View {
    let usage: StorageUsageReport?

    @State private var progress = 0.0

    var body: some View {
        Canvas { context, size in
            let rect = CGRect(origin: .zero, size: size).insetBy(dx: 10, dy: 10)
            let center = CGPoint(x: rect.midX, y: rect.midY)
            let start = Angle(degrees: 135)
            let end = Angle(degrees: 405)

            var background = Path()
            background.addArc(center: center, radius: rect.width / 2, startAngle: start, endAngle: end, clockwise: false)
            context.stroke(background, with: .color(Color.white.opacity(0.08)), style: .init(lineWidth: 14, lineCap: .round))

            var foreground = Path()
            foreground.addArc(
                center: center,
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
                style: .init(lineWidth: 14, lineCap: .round)
            )
        }
        .overlay {
            VStack(spacing: 4) {
                Text((usage?.budget_pct ?? 0).formatted(.percent.precision(.fractionLength(0))))
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(.white)
                Text("used")
                    .font(.system(size: 11))
                    .foregroundStyle(.white.opacity(0.56))
            }
        }
        .onAppear {
            withAnimation(.smritiSpring.delay(0.08)) {
                progress = min(max((usage?.budget_pct ?? 0) / 100, 0), 1)
            }
        }
    }
}

// MARK: - Button Style

private struct SettingsPillButtonStyle: ButtonStyle {
    let fill: Color

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(fill.opacity(configuration.isPressed ? 0.75 : 1))
            )
    }
}

// MARK: - Section Card

private extension View {
    func sectionCard() -> some View {
        self
            .padding(18)
            .background(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(Color.white.opacity(0.06))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(Color.smritiStroke, lineWidth: 0.5)
            )
    }
}

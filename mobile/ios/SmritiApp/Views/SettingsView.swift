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
                    storageSection
                    watchFolderSection
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

import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var appModel: SmritiAppModel

    @Binding var backendHost: String
    @Binding var hasCompletedOnboarding: Bool

    @State private var draftHost = ""

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
        }
        .preferredColorScheme(.dark)
    }

    private var backendSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Backend host")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(.white)

            TextField("127.0.0.1:7777", text: $draftHost)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(.white)
                .padding(14)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.white.opacity(0.06))
                )

            Text("Smriti stays local: localhost, `.local`, and private LAN addresses only.")
                .font(.system(size: 12))
                .foregroundStyle(.white.opacity(0.56))

            HStack(spacing: 12) {
                Button("Save host") {
                    if SmritiAPI.isAllowedHostString(draftHost) {
                        backendHost = draftHost
                        appModel.configureHost(draftHost)
                    }
                }
                .buttonStyle(SettingsPillButtonStyle(fill: Color.smritiAccent))
                .disabled(!SmritiAPI.isAllowedHostString(draftHost))

                Button("Replay onboarding") {
                    hasCompletedOnboarding = false
                }
                .buttonStyle(SettingsPillButtonStyle(fill: Color.white.opacity(0.08)))
            }
        }
        .sectionCard()
    }

    private var storageSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Storage")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(.white)

            HStack(spacing: 20) {
                StorageRingChart(usage: appModel.storageUsage)
                    .frame(width: 120, height: 120)

                VStack(alignment: .leading, spacing: 8) {
                    Text(appModel.storageUsage?.total_human ?? "Waiting for runtime")
                        .font(.system(size: 22, weight: .semibold))
                        .foregroundStyle(.white)
                    Text("Your memory. On your device. That knows what surprised you.")
                        .font(.system(size: 13))
                        .foregroundStyle(.white.opacity(0.58))
                }
            }
        }
        .sectionCard()
    }

    private var watchFolderSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Watch folders")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(.white)

            Text("Folders are managed from the macOS Smriti app. iOS shows the current list read-only.")
                .font(.system(size: 13))
                .foregroundStyle(.white.opacity(0.58))

            if appModel.watchFolders.isEmpty {
                Text("No watch folders reported.")
                    .font(.system(size: 13))
                    .foregroundStyle(.white.opacity(0.44))
            } else {
                ForEach(appModel.watchFolders) { folder in
                    VStack(alignment: .leading, spacing: 6) {
                        Text(folder.path)
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(.white)
                        Text("\(folder.media_count_indexed) indexed • \(folder.media_count_pending) pending")
                            .font(.system(size: 12))
                            .foregroundStyle(.white.opacity(0.56))
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

    private var aboutSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("About Smriti")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(.white)
            Text(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0")
                .font(.system(size: 13))
                .foregroundStyle(.white.opacity(0.56))
            if let settingsStatusMessage = appModel.settingsStatusMessage {
                Text(settingsStatusMessage)
                    .font(.system(size: 12))
                    .foregroundStyle(.white.opacity(0.44))
            }
        }
        .sectionCard()
    }
}

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

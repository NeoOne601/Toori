import SwiftUI

struct OnboardingSheet: View {
    @EnvironmentObject private var appModel: SmritiAppModel
    @State private var ringTrim: CGFloat = 0.24
    @State private var ringRotation = 0.0

    var body: some View {
        ZStack {
            SmritiGlassBackground()
            VStack(alignment: .leading, spacing: 20) {
                header
                content
                Spacer(minLength: 0)
                footer
            }
            .padding(24)
        }
        .onAppear {
            Task {
                await appModel.refreshWatchFolders()
            }
            if appModel.onboardingStep == .ingesting {
                appModel.startOnboardingEventStreamIfNeeded()
            }
        }
        .onChange(of: appModel.onboardingStep) { _, newValue in
            if newValue == .ingesting {
                appModel.resetOnboardingCounter()
                appModel.startOnboardingEventStreamIfNeeded()
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Welcome to Smriti")
                .font(.system(size: 28, weight: .semibold))
            Text("Privacy is architecture. Everything stays local on this Mac.")
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var content: some View {
        switch appModel.onboardingStep {
        case .photos:
            onboardingCard(
                title: "Show Smriti your Photos Library",
                body: "Start with your Pictures folder so the memory graph has real material to grow from."
            ) {
                Button("Choose Photos Library") {
                    Task {
                        await appModel.pickAndAddWatchFolder(
                            defaultURL: FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Pictures"),
                            nextStep: .folders
                        )
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(Color.smritiAccent)
            }

        case .folders:
            onboardingCard(
                title: "Add any folder",
                body: "You can add more folders now, or continue once Smriti has enough places to watch."
            ) {
                VStack(alignment: .leading, spacing: 14) {
                    Button("Add folder") {
                        Task {
                            await appModel.pickAndAddWatchFolder(defaultURL: nil, nextStep: .folders)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Color.smritiAccent)

                    if appModel.watchFolders.isEmpty {
                        Text("No watch folders added yet.")
                            .font(.system(size: 13))
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(appModel.watchFolders) { folder in
                            HStack {
                                Image(systemName: folder.watchdog_active ? "checkmark.circle.fill" : "folder")
                                    .foregroundStyle(folder.watchdog_active ? Color.smritiAccent : .secondary)
                                Text(folder.path)
                                    .font(.system(size: 11))
                                    .lineLimit(1)
                                Spacer()
                                Text("\(folder.media_count_indexed)/\(folder.media_count_total)")
                                    .font(.system(size: 11, weight: .medium))
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    Button("Continue to ingestion") {
                        appModel.onboardingStep = .ingesting
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.smritiAccent)
                }
            }

        case .ingesting:
            onboardingCard(
                title: "Indexing your memories",
                body: "Smriti is listening for observation.created events from the local event stream."
            ) {
                HStack(spacing: 18) {
                    ZStack {
                        Circle()
                            .stroke(Color.white.opacity(0.08), lineWidth: 10)
                            .frame(width: 78, height: 78)
                        Circle()
                            .trim(from: 0.05, to: ringTrim)
                            .stroke(
                                LinearGradient(
                                    colors: [Color.smritiTeal, Color.smritiAccent],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                style: StrokeStyle(lineWidth: 10, lineCap: .round)
                            )
                            .frame(width: 78, height: 78)
                            .rotationEffect(.degrees(ringRotation))
                    }
                    VStack(alignment: .leading, spacing: 8) {
                        Text("\(appModel.ingestionCount)")
                            .font(.system(size: 22, weight: .semibold))
                        Text(appModel.onboardingStatus)
                            .font(.system(size: 13))
                            .foregroundStyle(.secondary)
                    }
                }
                .onAppear {
                    withAnimation(.smritiSpring.repeatForever(autoreverses: false)) {
                        ringTrim = 0.94
                        ringRotation = 360
                    }
                }
            }
        }
    }

    private var footer: some View {
        HStack {
            Button("Skip") {
                appModel.skipOnboarding()
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)

            Spacer()

            if appModel.onboardingStep == .ingesting {
                Button("Done") {
                    appModel.finishOnboarding()
                }
                .buttonStyle(.borderedProminent)
                .tint(Color.smritiAccent)
            }
        }
    }

    private func onboardingCard<Content: View>(
        title: String,
        body: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title)
                .font(.system(size: 22, weight: .semibold))
            Text(body)
                .font(.system(size: 17))
                .foregroundStyle(.secondary)
            content()
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.white.opacity(0.05))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color.smritiGlassStroke, lineWidth: 0.5)
        )
    }
}

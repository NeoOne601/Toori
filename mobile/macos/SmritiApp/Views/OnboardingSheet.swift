import Photos
import SwiftUI

struct OnboardingSheet: View {
    private enum FolderStage {
        case pictures
        case additional
    }

    @EnvironmentObject private var appModel: SmritiAppModel
    @StateObject private var photoLibrary = SmritiPhotoLibrary()
    @State private var ringTrim: CGFloat = 0.24
    @State private var ringRotation = 0.0
    @State private var folderStage: FolderStage = .pictures

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
            synchronizeFolderStage()
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
        .onChange(of: appModel.watchFolders.count) { _, _ in
            synchronizeFolderStage()
        }
    }

    private var isPhotosAuthorized: Bool {
        photoLibrary.authorizationStatus == .authorized || photoLibrary.authorizationStatus == .limited
    }

    private var canFinish: Bool {
        isPhotosAuthorized || !appModel.watchFolders.isEmpty
    }

    private var currentStepNumber: Int {
        switch appModel.onboardingStep {
        case .photos:
            return 1
        case .folders:
            return folderStage == .pictures ? 2 : 3
        case .ingesting:
            return 4
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Welcome to Smriti")
                .font(.system(size: 28, weight: .semibold))
            Text("Step \(currentStepNumber) of 4")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Color.smritiAccent)
            Text("Privacy is architecture. Everything stays local on this Mac.")
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var content: some View {
        switch appModel.onboardingStep {
        case .photos:
            photosPermissionCard
        case .folders:
            switch folderStage {
            case .pictures:
                picturesFolderCard
            case .additional:
                additionalFolderCard
            }
        case .ingesting:
            ingestingCard
        }
    }

    private var photosPermissionCard: some View {
        onboardingCard(
            icon: "photo.on.rectangle.angled",
            title: "Connect your Photos",
            body: "Smriti privately watches your Photos library and remembers what surprised you. Your photos never leave your device."
        ) {
            VStack(alignment: .leading, spacing: 14) {
                if isPhotosAuthorized {
                    Text("Importing \(photoLibrary.newAssetCount) memories…")
                        .font(.system(size: 17, weight: .medium))
                        .foregroundStyle(.white)
                        .contentTransition(.numericText())

                    Button("Continue to folders") {
                        appModel.onboardingStep = .folders
                        synchronizeFolderStage()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Color.smritiAccent)
                } else {
                    Button("Allow Photos Access") {
                        Task { @MainActor in
                            await photoLibrary.requestAccess()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Color.smritiAccent)
                }

                Button("Use folders instead") {
                    appModel.onboardingStep = .folders
                    folderStage = .pictures
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
            }
        }
    }

    private var picturesFolderCard: some View {
        onboardingCard(
            icon: "photo.stack",
            title: "Show Smriti your Photos Library",
            body: "Start with your Pictures folder so the memory graph has real material to grow from."
        ) {
            VStack(alignment: .leading, spacing: 14) {
                Button("Choose Pictures folder") {
                    Task {
                        let countBefore = appModel.watchFolders.count
                        await appModel.pickAndAddWatchFolder(
                            defaultURL: FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Pictures"),
                            nextStep: nil
                        )
                        if appModel.watchFolders.count > countBefore {
                            folderStage = .additional
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(Color.smritiAccent)

                if !appModel.watchFolders.isEmpty {
                    Button("Continue") {
                        folderStage = .additional
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.smritiAccent)
                }
            }
        }
    }

    private var additionalFolderCard: some View {
        onboardingCard(
            icon: "folder.badge.plus",
            title: "Add any folder",
            body: "You can add more folders now, or continue once Smriti has enough places to watch."
        ) {
            VStack(alignment: .leading, spacing: 14) {
                Button("Add folder") {
                    Task {
                        await appModel.pickAndAddWatchFolder(defaultURL: nil, nextStep: nil)
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
    }

    private var ingestingCard: some View {
        onboardingCard(
            icon: "sparkles.rectangle.stack",
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

    private var footer: some View {
        HStack {
            Button("Skip") {
                appModel.skipOnboarding()
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)

            Spacer()

            if canFinish {
                Button("Done") {
                    appModel.finishOnboarding()
                }
                .buttonStyle(.borderedProminent)
                .tint(Color.smritiAccent)
            }
        }
    }

    private func synchronizeFolderStage() {
        if appModel.watchFolders.isEmpty {
            if appModel.onboardingStep == .folders {
                folderStage = .pictures
            }
        } else {
            folderStage = .additional
        }
    }

    private func onboardingCard<Content: View>(
        icon: String,
        title: String,
        body: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Image(systemName: icon)
                .font(.system(size: 64, weight: .medium))
                .foregroundStyle(Color.smritiAccent)
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

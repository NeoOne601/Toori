import Photos
import SwiftUI

struct OnboardingFlow: View {
    let onGetStarted: () -> Void

    @StateObject private var photoLibrary: SmritiPhotoLibrary
    @State private var selection = 0
    @State private var buttonScale: CGFloat = 1

    init(onGetStarted: @escaping () -> Void) {
        self.onGetStarted = onGetStarted
        let host = UserDefaults.standard.string(forKey: "smriti.backendHost") ?? "127.0.0.1:7777"
        _photoLibrary = StateObject(wrappedValue: SmritiPhotoLibrary(host: host))
    }

    var body: some View {
        ZStack {
            Color.smritiCanvas.ignoresSafeArea()

            TabView(selection: $selection) {
                OnboardingCard(
                    title: "Connect your Photos",
                    subtitle: "Smriti privately watches your Photos library and remembers what surprised you. Your photos never leave your device."
                ) {
                    PhotosPermissionScene()
                } footer: {
                    VStack(spacing: 12) {
                        if isPhotosAuthorized {
                            Text("Importing \(photoLibrary.newAssetCount) memories…")
                                .font(.system(size: 15, weight: .medium))
                                .foregroundStyle(.white.opacity(0.84))
                                .contentTransition(.numericText())

                            Button("Continue") {
                                withAnimation(.smritiSpring) {
                                    selection = 1
                                }
                            }
                            .buttonStyle(OnboardingCTAButtonStyle())
                        } else {
                            Button("Allow Photos Access") {
                                Task { @MainActor in
                                    await photoLibrary.requestAccess()
                                    if isPhotosAuthorized {
                                        withAnimation(.smritiSpring) {
                                            selection = 1
                                        }
                                    }
                                }
                            }
                            .buttonStyle(OnboardingCTAButtonStyle())
                        }

                        Button("Continue without Photos") {
                            withAnimation(.smritiSpring) {
                                selection = 1
                            }
                        }
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(.white.opacity(0.68))
                    }
                }
                .tag(0)

                OnboardingCard(
                    title: "Your memory, alive",
                    subtitle: "Smriti watches what you see and remembers what surprised you."
                ) {
                    BreathingOrbScene()
                }
                .tag(1)

                OnboardingCard(
                    title: "Ask in plain language",
                    subtitle: "Say 'find the rainy afternoon with the telescope' and Smriti finds it."
                ) {
                    VoiceWaveScene()
                }
                .tag(2)

                OnboardingCard(
                    title: "Hum to find",
                    subtitle: "Sing a few notes. Smriti finds the moment."
                ) {
                    HumTransitionScene()
                } footer: {
                    Button {
                        withAnimation(.smritiSpring) {
                            buttonScale = 0.97
                        }
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
                            withAnimation(.smritiSpring) {
                                buttonScale = 1
                            }
                            onGetStarted()
                        }
                    } label: {
                        Text("Get started")
                            .font(.system(size: 17, weight: .semibold))
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity)
                            .frame(height: 56)
                            .background(
                                RoundedRectangle(cornerRadius: 16, style: .continuous)
                                    .fill(Color.smritiAccent)
                            )
                    }
                    .scaleEffect(buttonScale)
                }
                .tag(3)
            }
            .tabViewStyle(.page(indexDisplayMode: .always))
        }
        .preferredColorScheme(.dark)
    }

    private var isPhotosAuthorized: Bool {
        photoLibrary.authorizationStatus == .authorized || photoLibrary.authorizationStatus == .limited
    }
}

private struct OnboardingCTAButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 17, weight: .semibold))
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .frame(height: 56)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.smritiAccent.opacity(configuration.isPressed ? 0.82 : 1))
            )
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.smritiSpring, value: configuration.isPressed)
    }
}

private struct OnboardingCard<Scene: View, Footer: View>: View {
    let title: String
    let subtitle: String
    @ViewBuilder let scene: Scene
    @ViewBuilder var footer: Footer

    init(title: String, subtitle: String, @ViewBuilder scene: () -> Scene, @ViewBuilder footer: () -> Footer = { EmptyView() }) {
        self.title = title
        self.subtitle = subtitle
        self.scene = scene()
        self.footer = footer()
    }

    var body: some View {
        VStack(spacing: 28) {
            Spacer()
            scene
                .frame(height: 280)
            VStack(spacing: 12) {
                Text(title)
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundStyle(.white)
                Text(subtitle)
                    .font(.system(size: 17))
                    .foregroundStyle(.white.opacity(0.62))
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 320)
            }
            Spacer()
            footer
                .padding(.horizontal, 24)
                .padding(.bottom, 40)
        }
    }
}

private struct PhotosPermissionScene: View {
    var body: some View {
        ZStack {
            Circle()
                .fill(Color.smritiAccent.opacity(0.18))
                .frame(width: 180, height: 180)
                .blur(radius: 18)

            Circle()
                .stroke(Color.smritiAccent.opacity(0.56), lineWidth: 8)
                .frame(width: 170, height: 170)

            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 64, weight: .medium))
                .foregroundStyle(Color.smritiAccent)
        }
    }
}

private struct BreathingOrbScene: View {
    var body: some View {
        TimelineView(.animation) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            let sine = (sin(time * (.pi / 2)) + 1) / 2
            let scale = 0.97 + sine * 0.03

            ZStack {
                Circle()
                    .stroke(Color.smritiAccent.opacity(0.5), lineWidth: 6 + sine * 8)
                    .frame(width: 190, height: 190)
                    .blur(radius: 8)

                Circle()
                    .fill(
                        LinearGradient(
                            colors: [Color.white.opacity(0.12), Color.smritiAccent.opacity(0.3)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 164, height: 164)
                    .overlay(
                        Image(systemName: "brain.head.profile.fill")
                            .font(.system(size: 60, weight: .medium))
                            .foregroundStyle(.white.opacity(0.88))
                    )
            }
            .scaleEffect(scale)
        }
    }
}

private struct VoiceWaveScene: View {
    var body: some View {
        TimelineView(.animation) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate

            HStack(spacing: 10) {
                ForEach(0..<7, id: \.self) { index in
                    let phase = time * 1.8 + Double(index) * 0.16
                    let height = 26 + CGFloat((sin(phase * .pi * 2) + 1) / 2) * 110
                    Capsule(style: .continuous)
                        .fill(index.isMultiple(of: 2) ? Color.smritiAccent : Color.smritiTeal)
                        .frame(width: 14, height: height)
                }
            }
        }
    }
}

private struct HumTransitionScene: View {
    var body: some View {
        TimelineView(.animation) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            let progress = (sin(time * 1.2) + 1) / 2

            VStack(spacing: 22) {
                HStack(spacing: 6) {
                    ForEach(0..<5, id: \.self) { index in
                        let phase = time * 2.2 + Double(index) * 0.24
                        Capsule(style: .continuous)
                            .fill(Color.smritiAccent)
                            .frame(width: 8, height: 16 + CGFloat((sin(phase * .pi * 2) + 1) / 2) * 40)
                    }
                }
                .opacity(1 - progress * 0.4)

                LazyVGrid(columns: Array(repeating: GridItem(.fixed(54), spacing: 10), count: 3), spacing: 10) {
                    ForEach(0..<9, id: \.self) { index in
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(index.isMultiple(of: 3) ? Color.smritiAccent.opacity(0.45) : Color.white.opacity(0.08))
                            .frame(width: 54, height: 54)
                            .overlay(
                                Image(systemName: "photo")
                                    .font(.system(size: 16, weight: .medium))
                                    .foregroundStyle(.white.opacity(0.74))
                            )
                            .scaleEffect(0.88 + progress * 0.12)
                            .opacity(0.5 + progress * 0.5)
                    }
                }
            }
        }
    }
}

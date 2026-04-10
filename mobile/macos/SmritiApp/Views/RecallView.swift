import SwiftUI

struct RecallView: View {
    @EnvironmentObject private var appModel: SmritiAppModel
    @State private var recallTask: Task<Void, Never>?
    @State private var dragTranslation: CGFloat = 0

    var body: some View {
        SmritiGlassSurface {
            VStack(alignment: .leading, spacing: 18) {
                Picker("Surface", selection: $appModel.selectedTab) {
                    ForEach(SmritiAppModel.RootTab.allCases) { tab in
                        Text(tab.rawValue).tag(tab)
                    }
                }
                .pickerStyle(.segmented)

                ZStack {
                    if appModel.selectedTab == .recall {
                        recallSurface
                            .transition(
                                .asymmetric(
                                    insertion: .move(edge: .leading).combined(with: .opacity),
                                    removal: .move(edge: .trailing).combined(with: .opacity)
                                )
                            )
                    } else {
                        MandalaView()
                            .transition(
                                .asymmetric(
                                    insertion: .move(edge: .trailing).combined(with: .opacity),
                                    removal: .move(edge: .leading).combined(with: .opacity)
                                )
                            )
                    }
                }
                .animation(.smritiSpring, value: appModel.selectedTab)
                .gesture(
                    DragGesture(minimumDistance: 20)
                        .onChanged { value in
                            dragTranslation = value.translation.width
                        }
                        .onEnded { value in
                            defer { dragTranslation = 0 }
                            if value.translation.width < -60 {
                                appModel.selectedTab = .mandala
                            } else if value.translation.width > 60 {
                                appModel.selectedTab = .recall
                            }
                        }
                )
            }
        }
        .padding(20)
        .onChange(of: appModel.recallQuery) { _, _ in
            scheduleRecall()
        }
        .onChange(of: appModel.selectedTab) { _, newValue in
            if newValue == .mandala {
                Task {
                    await appModel.loadMandalaIfNeeded()
                }
            }
        }
        .onAppear {
            scheduleRecall()
        }
    }

    private var recallSurface: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(Color.smritiAccent)
                TextField("Search your memory…", text: $appModel.recallQuery)
                    .textFieldStyle(.plain)
                    .font(.system(size: 17))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.white.opacity(0.06))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(Color.smritiGlassStroke, lineWidth: 0.5)
            )

            Group {
                if appModel.recallQuery.trimmingCharacters(in: .whitespacesAndNewlines).count < 2 {
                    RecallEmptyState(message: "Your memory is building. Show Smriti what matters.")
                } else if appModel.isSearching {
                    ProgressView()
                        .controlSize(.small)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if appModel.recallResults.isEmpty {
                    RecallEmptyState(message: "Your memory is building. Show Smriti what matters.")
                } else {
                    ScrollView(.vertical, showsIndicators: false) {
                        LazyVStack(spacing: 12) {
                            ForEach(Array(appModel.recallResults.enumerated()), id: \.element.id) { index, item in
                                SmritiRecallCard(item: item, index: index) {
                                    appModel.openDetail(for: item)
                                }
                            }
                        }
                        .padding(.bottom, 4)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func scheduleRecall() {
        recallTask?.cancel()
        recallTask = Task {
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            await appModel.runRecall()
        }
    }
}

private struct RecallEmptyState: View {
    let message: String
    @State private var shimmer = false

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "sparkles")
                .font(.system(size: 32, weight: .medium))
                .foregroundStyle(
                    LinearGradient(
                        colors: [Color.smritiTeal, Color.smritiAccent],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .scaleEffect(shimmer ? 1.08 : 0.94)
                .opacity(shimmer ? 1 : 0.65)
                .animation(.smritiSpring.repeatForever(autoreverses: true), value: shimmer)

            Text(message)
                .font(.system(size: 17))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 240)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            shimmer = true
        }
    }
}

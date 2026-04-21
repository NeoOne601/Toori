import SwiftUI

@main
struct TooriLensApp: App {
    @StateObject private var viewModel = LensAppViewModel()
    @State private var showGemmaOnboarding = false

    var body: some Scene {
        WindowGroup {
            RealityCheckView()
                .environmentObject(viewModel)
                .task { await viewModel.bootstrap() }
        }
    }
}

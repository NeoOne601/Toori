import SwiftUI

@main
struct TooriLensApp: App {
    @StateObject private var viewModel = LensAppViewModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(viewModel)
        }
    }
}

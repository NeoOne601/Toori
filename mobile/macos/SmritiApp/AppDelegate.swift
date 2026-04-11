import Cocoa
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSPopoverDelegate, NSWindowDelegate {
    private let appModel = SmritiAppModel.shared
    private let daemonPIDKey = "smriti.daemonPID"
    private let daemonOwnedKey = "smriti.daemonLaunchedByApp"

    private var statusItem: NSStatusItem!
    private var popover: NSPopover!
    private var detailPanel: NSPanel?
    private var daemonProcess: Process?
    private var hasCheckedBackend = false
    private var healthTask: Task<Void, Never>?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        configureModelCallbacks()
        configureStatusItem()
        configurePopover()
    }

    func applicationWillTerminate(_ notification: Notification) {
        healthTask?.cancel()
        if UserDefaults.standard.bool(forKey: daemonOwnedKey) {
            let savedPID = UserDefaults.standard.integer(forKey: daemonPIDKey)
            if savedPID > 0 {
                kill(pid_t(savedPID), SIGTERM)
            }
        }
    }

    private func configureModelCallbacks() {
        appModel.detailPresenter = { [weak self] item in
            self?.showDetailPanel(for: item)
        }
        appModel.detailDismiss = { [weak self] in
            self?.detailPanel?.close()
        }
        appModel.folderPicker = { [weak self] defaultURL in
            await self?.selectDirectory(defaultURL: defaultURL)
        }
        appModel.sharePresenter = { [weak self] items in
            self?.showSharingPicker(items: items)
        }
    }

    private func configureStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = statusItem.button {
            let image = NSImage(systemSymbolName: "brain.head.profile.fill", accessibilityDescription: "Smriti")
            image?.isTemplate = false
            button.image = image
            button.contentTintColor = NSColor(smritiAccent: ())
            button.action = #selector(togglePopover(_:))
            button.target = self
        }
    }

    private func configurePopover() {
        popover = NSPopover()
        popover.behavior = .transient
        popover.animates = true
        popover.delegate = self
        popover.contentSize = NSSize(width: 380, height: 520)

        let rootView = PopoverRootView()
            .environmentObject(appModel)
        let hostingController = NSHostingController(rootView: rootView)
        hostingController.view.wantsLayer = true
        hostingController.view.layer?.backgroundColor = NSColor.clear.cgColor
        popover.contentViewController = hostingController
    }

    @objc
    private func togglePopover(_ sender: AnyObject?) {
        guard let button = statusItem.button else { return }
        if popover.isShown {
            popover.performClose(sender)
            return
        }
        popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
        withAnimation(.smritiSpring) {
            button.highlight(true)
        }
        if !hasCheckedBackend {
            hasCheckedBackend = true
            ensureBackendReady()
        }
    }

    func popoverDidClose(_ notification: Notification) {
        statusItem.button?.highlight(false)
    }

    private func ensureBackendReady() {
        healthTask?.cancel()
        healthTask = Task { @MainActor in
            appModel.backendPhase = .checking
            if await isBackendHealthy() {
                appModel.backendPhase = .ready
                appModel.maybePresentOnboarding()
                return
            }
            appModel.backendPhase = .launching
            launchDaemonIfNeeded()
            let deadline = Date().addingTimeInterval(20)
            while Date() < deadline {
                if await isBackendHealthy() {
                    appModel.backendPhase = .ready
                    appModel.maybePresentOnboarding()
                    return
                }
                try? await Task.sleep(for: .milliseconds(500))
            }
            appModel.backendPhase = .failed("The local runtime didn’t become healthy in time.")
        }
    }

    private func launchDaemonIfNeeded() {
        guard daemonProcess == nil else { return }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["/Users/macuser/toori/scripts/run_runtime.sh"]
        process.currentDirectoryURL = URL(fileURLWithPath: "/Users/macuser/toori")
        var environment = ProcessInfo.processInfo.environment
        environment["TOORI_DATA_DIR"] = ".toori"
        environment["TOORI_HOST"] = "127.0.0.1"
        environment["TOORI_PORT"] = "7777"
        process.environment = environment
        process.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                self?.daemonProcess = nil
            }
        }
        do {
            try process.run()
            daemonProcess = process
            UserDefaults.standard.set(Int(process.processIdentifier), forKey: daemonPIDKey)
            UserDefaults.standard.set(true, forKey: daemonOwnedKey)
        } catch {
            appModel.backendPhase = .failed("Failed to launch the local runtime.")
        }
    }

    private func isBackendHealthy() async -> Bool {
        do {
            return try await appModel.api.healthCheck()
        } catch {
            UserDefaults.standard.set(false, forKey: daemonOwnedKey)
            return false
        }
    }

    private func showDetailPanel(for item: SmritiRecallItem) {
        let root = DetailSheet(item: item)
            .environmentObject(appModel)

        let controller = NSHostingController(rootView: root)
        controller.view.wantsLayer = true
        controller.view.layer?.backgroundColor = NSColor.clear.cgColor

        let panel = detailPanel ?? NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 720, height: 820),
            styleMask: [.titled, .closable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        panel.titleVisibility = .hidden
        panel.titlebarAppearsTransparent = true
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.isFloatingPanel = true
        panel.hasShadow = true
        panel.delegate = self
        panel.contentViewController = controller
        panel.center()
        panel.makeKeyAndOrderFront(nil)
        detailPanel = panel
    }

    func windowWillClose(_ notification: Notification) {
        if let panel = notification.object as? NSPanel, panel == detailPanel {
            appModel.selectedRecallItem = nil
        }
    }

    private func selectDirectory(defaultURL: URL?) async -> URL? {
        await withCheckedContinuation { continuation in
            DispatchQueue.main.async {
                let panel = NSOpenPanel()
                panel.canChooseDirectories = true
                panel.canChooseFiles = false
                panel.allowsMultipleSelection = false
                panel.directoryURL = defaultURL
                panel.begin { response in
                    continuation.resume(returning: response == .OK ? panel.url : nil)
                }
            }
        }
    }

    private func showSharingPicker(items: [Any]) {
        guard let contentView = detailPanel?.contentView else { return }
        let picker = NSSharingServicePicker(items: items)
        let point = contentView.convert(contentView.bounds.centerPoint, to: nil)
        picker.show(relativeTo: NSRect(origin: point, size: .zero), of: contentView, preferredEdge: .minY)
    }
}

private struct PopoverRootView: View {
    @EnvironmentObject var appModel: SmritiAppModel
    @AppStorage("smriti.hasCompletedOnboarding") private var hasCompletedOnboarding = false
    @State private var showGemmaDownload = false

    var body: some View {
        SmritiRootView()
            .sheet(isPresented: $showGemmaDownload) {
                GemmaDownloadView()
            }
            .onChange(of: hasCompletedOnboarding) { _, _ in checkGemma() }
            .onAppear { checkGemma() }
    }

    private func checkGemma() {
        // Only trigger if backend is ready and onboarding is done
        // Note: appModel.backendPhase might not be visible here if it's enum, 
        // but hasCompletedOnboarding is true. We'll simply check GemmaManager.
        if hasCompletedOnboarding {
            let manager = GemmaModelManager.shared
            if !manager.isModelPresent(), manager.detectTier() != .base {
                showGemmaDownload = true
            }
        }
    }
}

private extension NSRect {
    var centerPoint: NSPoint {
        NSPoint(x: midX, y: midY)
    }
}

extension NSColor {
    convenience init(smritiAccent _: Void) {
        self.init(red: 0.4196, green: 0.3607, blue: 0.9058, alpha: 1.0)
    }
}

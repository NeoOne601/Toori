import Foundation
import SwiftUI

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

@MainActor
public final class SmritiEventStore: ObservableObject {
    public enum ConnectionState: Equatable, Sendable {
        case idle
        case connecting
        case live
        case reconnecting(Int)
        case failed(String)
    }

    @Published public private(set) var observations: [ObservationSummary] = []
    @Published public private(set) var connectionState: ConnectionState = .idle

    private let coordinator = EventCoordinator()
    private var lifecycleTokens: [NSObjectProtocol] = []
    private var bootstrapTask: Task<Void, Never>?
    private var isRunning = false
    private var host: String

    public init(host: String = "127.0.0.1:7777") {
        self.host = host
        installLifecycleObservers()
    }

    deinit {
        lifecycleTokens.forEach(NotificationCenter.default.removeObserver)
    }

    public var orbSlots: [PulseOrbSlot] {
        let mapped = observations.prefix(9).map(PulseOrbSlot.observation)
        let placeholderStart = mapped.count
        let placeholders = (placeholderStart..<9).map(PulseOrbSlot.placeholder)
        return Array(mapped) + placeholders
    }

    public func updateHost(_ host: String) {
        self.host = host
        if isRunning {
            restart()
        }
    }

    public func start() {
        guard !isRunning else { return }
        isRunning = true
        bootstrap()
        Task { await coordinator.start(host: host, eventSink: Self.makeEventSink(store: self)) }
    }

    public func stop() {
        isRunning = false
        bootstrapTask?.cancel()
        bootstrapTask = nil
        Task { await coordinator.stop() }
        connectionState = .idle
    }

    public func pauseForBackground() {
        guard isRunning else { return }
        Task { await coordinator.pause() }
        connectionState = .idle
    }

    public func resumeFromForeground() {
        guard isRunning else { return }
        Task { await coordinator.resume() }
    }

    private func restart() {
        stop()
        start()
    }

    private func bootstrap() {
        bootstrapTask?.cancel()
        bootstrapTask = Task.detached(priority: .utility) { [host] in
            do {
                let api = try SmritiAPI(host: host)
                let response = try await api.listObservations(limit: 9, summaryOnly: true)
                await MainActor.run { [weak self] in
                    self?.observations = Array(response.observations.prefix(9))
                }
            } catch {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    if self.observations.isEmpty {
                        self.connectionState = .failed(error.localizedDescription)
                    }
                }
            }
        }
    }

    private func installLifecycleObservers() {
        #if canImport(UIKit)
        lifecycleTokens.append(
            NotificationCenter.default.addObserver(
                forName: UIApplication.willResignActiveNotification,
                object: nil,
                queue: nil
            ) { [weak self] _ in
                Task { @MainActor in
                    self?.pauseForBackground()
                }
            }
        )
        lifecycleTokens.append(
            NotificationCenter.default.addObserver(
                forName: UIApplication.didBecomeActiveNotification,
                object: nil,
                queue: nil
            ) { [weak self] _ in
                Task { @MainActor in
                    self?.resumeFromForeground()
                }
            }
        )
        #endif

        #if canImport(AppKit)
        lifecycleTokens.append(
            NotificationCenter.default.addObserver(
                forName: NSApplication.willTerminateNotification,
                object: nil,
                queue: nil
            ) { [weak self] _ in
                Task { @MainActor in
                    self?.stop()
                }
            }
        )
        #endif
    }

    private func insert(_ observation: ObservationSummary) {
        observations.removeAll { $0.id == observation.id }
        observations.insert(observation, at: 0)
        if observations.count > 9 {
            observations = Array(observations.prefix(9))
        }
    }

    private static func makeEventSink(store: SmritiEventStore) -> EventCoordinator.EventSink {
        EventCoordinator.EventSink(
            onConnectionState: { state in
                await MainActor.run {
                    store.connectionState = state
                }
            },
            onObservation: { observation in
                await MainActor.run {
                    withAnimation(.smritiSpring) {
                        store.insert(observation)
                    }
                }
            }
        )
    }
}

public actor EventCoordinator {
    public struct EventSink: Sendable {
        public let onConnectionState: @Sendable (SmritiEventStore.ConnectionState) async -> Void
        public let onObservation: @Sendable (ObservationSummary) async -> Void

        public init(
            onConnectionState: @Sendable @escaping (SmritiEventStore.ConnectionState) async -> Void,
            onObservation: @Sendable @escaping (ObservationSummary) async -> Void
        ) {
            self.onConnectionState = onConnectionState
            self.onObservation = onObservation
        }
    }

    private let schedule = [1, 2, 4, 8, 16, 30]
    private var host = "127.0.0.1:7777"
    private var socket: URLSessionWebSocketTask?
    private var sink: EventSink?
    private var loopTask: Task<Void, Never>?
    private var heartbeatTask: Task<Void, Never>?
    private var shouldRun = false
    private var isPaused = false
    private var backoffIndex = 0

    public init() {}

    public func start(host: String, eventSink: EventSink) {
        self.host = host
        self.sink = eventSink
        shouldRun = true
        isPaused = false
        if loopTask == nil {
            loopTask = Task { [weak self] in
                await self?.runLoop()
            }
        }
    }

    public func pause() {
        guard shouldRun else { return }
        isPaused = true
        closeSocket()
        heartbeatTask?.cancel()
        heartbeatTask = nil
    }

    public func resume() {
        guard shouldRun else { return }
        isPaused = false
        if loopTask == nil {
            loopTask = Task { [weak self] in
                await self?.runLoop()
            }
        }
    }

    public func stop() {
        shouldRun = false
        isPaused = false
        backoffIndex = 0
        closeSocket()
        heartbeatTask?.cancel()
        heartbeatTask = nil
        loopTask?.cancel()
        loopTask = nil
    }

    private func runLoop() async {
        defer { loopTask = nil }
        while shouldRun && !Task.isCancelled {
            if isPaused {
                await sink?.onConnectionState(.idle)
                return
            }

            do {
                await sink?.onConnectionState(.connecting)
                let api = try SmritiAPI(host: host)
                let socket = URLSession.shared.webSocketTask(with: api.websocketURL)
                self.socket = socket
                socket.resume()
                backoffIndex = 0
                await sink?.onConnectionState(.live)
                heartbeatTask?.cancel()
                heartbeatTask = Task { [weak self] in
                    await self?.heartbeatLoop()
                }
                try await receiveLoop()
                heartbeatTask?.cancel()
                heartbeatTask = nil
                closeSocket()

                if shouldRun && !isPaused {
                    throw URLError(.networkConnectionLost)
                }
            } catch is CancellationError {
                closeSocket()
                heartbeatTask?.cancel()
                heartbeatTask = nil
                if !shouldRun || isPaused {
                    await sink?.onConnectionState(.idle)
                    return
                }
            } catch {
                closeSocket()
                heartbeatTask?.cancel()
                heartbeatTask = nil
                if !shouldRun || isPaused {
                    await sink?.onConnectionState(.idle)
                    return
                }
                let delay = schedule[min(backoffIndex, schedule.count - 1)]
                backoffIndex += 1
                await sink?.onConnectionState(.reconnecting(delay))
                try? await Task.sleep(for: .seconds(delay))
            }
        }
        await sink?.onConnectionState(.idle)
    }

    private func receiveLoop() async throws {
        guard let socket else { throw CancellationError() }
        let decoder = SmritiAPI.makeDecoder()
        while shouldRun && !isPaused && !Task.isCancelled {
            let message = try await socket.receive()
            let data: Data
            switch message {
            case .string(let text):
                data = Data(text.utf8)
            case .data(let payload):
                data = payload
            @unknown default:
                throw SmritiAPIError.badResponse
            }

            let event = try decoder.decode(EventMessage.self, from: data)
            if event.type == "observation.created",
               let observation = ObservationSummary.fromEventPayload(event.payload)
            {
                await sink?.onObservation(observation)
            }
        }
    }

    private func heartbeatLoop() async {
        guard let socket else { return }
        while shouldRun && !isPaused && !Task.isCancelled {
            try? await Task.sleep(for: .seconds(25))
            guard shouldRun, !isPaused, !Task.isCancelled else { return }
            let pongReceived = await ping(socket)
            if !pongReceived {
                socket.cancel(with: .goingAway, reason: nil)
                return
            }
        }
    }

    private func ping(_ socket: URLSessionWebSocketTask) async -> Bool {
        await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                await withCheckedContinuation { continuation in
                    socket.sendPing { error in
                        continuation.resume(returning: error == nil)
                    }
                }
            }
            group.addTask {
                try? await Task.sleep(for: .seconds(5))
                return false
            }

            let result = await group.next() ?? false
            group.cancelAll()
            return result
        }
    }

    private func closeSocket() {
        socket?.cancel(with: .goingAway, reason: nil)
        socket = nil
    }
}

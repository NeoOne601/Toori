import SwiftUI
import Speech
import AVFoundation
import UIKit

struct VoiceRecallButton: View {
    @EnvironmentObject private var appModel: SmritiAppModel
    @StateObject private var controller = VoiceRecallController()

    var body: some View {
        VStack(alignment: .trailing, spacing: 12) {
            if !controller.transcript.isEmpty {
                Text(controller.transcript)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(Color.black.opacity(0.58))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .stroke(Color.smritiStroke, lineWidth: 0.5)
                    )
                    .frame(maxWidth: 220, alignment: .trailing)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            }

            Button {
                controller.toggle {
                    await appModel.runRecall(query: $0)
                }
            } label: {
                ZStack {
                    Circle()
                        .fill(Color.smritiAccent)
                        .frame(width: 56, height: 56)
                        .shadow(color: Color.smritiAccent.opacity(0.35), radius: 24, y: 8)

                    if controller.isListening {
                        LiveWaveBars(level: controller.audioLevel)
                    } else {
                        Image(systemName: "mic.fill")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundStyle(.white)
                    }
                }
            }
            .buttonStyle(.plain)
        }
        .animation(.smritiSpring, value: controller.transcript)
    }
}

private struct LiveWaveBars: View {
    let level: Double

    var body: some View {
        TimelineView(.animation) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            HStack(spacing: 3) {
                ForEach(0..<5, id: \.self) { index in
                    let phase = time * 2.6 + Double(index) * 0.35
                    let base = (sin(phase * .pi * 2) + 1) / 2
                    let height = 10 + CGFloat(base) * (10 + CGFloat(level) * 24)
                    Capsule(style: .continuous)
                        .fill(Color.white)
                        .frame(width: 4, height: height)
                }
            }
            .frame(width: 30, height: 30)
        }
    }
}

@MainActor
final class VoiceRecallController: NSObject, ObservableObject {
    @Published private(set) var transcript = ""
    @Published private(set) var isListening = false
    @Published private(set) var audioLevel: Double = 0.08

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var audioEngine: AVAudioEngine?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var silenceTask: Task<Void, Never>?

    func toggle(onCommit: @escaping (String) async -> Void) {
        if isListening {
            stop(shouldCommit: true, onCommit: onCommit)
        } else {
            start(onCommit: onCommit)
        }
    }

    private func start(onCommit: @escaping (String) async -> Void) {
        guard !isListening else { return }

        SFSpeechRecognizer.requestAuthorization { [weak self] speechStatus in
            guard speechStatus == .authorized else { return }
            AVAudioApplication.requestRecordPermission { granted in
                guard granted, let self else { return }
                Task { @MainActor in
                    await self.begin(onCommit: onCommit)
                }
            }
        }
    }

    private func begin(onCommit: @escaping (String) async -> Void) async {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: [.duckOthers])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            return
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = false
        recognitionRequest = request

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        inputNode.removeTap(onBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
            guard let data = buffer.floatChannelData?[0] else { return }
            let samples = UnsafeBufferPointer(start: data, count: Int(buffer.frameLength))
            let squared = samples.reduce(0.0) { partial, value in
                partial + Double(value * value)
            }
            let rms = sqrt(squared / max(Double(samples.count), 1))
            Task { @MainActor in
                self?.audioLevel = max(0.08, min(rms * 12, 1.0))
            }
        }

        recognitionTask = recognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            if let result {
                Task { @MainActor in
                    self.transcript = result.bestTranscription.formattedString
                    self.scheduleSilenceCommit(onCommit: onCommit)
                }
            }
            if error != nil {
                Task { @MainActor in
                    self.stop(shouldCommit: false, onCommit: onCommit)
                }
            }
        }

        do {
            engine.prepare()
            try engine.start()
            audioEngine = engine
            isListening = true
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        } catch {
            stop(shouldCommit: false, onCommit: onCommit)
        }
    }

    private func scheduleSilenceCommit(onCommit: @escaping (String) async -> Void) {
        silenceTask?.cancel()
        silenceTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(1.2))
            guard let self else { return }
            await MainActor.run {
                self.stop(shouldCommit: true, onCommit: onCommit)
            }
        }
    }

    private func stop(shouldCommit: Bool, onCommit: @escaping (String) async -> Void) {
        guard isListening || !transcript.isEmpty else { return }

        let finalTranscript = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        silenceTask?.cancel()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest?.endAudio()
        recognitionRequest = nil

        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil

        isListening = false
        audioLevel = 0.08
        try? AVAudioSession.sharedInstance().setActive(false)

        if shouldCommit, !finalTranscript.isEmpty {
            Task {
                await onCommit(finalTranscript)
            }
        }
    }
}

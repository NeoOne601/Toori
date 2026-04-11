import SwiftUI
import AVFoundation

struct HumToFindButton: View {
    @EnvironmentObject private var appModel: SmritiAppModel
    @StateObject private var recorder = HumRecorder()

    var body: some View {
        Button {
            recorder.startRecording { audioBase64, sampleRate in
                Task {
                    await appModel.runHumToFind(audioBase64: audioBase64, sampleRate: sampleRate)
                }
            }
        } label: {
            HStack(spacing: 10) {
                Image(systemName: recorder.isRecording ? "waveform" : "music.note")
                    .font(.system(size: 14, weight: .semibold))
                Text(recorder.isRecording ? "Listening..." : "Hum to find")
                    .font(.system(size: 14, weight: .semibold))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.white.opacity(recorder.isRecording ? 0.14 : 0.08))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(Color.smritiStroke, lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .disabled(recorder.isRecording || appModel.isAudioLoading)
    }
}

@MainActor
final class HumRecorder: ObservableObject {
    @Published private(set) var isRecording = false

    private var engine: AVAudioEngine?
    private var sampleRate = 16_000
    private var samples: [Float] = []

    func startRecording(onComplete: @escaping (String, Int) -> Void) {
        guard !isRecording else { return }

        AVAudioApplication.requestRecordPermission { [weak self] granted in
            guard granted, let self else { return }
            Task { @MainActor in
                await self.begin(onComplete: onComplete)
            }
        }
    }

    private func begin(onComplete: @escaping (String, Int) -> Void) async {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: [.duckOthers])
            try session.setActive(true)
        } catch {
            return
        }

        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)

        samples.removeAll(keepingCapacity: true)
        sampleRate = max(8_000, min(Int(format.sampleRate.rounded()), 48_000))
        isRecording = true

        input.installTap(onBus: 0, bufferSize: 2048, format: format) { [weak self] buffer, _ in
            guard let self, let data = buffer.floatChannelData?[0] else { return }
            let frameCount = Int(buffer.frameLength)
            let frameSamples = Array(UnsafeBufferPointer(start: data, count: frameCount))
            Task { @MainActor in
                self.samples.append(contentsOf: frameSamples)
            }
        }

        do {
            engine.prepare()
            try engine.start()
            self.engine = engine

            Task { [weak self] in
                try? await Task.sleep(for: .seconds(3))
                self?.finish(onComplete: onComplete)
            }
        } catch {
            finishImmediately()
        }
    }

    private func finish(onComplete: @escaping (String, Int) -> Void) {
        guard isRecording else { return }

        let engine = engine
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        self.engine = nil
        isRecording = false

        let pcmData = samples.withUnsafeBufferPointer { buffer in
            Data(buffer: buffer)
        }
        let audioBase64 = pcmData.base64EncodedString()
        onComplete(audioBase64, sampleRate)

        try? AVAudioSession.sharedInstance().setActive(false)
    }

    private func finishImmediately() {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        isRecording = false
        samples.removeAll()
    }
}

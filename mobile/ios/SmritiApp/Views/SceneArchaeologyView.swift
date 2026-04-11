import SwiftUI
import AVFoundation
import SmritiKit

struct SceneArchaeologyView: View {
    @Environment(\.dismiss) var dismiss
    @EnvironmentObject var appModel: SmritiAppModel
    
    @StateObject private var cameraModel = CameraModel()
    
    @State private var isAnalyzing = false
    @State private var matchResults: [ObservationSummary] = []
    @State private var narration: String?
    
    var body: some View {
        ZStack {
            CameraPreviewView(session: cameraModel.session)
                .ignoresSafeArea()
            
            VStack {
                HStack {
                    Button(action: { dismiss() }) {
                        Image(systemName: "xmark")
                            .font(.title3.bold())
                            .foregroundColor(.white)
                            .padding(12)
                            .background(Color.black.opacity(0.6))
                            .clipShape(Circle())
                    }
                    .padding()
                    Spacer()
                }
                
                Spacer()
                
                if isAnalyzing {
                    ProgressView("Analyzing Scene...")
                        .tint(.white)
                        .foregroundColor(.white)
                        .padding()
                        .background(.thinMaterial)
                        .cornerRadius(12)
                        .padding(.bottom, 20)
                } else if !matchResults.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        if let narration {
                            Text(narration)
                                .font(.subheadline)
                                .foregroundColor(.white)
                                .padding()
                                .background(Color.black.opacity(0.7))
                                .cornerRadius(12)
                        }
                        
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 8) {
                                ForEach(matchResults, id: \.id) { hit in
                                    Text(hit.summary ?? "Unknown Scene")
                                        .font(.caption)
                                        .padding()
                                        .background(Color.smritiAccent.opacity(0.8))
                                        .cornerRadius(8)
                                        .foregroundColor(.white)
                                }
                            }
                        }
                    }
                    .padding()
                    .padding(.bottom, 10)
                }
                
                Button {
                    captureAndAnalyze()
                } label: {
                    Circle()
                        .strokeBorder(.white.opacity(0.5), lineWidth: 4)
                        .background(Circle().fill(Color.smritiAccent))
                        .frame(width: 72, height: 72)
                        .overlay(
                            Image(systemName: "camera.viewfinder")
                                .font(.title)
                                .foregroundColor(.white)
                        )
                }
                .padding(.bottom, 40)
            }
        }
        .onAppear { cameraModel.start() }
        .onDisappear { cameraModel.stop() }
    }
    
    private func captureAndAnalyze() {
        isAnalyzing = true
        matchResults = []
        narration = nil
        
        cameraModel.captureImage { image in
            Task {
                defer { isAnalyzing = false }
                guard let image = image,
                      let jpeg = image.jpegData(compressionQuality: 0.6) else { return }
                
                let b64 = jpeg.base64EncodedString()
                let req = AnalyzeRequest(image_base64: b64, session_id: "archaeology", query: "scenes visually similar to this")
                
                do {
                    let api = try SmritiAPI(host: appModel.backendHost)
                    let response = try await api.analyze(req)
                    
                    let hits = response.hits
                    self.matchResults = hits
                    
                    let available = await GemmaModelManager.shared.isAvailable()
                    if let firstHit = hits.first, let summary = firstHit.summary, available {
                        // The constraint stated using hit.summary
                        let prompt = "Describe the difference between the current frame and this retrieved scene: '\(summary)'."
                        self.narration = try? await GemmaModelManager.shared.generate(prompt: prompt, maxTokens: 100)
                    }
                } catch {
                    print("Analyze failed: \(error)")
                }
            }
        }
    }
}

class CameraModel: NSObject, ObservableObject, AVCapturePhotoCaptureDelegate {
    let session = AVCaptureSession()
    private let output = AVCapturePhotoOutput()
    private var completion: ((UIImage?) -> Void)?
    
    override init() {
        super.init()
        session.beginConfiguration()
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input),
              session.canAddOutput(output) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)
        session.addOutput(output)
        session.commitConfiguration()
    }
    
    func start() { Task(priority: .background) { session.startRunning() } }
    func stop() { session.stopRunning() }
    
    func captureImage(completion: @escaping (UIImage?) -> Void) {
        self.completion = completion
        output.capturePhoto(with: AVCapturePhotoSettings(), delegate: self)
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let data = photo.fileDataRepresentation(), let image = UIImage(data: data) {
            completion?(image)
        } else {
            completion?(nil)
        }
    }
}

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> VideoPreviewView {
        let view = VideoPreviewView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        return view
    }
    func updateUIView(_ uiView: VideoPreviewView, context: Context) {}
}

class VideoPreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

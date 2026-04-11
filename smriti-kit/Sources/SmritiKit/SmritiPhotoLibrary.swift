@preconcurrency import AVFoundation
import Combine
import Foundation
import Photos
import UniformTypeIdentifiers

public final class SmritiPhotoLibrary: NSObject, ObservableObject, PHPhotoLibraryChangeObserver {
    @Published public private(set) var authorizationStatus: PHAuthorizationStatus
    @Published public private(set) var newAssetCount: Int = 0

    private let host: String
    private var trackedFetchResult: PHFetchResult<PHAsset>?
    private var isRegistered = false

    public init(host: String = "127.0.0.1:7777") {
        self.host = host
        self.authorizationStatus = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        super.init()
    }

    deinit {
        if isRegistered {
            PHPhotoLibrary.shared().unregisterChangeObserver(self)
        }
    }

    @MainActor
    public func requestAccess() async {
        let existingStatus = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        authorizationStatus = existingStatus

        if existingStatus == .authorized || existingStatus == .limited {
            registerForChangesIfNeeded()
            trackedFetchResult = Self.makeTrackedFetchResult()
            await exportRecentAssets(limit: 200)
            return
        }

        let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        authorizationStatus = status

        guard status == .authorized || status == .limited else {
            return
        }

        registerForChangesIfNeeded()
        trackedFetchResult = Self.makeTrackedFetchResult()
        await exportRecentAssets(limit: 200)
    }

    public func photoLibraryDidChange(_ changeInstance: PHChange) {
        Task {
            await exportNewAssets(from: changeInstance)
        }
    }

    private func registerForChangesIfNeeded() {
        guard !isRegistered else { return }
        PHPhotoLibrary.shared().register(self)
        isRegistered = true
    }

    private func exportRecentAssets(limit: Int) async {
        let options = Self.makeFetchOptions(limit: limit)
        let fetchResult = PHAsset.fetchAssets(with: options)
        trackedFetchResult = Self.makeTrackedFetchResult()
        await exportAssets(fetchResult.objects(at: IndexSet(integersIn: 0..<fetchResult.count)))
    }

    private func exportNewAssets(from change: PHChange) async {
        guard let trackedFetchResult else { return }
        guard let details = change.changeDetails(for: trackedFetchResult) else { return }
        self.trackedFetchResult = details.fetchResultAfterChanges
        let inserted = details.insertedObjects.filter {
            $0.mediaType == .image || $0.mediaType == .video
        }
        guard !inserted.isEmpty else { return }
        await exportAssets(inserted)
    }

    private func exportAssets(_ assets: [PHAsset]) async {
        for asset in assets {
            guard let tempURL = try? await exportTempFile(for: asset) else { continue }
            do {
                let api = try SmritiAPI(host: host)
                _ = try await api.ingestFile(path: tempURL.path)
                try? FileManager.default.removeItem(at: tempURL)
                await MainActor.run {
                    self.newAssetCount += 1
                }
            } catch {
                try? FileManager.default.removeItem(at: tempURL)
            }
        }
    }

    private func exportTempFile(for asset: PHAsset) async throws -> URL {
        switch asset.mediaType {
        case .image:
            return try await exportImageAsset(asset)
        case .video:
            return try await exportVideoAsset(asset)
        default:
            throw PhotoLibraryExportError.unsupportedMediaType
        }
    }

    private func exportImageAsset(_ asset: PHAsset) async throws -> URL {
        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.isNetworkAccessAllowed = false
        options.isSynchronous = false
        options.version = .current

        let result: (Data, String?) = try await withCheckedThrowingContinuation { continuation in
            PHImageManager.default().requestImageDataAndOrientation(for: asset, options: options) { data, dataUTI, _, info in
                if let cancelled = info?[PHImageCancelledKey] as? Bool, cancelled {
                    continuation.resume(throwing: PhotoLibraryExportError.cancelled)
                    return
                }
                if let degraded = info?[PHImageResultIsDegradedKey] as? Bool, degraded {
                    return
                }
                if let error = info?[PHImageErrorKey] as? Error {
                    continuation.resume(throwing: error)
                    return
                }
                guard let data else {
                    continuation.resume(throwing: PhotoLibraryExportError.missingData)
                    return
                }
                continuation.resume(returning: (data, dataUTI))
            }
        }

        let ext = Self.preferredExtension(for: result.1) ?? "jpg"
        let tempURL = makeTempURL(for: asset, ext: ext)
        try result.0.write(to: tempURL, options: Data.WritingOptions.atomic)
        try markExcludedFromBackupIfNeeded(tempURL)
        return tempURL
    }

    private func exportVideoAsset(_ asset: PHAsset) async throws -> URL {
        let options = PHVideoRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.isNetworkAccessAllowed = false
        options.version = .current

        let avAsset: AVAsset = try await withCheckedThrowingContinuation { continuation in
            PHImageManager.default().requestAVAsset(forVideo: asset, options: options) { asset, _, info in
                if let cancelled = info?[PHImageCancelledKey] as? Bool, cancelled {
                    continuation.resume(throwing: PhotoLibraryExportError.cancelled)
                    return
                }
                if let error = info?[PHImageErrorKey] as? Error {
                    continuation.resume(throwing: error)
                    return
                }
                guard let asset else {
                    continuation.resume(throwing: PhotoLibraryExportError.missingData)
                    return
                }
                continuation.resume(returning: asset)
            }
        }

        guard let exportSession = AVAssetExportSession(asset: avAsset, presetName: AVAssetExportPresetHighestQuality) else {
            throw PhotoLibraryExportError.exportSessionUnavailable
        }
        let sessionBox = ExportSessionBox(exportSession)

        let tempURL = makeTempURL(for: asset, ext: "mp4")
        if FileManager.default.fileExists(atPath: tempURL.path) {
            try? FileManager.default.removeItem(at: tempURL)
        }
        sessionBox.session.outputURL = tempURL
        sessionBox.session.outputFileType = AVFileType.mp4
        sessionBox.session.shouldOptimizeForNetworkUse = false

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            sessionBox.session.exportAsynchronously {
                switch sessionBox.session.status {
                case .completed:
                    continuation.resume(returning: ())
                case .failed:
                    continuation.resume(throwing: sessionBox.session.error ?? PhotoLibraryExportError.exportFailed)
                case .cancelled:
                    continuation.resume(throwing: PhotoLibraryExportError.cancelled)
                default:
                    continuation.resume(throwing: PhotoLibraryExportError.exportFailed)
                }
            }
        }

        try markExcludedFromBackupIfNeeded(tempURL)
        return tempURL
    }

    private func markExcludedFromBackupIfNeeded(_ url: URL) throws {
        #if canImport(UIKit)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutableURL = url
        try mutableURL.setResourceValues(values)
        #endif
    }

    private func makeTempURL(for asset: PHAsset, ext: String) -> URL {
        let sanitizedIdentifier = asset.localIdentifier
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: ":", with: "_")
        return FileManager.default.temporaryDirectory
            .appendingPathComponent(sanitizedIdentifier)
            .appendingPathExtension(ext)
    }

    private static func preferredExtension(for uti: String?) -> String? {
        guard let uti, !uti.isEmpty else { return nil }
        return UTType(importedAs: uti).preferredFilenameExtension
    }

    private static func makeTrackedFetchResult() -> PHFetchResult<PHAsset> {
        PHAsset.fetchAssets(with: makeFetchOptions(limit: 0))
    }

    private static func makeFetchOptions(limit: Int) -> PHFetchOptions {
        let options = PHFetchOptions()
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        options.predicate = NSPredicate(
            format: "mediaType == %d OR mediaType == %d",
            PHAssetMediaType.image.rawValue,
            PHAssetMediaType.video.rawValue
        )
        if limit > 0 {
            options.fetchLimit = limit
        }
        return options
    }
}

private enum PhotoLibraryExportError: Error {
    case cancelled
    case missingData
    case unsupportedMediaType
    case exportSessionUnavailable
    case exportFailed
}

private final class ExportSessionBox: @unchecked Sendable {
    let session: AVAssetExportSession

    init(_ session: AVAssetExportSession) {
        self.session = session
    }
}

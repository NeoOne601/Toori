import Foundation
import SwiftUI
import ImageIO
import CoreGraphics

#if canImport(UIKit)
import UIKit

public typealias PlatformImage = UIImage

public extension Image {
    init(smritiPlatformImage image: PlatformImage) {
        self.init(uiImage: image)
    }
}
#elseif canImport(AppKit)
import AppKit

public typealias PlatformImage = NSImage

public extension Image {
    init(smritiPlatformImage image: PlatformImage) {
        self.init(nsImage: image)
    }
}
#endif

public enum SmritiImageCacheError: Error {
    case badResponse
    case decodeFailed
}

public actor SmritiImageCache {
    public static let shared = SmritiImageCache()

    private let cache = NSCache<NSString, PlatformImage>()
    private var inFlight: [URL: Task<Data, Error>] = [:]

    public init() {
        cache.countLimit = 200
        cache.totalCostLimit = 52_428_800
    }

    public func image(for url: URL) async throws -> PlatformImage {
        let key = url.absoluteString as NSString
        if let cached = cache.object(forKey: key) {
            return cached
        }
        if let existing = inFlight[url] {
            let data = try await existing.value
            let image = try Self.decodeImage(from: data)
            cache.setObject(image, forKey: key, cost: Self.cost(for: image))
            return image
        }

        let task = Task.detached(priority: .utility) {
            if url.isFileURL {
                return try Data(contentsOf: url)
            } else {
                let (remoteData, response) = try await URLSession.shared.data(from: url)
                guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                    throw SmritiImageCacheError.badResponse
                }
                _ = http
                return remoteData
            }
        }

        inFlight[url] = task
        defer { inFlight[url] = nil }

        let data = try await task.value
        let image = try Self.decodeImage(from: data)
        cache.setObject(image, forKey: key, cost: Self.cost(for: image))
        return image
    }

    public func clear() {
        cache.removeAllObjects()
        inFlight.values.forEach { $0.cancel() }
        inFlight.removeAll()
    }

    private static func decodeImage(from data: Data) throws -> PlatformImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            throw SmritiImageCacheError.decodeFailed
        }

        let options: [CFString: Any] = [
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceCreateThumbnailFromImageIfAbsent: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
        ]

        if let cgImage = CGImageSourceCreateImageAtIndex(source, 0, options as CFDictionary) {
            return makePlatformImage(from: cgImage)
        }

        if let dataImage = Self.imageFromDataFallback(data: data) {
            return dataImage
        }

        throw SmritiImageCacheError.decodeFailed
    }

    private static func imageFromDataFallback(data: Data) -> PlatformImage? {
        #if canImport(UIKit)
        return UIImage(data: data)
        #elseif canImport(AppKit)
        return NSImage(data: data)
        #else
        return nil
        #endif
    }

    private static func makePlatformImage(from cgImage: CGImage) -> PlatformImage {
        #if canImport(UIKit)
        return UIImage(cgImage: cgImage)
        #elseif canImport(AppKit)
        return NSImage(cgImage: cgImage, size: .zero)
        #endif
    }

    private static func cost(for image: PlatformImage) -> Int {
        #if canImport(UIKit)
        guard let cgImage = image.cgImage else {
            return 1
        }
        return max(cgImage.bytesPerRow * cgImage.height, 1)
        #elseif canImport(AppKit)
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return 1
        }
        return max(cgImage.bytesPerRow * cgImage.height, 1)
        #endif
    }
}

public struct SmritiAsyncImage<Placeholder: View>: View {
    private let url: URL?
    private let cache: SmritiImageCache
    private let contentMode: ContentMode
    private let placeholder: () -> Placeholder

    @State private var loadedImage: PlatformImage?
    @State private var isLoading = false

    public init(
        url: URL?,
        cache: SmritiImageCache = .shared,
        contentMode: ContentMode = .fill,
        @ViewBuilder placeholder: @escaping () -> Placeholder
    ) {
        self.url = url
        self.cache = cache
        self.contentMode = contentMode
        self.placeholder = placeholder
    }

    public var body: some View {
        ZStack {
            if let loadedImage {
                Image(smritiPlatformImage: loadedImage)
                    .resizable()
                    .aspectRatio(contentMode: contentMode)
            } else {
                placeholder()
                    .overlay {
                        if isLoading {
                            SmritiShimmerOverlay()
                                .allowsHitTesting(false)
                        }
                    }
            }
        }
        .task(id: url) {
            await load()
        }
    }

    private func load() async {
        guard let url else {
            loadedImage = nil
            isLoading = false
            return
        }

        isLoading = true
        defer { isLoading = false }

        do {
            let image = try await cache.image(for: url)
            loadedImage = image
        } catch {
            loadedImage = nil
        }
    }
}

private struct SmritiShimmerOverlay: View {
    private let cycle: TimeInterval = 1.2

    var body: some View {
        GeometryReader { proxy in
            TimelineView(.animation) { timeline in
                let time = timeline.date.timeIntervalSinceReferenceDate
                let progress = time.truncatingRemainder(dividingBy: cycle) / cycle
                let travel = proxy.size.width * 1.8
                let offset = -proxy.size.width * 0.9 + CGFloat(progress) * travel

                LinearGradient(
                    colors: [
                        Color.smritiShimmerBase,
                        Color.smritiShimmerHighlight,
                        Color.smritiShimmerBase,
                    ],
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .frame(width: proxy.size.width * 1.4, height: proxy.size.height * 1.2)
                .rotationEffect(.degrees(14))
                .offset(x: offset)
                .blendMode(.screen)
            }
        }
        .clipped()
    }
}

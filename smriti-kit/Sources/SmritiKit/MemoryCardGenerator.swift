import Foundation
import CoreGraphics
#if os(macOS)
import AppKit
fileprivate typealias PlatformColor = NSColor
#else
import UIKit
fileprivate typealias PlatformColor = UIColor
#endif

public class MemoryCardGenerator {
    
    public init() {}
    
    public func generateMemoryCard(imagePath: String?, date: Date, summary: String?) -> URL? {
        var heroImage: PlatformImage? = nil
        if let path = imagePath {
            heroImage = PlatformImage(contentsOfFile: path)
        }
        
        let targetSize = CGSize(width: 1080, height: 1080)
        let rect = CGRect(origin: .zero, size: targetSize)
        
        let tempDir = FileManager.default.temporaryDirectory
        let outURL = tempDir.appendingPathComponent(UUID().uuidString + ".png")
        
        #if os(macOS)
        let finalImage = NSImage(size: targetSize)
        finalImage.lockFocus()
        
        PlatformColor(white: 0.1, alpha: 1.0).set()
        rect.fill()
        
        let imageRect = CGRect(x: 40, y: 160, width: 1000, height: 880)
        if let img = heroImage {
            img.draw(in: imageRect, from: NSRect(origin: .zero, size: img.size), operation: NSCompositingOperation.sourceOver, fraction: 1.0)
        }
        
        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.alignment = .center
        
        let dateString = date.formatted(date: .abbreviated, time: .omitted)
        let text = "\(summary ?? "Memory")\n\(dateString)"
        
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 32, weight: .semibold),
            .foregroundColor: NSColor.white,
            .paragraphStyle: paragraphStyle
        ]
        
        let textRect = CGRect(x: 40, y: 40, width: 1000, height: 100)
        text.draw(in: textRect, withAttributes: attributes)
        
        finalImage.unlockFocus()
        
        if let tiffData = finalImage.tiffRepresentation,
           let bitmap = NSBitmapImageRep(data: tiffData),
           let png = bitmap.representation(using: .png, properties: [:]) {
            try? png.write(to: outURL)
            return outURL
        }
        #else
        UIGraphicsBeginImageContextWithOptions(targetSize, true, 2.0)
        guard let context = UIGraphicsGetCurrentContext() else { return nil }
        
        PlatformColor(white: 0.1, alpha: 1.0).setFill()
        context.fill(rect)
        
        let imageRect = CGRect(x: 40, y: 40, width: 1000, height: 880)
        if let img = heroImage {
            img.draw(in: imageRect)
        }
        
        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.alignment = .center
        
        let dateString = date.formatted(date: .abbreviated, time: .omitted)
        let text = "\(summary ?? "Memory")\n\(dateString)"
        
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 32, weight: .semibold),
            .foregroundColor: UIColor.white,
            .paragraphStyle: paragraphStyle
        ]
        
        let textRect = CGRect(x: 40, y: 940, width: 1000, height: 100)
        text.draw(in: textRect, withAttributes: attributes)
        
        let finalImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        if let png = finalImage?.pngData() {
            try? png.write(to: outURL)
            return outURL
        }
        #endif
        
        return nil
    }
}

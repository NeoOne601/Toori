import UIKit

extension UIImage {
    /// Resizes the image to fit within the specified dimension while maintaining aspect ratio.
    /// Also handles compression to JPEG Data.
    func toOptimizedData(maxDimension: CGFloat = 1024, compressionQuality: CGFloat = 0.8) -> Data? {
        let size = self.size
        
        let widthRatio  = maxDimension / size.width
        let heightRatio = maxDimension / size.height
        
        // Only downscale if the image is actually larger than the max dimension
        guard widthRatio < 1.0 || heightRatio < 1.0 else {
            return self.jpegData(compressionQuality: compressionQuality)
        }
        
        let newSize: CGSize
        if widthRatio < heightRatio {
            newSize = CGSize(width: maxDimension, height: size.height * widthRatio)
        } else {
            newSize = CGSize(width: size.width * heightRatio, height: maxDimension)
        }
        
        let rect = CGRect(origin: .zero, size: newSize)
        
        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        self.draw(in: rect)
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return newImage?.jpegData(compressionQuality: compressionQuality)
    }
}

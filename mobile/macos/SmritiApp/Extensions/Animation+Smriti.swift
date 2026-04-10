import SwiftUI

extension Animation {
    static let smritiSpring = Animation.spring(response: 0.38, dampingFraction: 0.72)

    static func smritiStagger(index: Int) -> Animation {
        .smritiSpring.delay(Double(index) * 0.04)
    }
}

import CoreGraphics
import Foundation

struct DetectedObject: Identifiable, Equatable {
    let id = UUID()
    let label: String
    let localizedLabel: String
    let confidence: Float
    let boundingBox: CGRect
    let imageSize: CGSize
    let timestamp: Date
}
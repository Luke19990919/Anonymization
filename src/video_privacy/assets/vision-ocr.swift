import Foundation
import Vision
import CoreGraphics

func readExact(_ count: Int) -> Data? {
    var bytes = Data()
    while bytes.count < count {
        guard let chunk = try? FileHandle.standardInput.read(upToCount: count - bytes.count), !chunk.isEmpty else { return nil }
        bytes.append(chunk)
    }
    return bytes
}

struct OCRBox: Encodable {
    let rect: [Int]
    let text: String
    let confidence: Float
}

func run() throws {
    while let header = readExact(8) {
        let width = header.prefix(4).enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
        let height = header.suffix(4).enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
        guard width > 0, height > 0, width <= 16384, height <= 16384,
              let pixels = readExact(width * height * 3),
              let provider = CGDataProvider(data: pixels as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8,
                                  bitsPerPixel: 24, bytesPerRow: width * 3,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { throw NSError(domain: "InvalidFrame", code: 1) }
        try autoreleasepool {
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.recognitionLanguages = ["zh-Hant", "en-US"]
            request.usesLanguageCorrection = false
            request.minimumTextHeight = 0
            try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
            let boxes = (request.results ?? []).compactMap { observation -> OCRBox? in
                guard let candidate = observation.topCandidates(1).first else { return nil }
                let box = observation.boundingBox
                return OCRBox(rect: [max(0,Int(floor(box.minX * Double(width)))),
                                     max(0,Int(floor((1 - box.maxY) * Double(height)))),
                                     min(width,Int(ceil(box.maxX * Double(width)))),
                                     min(height,Int(ceil((1 - box.minY) * Double(height))))],
                              text: candidate.string, confidence: candidate.confidence)
            }
            var json = try JSONEncoder().encode(boxes)
            json.append(10)
            FileHandle.standardOutput.write(json)
        }
    }
}

do { try run() } catch { exit(1) }

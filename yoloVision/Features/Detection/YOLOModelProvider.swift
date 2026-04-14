import CoreML
import Foundation

enum YOLOModelProviderError: LocalizedError {
    case modelNotFound

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "YOLO 모델 파일을 찾지 못했습니다. 앱 번들에 yolov8n/yolov8s/yolo11s 계열 .mlpackage 또는 .mlmodelc를 추가해 주세요."
        }
    }
}

final class YOLOModelProvider {
    static let prioritizedModelNames: [String] = [
        "yolo11x",
        "yolo11l",
        "yolo11m",
        "yolo11s",
        "yolo11n",
        "yolov8x",
        "yolov8l",
        "yolov8m",
        "yolov8s",
        "yolov8n",
        "yolov8"
    ]

    private static let supportedExtensions: [String] = ["mlmodelc", "mlpackage"]
    private static let candidateSubdirectories: [String?] = [nil, "Resources/ML", "ML"]

    private(set) var model: MLModel?
    private(set) var loadedModelName: String?

    func availableModelNames() -> [String] {
        var found: [String] = []
        for name in Self.prioritizedModelNames {
            if findModelURL(named: name) != nil {
                    found.append(name)
            }
        }
        return found
    }

    func clearLoadedModel() {
        model = nil
        loadedModelName = nil
    }

    func loadIfNeeded(preferredModelName: String? = nil) throws -> MLModel {
        if let preferredModelName,
           let model,
           loadedModelName == preferredModelName {
            return model
        }

        if preferredModelName == nil, let model {
            return model
        }

        if let model {
            if preferredModelName == nil {
                return model
            }
            clearLoadedModel()
        }

        let config = MLModelConfiguration()
        config.computeUnits = .all

        let candidateNames: [String]
        if let preferredModelName {
            candidateNames = [preferredModelName]
        } else {
            candidateNames = Self.prioritizedModelNames
        }

        var candidates: [(String, String)] = []
        for name in candidateNames {
            for ext in Self.supportedExtensions {
                candidates.append((name, ext))
            }
        }

        for (name, ext) in candidates {
            if let url = findModelURL(named: name, preferredExtension: ext) {
                let loadedModel = try MLModel(contentsOf: url, configuration: config)
                model = loadedModel
                loadedModelName = name
                return loadedModel
            }
        }

        throw YOLOModelProviderError.modelNotFound
    }

    private func findModelURL(named modelName: String, preferredExtension: String? = nil) -> URL? {
        let extensionsToSearch: [String]
        if let preferredExtension {
            extensionsToSearch = [preferredExtension]
        } else {
            extensionsToSearch = Self.supportedExtensions
        }

        for subdirectory in Self.candidateSubdirectories {
            for ext in extensionsToSearch {
                if let url = Bundle.main.url(forResource: modelName, withExtension: ext, subdirectory: subdirectory) {
                    return url
                }
            }
        }

        return nil
    }
}
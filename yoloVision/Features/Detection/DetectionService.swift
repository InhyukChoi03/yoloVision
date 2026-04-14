import Combine
import CoreML
import CoreVideo
import Foundation
import Vision

final class DetectionService: ObservableObject {
    static let autoModelOption = "자동(우선순위)"

    @Published private(set) var isModelReady = false
    @Published private(set) var topDetectedLabels: [String] = []
    @Published private(set) var latestDetections: [DetectedObject] = []
    @Published private(set) var statusMessage: String?
    @Published private(set) var activeModelName: String?
    @Published private(set) var availableModelOptions: [String] = [autoModelOption]
    @Published var selectedModelOption: String = autoModelOption
    @Published private(set) var currentInferenceFPS: Double = 0
    @Published private(set) var currentInferenceLatencyMs: Double = 0
    @Published private(set) var recentAverageFPS: Double = 0
    @Published private(set) var recentAverageLatencyMs: Double = 0
    @Published private(set) var inferenceSampleCount: Int = 0
    @Published private(set) var livePerformanceLog: String = "추론 속도 측정 대기 중"

    private let modelProvider = YOLOModelProvider()
    private let processingQueue = DispatchQueue(label: "yoloVision.detection.processing", qos: .userInitiated)

    private var request: VNCoreMLRequest?
    private var isProcessing = false
    private var lastInferenceTime = Date.distantPast
    private let minInferenceInterval: TimeInterval = 0.12
    private let confidenceThreshold: Float = 0.2
    private let genericConfidenceThreshold: Float = 0.4
    private var unsupportedOutputTypeNotified = false
    private var recentInferenceDurations: [TimeInterval] = []
    private let recentInferenceWindowSize = 30
    private var totalInferenceDuration: TimeInterval = 0
    private var totalInferenceSamples: Int = 0
    private var lastConsoleLogAt = Date.distantPast

    private let preferredLabels: Set<String> = [
        "person", "door", "stairs", "stair", "staircase", "toilet", "wall", "chair", "clock"
    ]

    private let koreanLabelMap: [String: String] = [
        "person": "사람",
        "door": "문",
        "stairs": "계단",
        "stair": "계단",
        "staircase": "계단",
        "toilet": "변기",
        "wall": "벽",
        "chair": "의자",
        "clock": "시계",
        "bench": "벤치",
        "couch": "소파",
        "tv": "TV",
        "dining table": "테이블"
    ]

    init() {
        resetPerformanceMetrics()
        refreshModelOptions()
    }

    func refreshModelOptions() {
        let models = modelProvider.availableModelNames()
        let options = [Self.autoModelOption] + models
        availableModelOptions = options

        if !options.contains(selectedModelOption) {
            selectedModelOption = Self.autoModelOption
        }
    }

    func applySelectedModel() async {
        request = nil
        isModelReady = false
        latestDetections = []
        topDetectedLabels = []
        resetPerformanceMetrics()
        unsupportedOutputTypeNotified = false
        modelProvider.clearLoadedModel()
        await startIfNeeded(forceReload: true)
    }

    func startIfNeeded(forceReload: Bool = false) async {
        if !forceReload, request != nil {
            return
        }

        do {
            let preferredModelName = selectedModelOption == Self.autoModelOption
                ? nil
                : selectedModelOption
            let model = try modelProvider.loadIfNeeded(preferredModelName: preferredModelName)
            let visionModel = try VNCoreMLModel(for: model)

            let request = VNCoreMLRequest(model: visionModel)
            request.imageCropAndScaleOption = .scaleFill

            await MainActor.run {
                self.request = request
                self.isModelReady = true
                self.activeModelName = self.modelProvider.loadedModelName
                self.statusMessage = nil
            }
        } catch {
            await MainActor.run {
                self.isModelReady = false
                self.activeModelName = nil
                self.statusMessage = "AI 모델 준비 실패: \(error.localizedDescription)"
            }
        }
    }

    func handleFrame(_ pixelBuffer: CVPixelBuffer) {
        guard let request else { return }

        processingQueue.async {
            let now = Date()
            guard now.timeIntervalSince(self.lastInferenceTime) >= self.minInferenceInterval else { return }
            guard !self.isProcessing else { return }

            self.isProcessing = true
            self.lastInferenceTime = now
            let inferenceStartedAt = DispatchTime.now().uptimeNanoseconds

            defer {
                self.isProcessing = false
            }

            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
            do {
                try handler.perform([request])
                let imageSize = CGSize(
                    width: CGFloat(CVPixelBufferGetHeight(pixelBuffer)),
                    height: CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                )

                guard let rawResults = request.results else {
                    self.recordPerformanceSample(startedAt: inferenceStartedAt, now: now)
                    Task { @MainActor in
                        self.latestDetections = []
                        self.topDetectedLabels = []
                    }
                    return
                }

                if !rawResults.isEmpty,
                   !(rawResults.first is VNRecognizedObjectObservation),
                   !self.unsupportedOutputTypeNotified {
                    self.unsupportedOutputTypeNotified = true
                    let typeName = String(describing: type(of: rawResults[0]))
                    Task { @MainActor in
                        self.statusMessage = "현재 모델 출력 형식(\(typeName))은 Vision 객체 탐지 결과와 달라서 박스/라벨이 비어 있습니다. YOLOv8 CoreML을 nms=True로 다시 export해 주세요."
                    }
                }

                let observations = rawResults.compactMap { $0 as? VNRecognizedObjectObservation }
                let mapped = observations.compactMap { obs -> DetectedObject? in
                    guard let best = obs.labels.first else { return nil }
                    let normalized = best.identifier.lowercased()
                    guard best.confidence >= self.confidenceThreshold else { return nil }

                    let isPreferred = self.preferredLabels.contains(normalized)
                    if !isPreferred && best.confidence < self.genericConfidenceThreshold {
                        return nil
                    }

                    let localizedLabel = self.koreanLabelMap[normalized] ?? best.identifier

                    return DetectedObject(
                        label: best.identifier,
                        localizedLabel: localizedLabel,
                        confidence: best.confidence,
                        boundingBox: obs.boundingBox,
                        imageSize: imageSize,
                        timestamp: now
                    )
                }

                let topLabels = mapped
                    .sorted(by: { $0.confidence > $1.confidence })
                    .prefix(3)
                    .map { String(format: "%@ %.0f%%", $0.localizedLabel, $0.confidence * 100) }

                self.recordPerformanceSample(startedAt: inferenceStartedAt, now: now)

                Task { @MainActor in
                    self.latestDetections = mapped
                    self.topDetectedLabels = Array(topLabels)
                    if !mapped.isEmpty {
                        self.statusMessage = nil
                    }
                }
            } catch {
                Task { @MainActor in
                    self.statusMessage = "프레임 추론 실패: \(error.localizedDescription)"
                }
            }
        }
    }

    private func resetPerformanceMetrics() {
        recentInferenceDurations = []
        totalInferenceDuration = 0
        totalInferenceSamples = 0
        lastConsoleLogAt = .distantPast

        Task { @MainActor in
            self.currentInferenceFPS = 0
            self.currentInferenceLatencyMs = 0
            self.recentAverageFPS = 0
            self.recentAverageLatencyMs = 0
            self.inferenceSampleCount = 0
            self.livePerformanceLog = "추론 속도 측정 대기 중"
        }
    }

    private func recordPerformanceSample(startedAt: UInt64, now: Date) {
        let finishedAt = DispatchTime.now().uptimeNanoseconds
        let duration = Double(finishedAt - startedAt) / 1_000_000_000
        guard duration > 0 else { return }

        recentInferenceDurations.append(duration)
        if recentInferenceDurations.count > recentInferenceWindowSize {
            recentInferenceDurations.removeFirst(recentInferenceDurations.count - recentInferenceWindowSize)
        }

        totalInferenceDuration += duration
        totalInferenceSamples += 1

        let currentFPS = 1.0 / duration
        let currentLatencyMs = duration * 1000

        let recentAvgDuration = recentInferenceDurations.reduce(0, +) / Double(recentInferenceDurations.count)
        let recentAvgFPS = recentAvgDuration > 0 ? 1.0 / recentAvgDuration : 0
        let recentAvgLatencyMs = recentAvgDuration * 1000

        let modelName = modelProvider.loadedModelName ?? activeModelName ?? "미정"
        let logLine = String(
            format: "모델 %@ | 현재 %.1f FPS (%.0fms) | 최근 평균 %.1f FPS (%.0fms) | 추론 %d회",
            modelName,
            currentFPS,
            currentLatencyMs,
            recentAvgFPS,
            recentAvgLatencyMs,
            totalInferenceSamples
        )

        if now.timeIntervalSince(lastConsoleLogAt) >= 1.0 {
            lastConsoleLogAt = now
            print("[DetectionPerf] \(logLine)")
        }

        Task { @MainActor in
            self.currentInferenceFPS = currentFPS
            self.currentInferenceLatencyMs = currentLatencyMs
            self.recentAverageFPS = recentAvgFPS
            self.recentAverageLatencyMs = recentAvgLatencyMs
            self.inferenceSampleCount = self.totalInferenceSamples
            self.livePerformanceLog = logLine
        }
    }
}
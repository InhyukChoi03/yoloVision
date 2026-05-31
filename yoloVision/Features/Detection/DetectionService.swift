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

    let speech = SpeechService()

    @Published var isVoiceEnabled: Bool = true {
        didSet {
            speech.isEnabled = isVoiceEnabled
            if !isVoiceEnabled {
                labelAnnouncementTimes.removeAll()
            }
        }
    }

    private var labelAnnouncementTimes: [String: Date] = [:]
    private let labelAnnouncementCooldown: TimeInterval = 3.0

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

    /// 보행 보조에 의미 있는 COCO 클래스만 통과시킨다(나머지는 시계/침대/노트북 등 무시).
    private let allowedLabels: Set<String> = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "train",
        "dog",
        "cat",
        "traffic light",
        "stop sign",
        "fire hydrant",
        "bench"
    ]

    /// 이동 장애물(우선 안내 + 더 낮은 신뢰도 임계값 적용).
    private let preferredLabels: Set<String> = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "train",
        "dog",
        "cat"
    ]

    private let koreanLabelMap: [String: String] = [
        "person": "사람",
        "bicycle": "자전거",
        "car": "자동차",
        "motorcycle": "오토바이",
        "bus": "버스",
        "truck": "트럭",
        "train": "기차",
        "dog": "개",
        "cat": "고양이",
        "traffic light": "신호등",
        "stop sign": "정지 표지판",
        "fire hydrant": "소화전",
        "bench": "벤치"
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
        labelAnnouncementTimes.removeAll()
        speech.stop()
        resetPerformanceMetrics()
        unsupportedOutputTypeNotified = false
        modelProvider.clearLoadedModel()
        await startIfNeeded(forceReload: true)
    }

    /// 탐지를 멈출 때(정지/메뉴 복귀) 음성과 안내 상태를 정리한다.
    func stopGuidance() {
        speech.stop()
        labelAnnouncementTimes.removeAll()
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
                    guard self.allowedLabels.contains(normalized) else { return nil }
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
                    self.announceIfNeeded(mapped)
                }
            } catch {
                Task { @MainActor in
                    self.statusMessage = "프레임 추론 실패: \(error.localizedDescription)"
                }
            }
        }
    }

    /// 보행에 중요한 객체 1개를 골라 방향과 함께 안내한다.
    /// 라벨별 쿨다운으로 같은 객체의 반복 알림을 완화한다.
    private func announceIfNeeded(_ detections: [DetectedObject]) {
        guard isVoiceEnabled, !detections.isEmpty else { return }

        let preferred = detections.filter { preferredLabels.contains($0.label.lowercased()) }
        let pool = preferred.isEmpty ? detections : preferred
        guard let candidate = pool.max(by: { $0.confidence < $1.confidence }) else { return }

        let now = Date()
        let key = candidate.localizedLabel
        if let last = labelAnnouncementTimes[key], now.timeIntervalSince(last) < labelAnnouncementCooldown {
            return
        }

        labelAnnouncementTimes[key] = now
        pruneAnnouncements(now: now)

        let phrase = "\(directionPhrase(for: candidate.boundingBox)) \(candidate.localizedLabel)"
        speech.enqueue(phrase)
    }

    private func directionPhrase(for boundingBox: CGRect) -> String {
        let midX = boundingBox.midX
        if midX < 0.4 {
            return "왼쪽에"
        } else if midX > 0.6 {
            return "오른쪽에"
        } else {
            return "정면에"
        }
    }

    private func pruneAnnouncements(now: Date) {
        let window = labelAnnouncementCooldown * 2
        labelAnnouncementTimes = labelAnnouncementTimes.filter { now.timeIntervalSince($0.value) < window }
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
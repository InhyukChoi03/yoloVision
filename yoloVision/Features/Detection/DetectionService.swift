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
    private let urgentAnnouncementCooldown: TimeInterval = 2.5

    /// 라벨별 마지막 "주의" 경고 시각. 지속적으로 가까운 동일 객체를 반복해서 끊지 않기 위해 사용.
    private var lastUrgentAnnouncementTimes: [String: Date] = [:]

    /// 이 거리(m) 이하이면 "주의" 경고로 안내한다.
    private let nearWarningDistance: Float = 1.5

    /// 음성 안내 전 최소 연속 관측 프레임 수(순간 오탐 억제). processingQueue에서만 갱신.
    private var trackObservationCounts: [String: (count: Int, lastSeen: Date)] = [:]
    private let minObservationsToAnnounce = 3

    /// 라벨+대략 위치별 거리 EMA 상태(프레임 간 거리 튐 완화). processingQueue에서만 접근.
    private var smoothedDistances: [String: (value: Float, timestamp: Date)] = [:]
    private let distanceSmoothingFactor: Float = 0.4
    private let distanceTrackTTL: TimeInterval = 1.0

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
        lastUrgentAnnouncementTimes.removeAll()
        processingQueue.async {
            self.smoothedDistances.removeAll()
            self.trackObservationCounts.removeAll()
        }
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
        lastUrgentAnnouncementTimes.removeAll()
        processingQueue.async {
            self.smoothedDistances.removeAll()
            self.trackObservationCounts.removeAll()
        }
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

    func handleFrame(_ pixelBuffer: CVPixelBuffer, depthData: CVPixelBuffer? = nil) {
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
                    let rawDistance = self.estimateDistance(for: obs.boundingBox, depthData: depthData)
                    let estimatedDistance = self.smoothedDistance(
                        rawDistance: rawDistance,
                        boundingBox: obs.boundingBox,
                        label: normalized,
                        now: now
                    )

                    return DetectedObject(
                        label: best.identifier,
                        localizedLabel: localizedLabel,
                        confidence: best.confidence,
                        boundingBox: obs.boundingBox,
                        imageSize: imageSize,
                        estimatedDistanceMeters: estimatedDistance,
                        timestamp: now
                    )
                }

                self.pruneSmoothedDistances(now: now)
                let stableKeys = self.updateTrackObservations(mapped, now: now)

                let topLabels = mapped
                    .sorted(by: { $0.confidence > $1.confidence })
                    .prefix(3)
                    .map { detection in
                        let base = String(format: "%@ %.0f%%", detection.localizedLabel, detection.confidence * 100)
                        guard let distance = detection.estimatedDistanceMeters else {
                            return base
                        }
                        return base + String(format: " %.1fm", distance)
                    }

                self.recordPerformanceSample(startedAt: inferenceStartedAt, now: now)

                Task { @MainActor in
                    self.latestDetections = mapped
                    self.topDetectedLabels = Array(topLabels)
                    if !mapped.isEmpty {
                        self.statusMessage = nil
                    }
                    self.announceIfNeeded(mapped, stableKeys: stableKeys)
                }
            } catch {
                Task { @MainActor in
                    self.statusMessage = "프레임 추론 실패: \(error.localizedDescription)"
                }
            }
        }
    }

    /// 가장 가까운(거리 미상이면 신뢰도 높은) 객체 1개를 골라 방향·거리와 함께 안내한다.
    /// - 순간적으로만 잡힌 오탐은 `stableKeys`(연속 관측된 트랙)만 후보로 삼아 거른다.
    /// - 근접 시 "주의" 경고를 보내되, 같은 객체가 계속 가까이 있으면 진행 중 발화를 끊지 않는다.
    private func announceIfNeeded(_ detections: [DetectedObject], stableKeys: Set<String>) {
        guard isVoiceEnabled, !detections.isEmpty else { return }

        let stable = detections.filter {
            stableKeys.contains(trackKey(label: $0.label.lowercased(), boundingBox: $0.boundingBox))
        }
        guard let candidate = selectAnnouncementCandidate(stable) else { return }

        let now = Date()
        let key = candidate.localizedLabel
        let isUrgent = (candidate.estimatedDistanceMeters.map { $0 <= nearWarningDistance }) ?? false
        let cooldown = isUrgent ? urgentAnnouncementCooldown : labelAnnouncementCooldown

        if let last = labelAnnouncementTimes[key], now.timeIntervalSince(last) < cooldown {
            return
        }

        // 같은 객체가 직전에도 "주의" 상태였다면 새로 끼어들지(force) 않고 자연스럽게 큐에 맡긴다.
        // 처음 가까워진 순간에만 진행 중 발화를 끊어 즉시 경고한다.
        let wasRecentlyUrgent = lastUrgentAnnouncementTimes[key]
            .map { now.timeIntervalSince($0) < urgentAnnouncementCooldown * 2 } ?? false
        let shouldInterrupt = isUrgent && !wasRecentlyUrgent

        labelAnnouncementTimes[key] = now
        if isUrgent {
            lastUrgentAnnouncementTimes[key] = now
        }
        pruneAnnouncements(now: now)

        let phrase = guidancePhrase(for: candidate, isUrgent: isUrgent)
        speech.enqueue(phrase, force: shouldInterrupt)
    }

    /// 충돌 관련성이 가장 큰 객체를 선택한다.
    /// - 거리 정보가 있으면 가장 가까운 객체(LiDAR 기기)
    /// - 거리 정보가 없으면 이동 장애물 우선 + 최고 신뢰도(비 LiDAR 기기 기존 동작)
    private func selectAnnouncementCandidate(_ detections: [DetectedObject]) -> DetectedObject? {
        let withDistance = detections.filter { $0.estimatedDistanceMeters != nil }
        if let nearest = withDistance.min(by: {
            ($0.estimatedDistanceMeters ?? .greatestFiniteMagnitude) < ($1.estimatedDistanceMeters ?? .greatestFiniteMagnitude)
        }) {
            return nearest
        }

        let preferred = detections.filter { preferredLabels.contains($0.label.lowercased()) }
        let pool = preferred.isEmpty ? detections : preferred
        return pool.max(by: { $0.confidence < $1.confidence })
    }

    private func guidancePhrase(for detection: DetectedObject, isUrgent: Bool) -> String {
        let direction = directionPhrase(for: detection.boundingBox)
        guard let distance = detection.estimatedDistanceMeters, distance.isFinite else {
            return "\(direction)에 \(detection.localizedLabel)"
        }
        if isUrgent {
            return String(format: "주의, %@ %.1f미터 앞에 %@", direction, distance, detection.localizedLabel)
        }
        return String(format: "%@ %.1f미터 앞에 %@", direction, distance, detection.localizedLabel)
    }

    private func directionPhrase(for boundingBox: CGRect) -> String {
        let midX = boundingBox.midX
        if midX < 0.4 {
            return "왼쪽"
        } else if midX > 0.6 {
            return "오른쪽"
        } else {
            return "정면"
        }
    }

    /// Vision 정규화 bbox 중심점 주변 depth를 샘플링해 중앙값 거리(m)를 계산한다.
    private func estimateDistance(for boundingBox: CGRect, depthData: CVPixelBuffer?) -> Float? {
        guard let depthData else { return nil }
        guard CVPixelBufferGetPixelFormatType(depthData) == kCVPixelFormatType_DepthFloat32 else { return nil }

        CVPixelBufferLockBaseAddress(depthData, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthData, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(depthData) else { return nil }

        let width = CVPixelBufferGetWidth(depthData)
        let height = CVPixelBufferGetHeight(depthData)
        guard width > 0, height > 0 else { return nil }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthData)
        let centerX = max(0, min(width - 1, Int(CGFloat(width) * boundingBox.midX)))
        let centerY = max(0, min(height - 1, Int(CGFloat(height) * (1 - boundingBox.midY))))

        let radius = 2
        var samples: [Float] = []
        samples.reserveCapacity((radius * 2 + 1) * (radius * 2 + 1))

        for y in max(0, centerY - radius)...min(height - 1, centerY + radius) {
            let rowPointer = baseAddress.advanced(by: y * bytesPerRow).assumingMemoryBound(to: Float32.self)
            for x in max(0, centerX - radius)...min(width - 1, centerX + radius) {
                let value = rowPointer[x]
                if value.isFinite, value > 0.05, value < 12.0 {
                    samples.append(value)
                }
            }
        }

        guard !samples.isEmpty else { return nil }
        samples.sort()
        return samples[samples.count / 2]
    }

    /// 라벨+대략 위치로 같은 객체를 추적해 거리를 EMA로 평활화한다(프레임 간 튐 완화).
    /// processingQueue에서만 호출된다.
    private func smoothedDistance(rawDistance: Float?, boundingBox: CGRect, label: String, now: Date) -> Float? {
        guard let raw = rawDistance else { return nil }

        let key = trackKey(label: label, boundingBox: boundingBox)
        let blended: Float
        if let previous = smoothedDistances[key], now.timeIntervalSince(previous.timestamp) < distanceTrackTTL {
            blended = distanceSmoothingFactor * raw + (1 - distanceSmoothingFactor) * previous.value
        } else {
            blended = raw
        }

        smoothedDistances[key] = (blended, now)
        return blended
    }

    private func trackKey(label: String, boundingBox: CGRect) -> String {
        let gridX = Int((boundingBox.midX * 6).rounded(.down))
        let gridY = Int((boundingBox.midY * 6).rounded(.down))
        return "\(label)#\(gridX)x\(gridY)"
    }

    private func pruneSmoothedDistances(now: Date) {
        smoothedDistances = smoothedDistances.filter { now.timeIntervalSince($0.value.timestamp) < distanceTrackTTL }
    }

    /// 트랙별 연속 관측 횟수를 갱신하고, 안내해도 좋은(연속 관측이 충분한) 트랙 키 집합을 반환한다.
    /// 짧게 깜빡이는 오탐은 관측이 누적되지 못해 자연히 제외된다. processingQueue에서만 호출된다.
    private func updateTrackObservations(_ detections: [DetectedObject], now: Date) -> Set<String> {
        var stableKeys: Set<String> = []

        for detection in detections {
            let key = trackKey(label: detection.label.lowercased(), boundingBox: detection.boundingBox)
            let count: Int
            if let previous = trackObservationCounts[key], now.timeIntervalSince(previous.lastSeen) < distanceTrackTTL {
                count = previous.count + 1
            } else {
                count = 1
            }
            trackObservationCounts[key] = (count, now)
            if count >= minObservationsToAnnounce {
                stableKeys.insert(key)
            }
        }

        trackObservationCounts = trackObservationCounts.filter { now.timeIntervalSince($0.value.lastSeen) < distanceTrackTTL }
        return stableKeys
    }

    private func pruneAnnouncements(now: Date) {
        let window = labelAnnouncementCooldown * 2
        labelAnnouncementTimes = labelAnnouncementTimes.filter { now.timeIntervalSince($0.value) < window }
        lastUrgentAnnouncementTimes = lastUrgentAnnouncementTimes.filter { now.timeIntervalSince($0.value) < window }
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
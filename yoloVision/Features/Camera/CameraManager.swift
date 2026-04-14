import AVFoundation
import Combine
import Foundation

enum CameraLens: String, CaseIterable, Identifiable {
    case wide
    case ultraWide
    case telephoto

    var id: String { rawValue }

    var title: String {
        switch self {
        case .wide:
            return "광각"
        case .ultraWide:
            return "초광각"
        case .telephoto:
            return "망원"
        }
    }

    var deviceType: AVCaptureDevice.DeviceType {
        switch self {
        case .wide:
            return .builtInWideAngleCamera
        case .ultraWide:
            return .builtInUltraWideCamera
        case .telephoto:
            return .builtInTelephotoCamera
        }
    }
}

enum CameraAuthorizationState: Equatable {
    case notDetermined
    case requesting
    case authorized
    case denied
    case restricted
    case failed(String)
}

final class CameraManager: NSObject, ObservableObject {
    @Published private(set) var authorizationState: CameraAuthorizationState = .notDetermined
    @Published private(set) var frameCounter: Int = 0
    @Published private(set) var isRunning: Bool = false
    @Published private(set) var availableLenses: [CameraLens] = []
    @Published var selectedLens: CameraLens = .wide
    @Published private(set) var activeLens: CameraLens = .wide

    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "yoloVision.camera.session", qos: .userInitiated)
    private let outputQueue = DispatchQueue(label: "yoloVision.camera.output", qos: .userInitiated)
    private let videoOutput = AVCaptureVideoDataOutput()
    private let frameHandlerQueue = DispatchQueue(label: "yoloVision.camera.frame-handler", qos: .userInitiated)

    private var isConfigured = false
    private var currentInput: AVCaptureDeviceInput?
    private var frameHandler: ((CVPixelBuffer) -> Void)?

    override init() {
        super.init()
        authorizationState = Self.mapAuthorizationStatus(AVCaptureDevice.authorizationStatus(for: .video))
        let lenses = Self.discoverAvailableLenses()
        availableLenses = lenses
        if lenses.contains(.wide) {
            selectedLens = .wide
        } else if let first = lenses.first {
            selectedLens = first
        }
    }

    func start() async {
        let isAuthorized = await ensureAuthorization()
        guard isAuthorized else { return }

        do {
            try await configureIfNeeded()
            await startSession()
        } catch {
            await MainActor.run {
                self.authorizationState = .failed("카메라 세션 구성에 실패했습니다: \(error.localizedDescription)")
                self.isRunning = false
            }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning {
                self.session.stopRunning()
            }

            Task { @MainActor in
                self.isRunning = false
            }
        }
    }

    func setFrameHandler(_ handler: @escaping (CVPixelBuffer) -> Void) {
        frameHandlerQueue.async {
            self.frameHandler = handler
        }
    }

    func clearFrameHandler() {
        frameHandlerQueue.async {
            self.frameHandler = nil
        }
    }

    func switchLens(to lens: CameraLens) async {
        guard lens != activeLens else { return }

        do {
            try await switchLensInternal(to: lens)
        } catch {
            await MainActor.run {
                self.authorizationState = .failed("렌즈 전환 실패: \(error.localizedDescription)")
                self.selectedLens = self.activeLens
            }
        }
    }

    private func ensureAuthorization() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            await MainActor.run {
                self.authorizationState = .authorized
            }
            return true
        case .notDetermined:
            await MainActor.run {
                self.authorizationState = .requesting
            }

            let granted = await AVCaptureDevice.requestAccess(for: .video)
            await MainActor.run {
                self.authorizationState = granted ? .authorized : .denied
            }
            return granted
        case .denied:
            await MainActor.run {
                self.authorizationState = .denied
                self.isRunning = false
            }
            return false
        case .restricted:
            await MainActor.run {
                self.authorizationState = .restricted
                self.isRunning = false
            }
            return false
        @unknown default:
            await MainActor.run {
                self.authorizationState = .failed("알 수 없는 카메라 권한 상태입니다.")
                self.isRunning = false
            }
            return false
        }
    }

    private func configureIfNeeded() async throws {
        guard !isConfigured else { return }

        try await withCheckedThrowingContinuation { continuation in
            sessionQueue.async {
                do {
                    self.session.beginConfiguration()
                    self.session.sessionPreset = .high

                    let preferredLens = self.availableLenses.contains(self.selectedLens)
                        ? self.selectedLens
                        : (self.availableLenses.contains(.wide) ? .wide : self.availableLenses.first ?? .wide)

                    try self.configureInput(for: preferredLens)

                    self.videoOutput.videoSettings = [
                        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
                    ]
                    self.videoOutput.alwaysDiscardsLateVideoFrames = true
                    self.videoOutput.setSampleBufferDelegate(self, queue: self.outputQueue)

                    guard self.session.canAddOutput(self.videoOutput) else {
                        throw CameraManagerError.cannotAddOutput
                    }
                    self.session.addOutput(self.videoOutput)

                    if let connection = self.videoOutput.connection(with: .video), connection.isVideoOrientationSupported {
                        connection.videoOrientation = .portrait
                    }

                    self.session.commitConfiguration()
                    self.isConfigured = true
                    continuation.resume(returning: ())
                } catch {
                    self.session.commitConfiguration()
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func startSession() async {
        await withCheckedContinuation { continuation in
            sessionQueue.async {
                if !self.session.isRunning {
                    self.session.startRunning()
                }

                Task { @MainActor in
                    self.isRunning = true
                }
                continuation.resume(returning: ())
            }
        }
    }

    private func configureInput(for lens: CameraLens) throws {
        guard let camera = AVCaptureDevice.default(lens.deviceType, for: .video, position: .back) else {
            throw CameraManagerError.lensUnavailable
        }

        let input = try AVCaptureDeviceInput(device: camera)
        let previousInput = currentInput
        if let previousInput {
            session.removeInput(previousInput)
        }

        guard session.canAddInput(input) else {
            if let previousInput, session.canAddInput(previousInput) {
                session.addInput(previousInput)
                currentInput = previousInput
            }
            throw CameraManagerError.cannotAddInput
        }

        session.addInput(input)
        currentInput = input
        Task { @MainActor in
            self.selectedLens = lens
            self.activeLens = lens
        }
    }

    private func switchLensInternal(to lens: CameraLens) async throws {
        try await withCheckedThrowingContinuation { continuation in
            sessionQueue.async {
                do {
                    self.session.beginConfiguration()
                    try self.configureInput(for: lens)
                    self.session.commitConfiguration()
                    continuation.resume(returning: ())
                } catch {
                    self.session.commitConfiguration()
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func discoverAvailableLenses() -> [CameraLens] {
        var result: [CameraLens] = []
        for lens in CameraLens.allCases {
            let hasLens = AVCaptureDevice.default(lens.deviceType, for: .video, position: .back) != nil
            if hasLens {
                result.append(lens)
            }
        }
        return result
    }

    private static func mapAuthorizationStatus(_ status: AVAuthorizationStatus) -> CameraAuthorizationState {
        switch status {
        case .authorized:
            return .authorized
        case .notDetermined:
            return .notDetermined
        case .denied:
            return .denied
        case .restricted:
            return .restricted
        @unknown default:
            return .failed("알 수 없는 카메라 권한 상태입니다.")
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        frameHandlerQueue.async {
            self.frameHandler?(pixelBuffer)
        }

        Task { @MainActor in
            self.frameCounter += 1
        }
    }
}

private enum CameraManagerError: LocalizedError {
    case backCameraUnavailable
    case lensUnavailable
    case cannotAddInput
    case cannotAddOutput

    var errorDescription: String? {
        switch self {
        case .backCameraUnavailable:
            return "후면 카메라를 찾을 수 없습니다."
        case .lensUnavailable:
            return "선택한 렌즈를 이 기기에서 사용할 수 없습니다."
        case .cannotAddInput:
            return "카메라 입력을 세션에 추가할 수 없습니다."
        case .cannotAddOutput:
            return "프레임 출력을 세션에 추가할 수 없습니다."
        }
    }
}
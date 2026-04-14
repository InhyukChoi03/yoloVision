//
//  ContentView.swift
//  yoloVision
//
//  Created by Inhyuk Choi on 4/11/26.
//

import SwiftUI

struct ContentView: View {
    private enum Screen {
        case menu
        case camera
    }

    @Environment(\.openURL) private var openURL
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var detectionService = DetectionService()
    @State private var currentScreen: Screen = .menu

    var body: some View {
        Group {
            switch currentScreen {
            case .menu:
                menuView
            case .camera:
                cameraView
            }
        }
        .onDisappear {
            cameraManager.stop()
        }
    }

    private var menuView: some View {
        ZStack {
            LinearGradient(
                colors: [.black, .blue.opacity(0.6)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            VStack(spacing: 18) {
                Text("보행 보조")
                    .font(.largeTitle.bold())
                    .foregroundStyle(.white)

                Text("카메라를 실행해 주변 객체를 탐지합니다")
                    .font(.callout)
                    .foregroundStyle(.white.opacity(0.9))

                VStack(alignment: .leading, spacing: 8) {
                    Text("모델 선택")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.9))

                    Picker("모델", selection: $detectionService.selectedModelOption) {
                        ForEach(detectionService.availableModelOptions, id: \.self) { option in
                            Text(option).tag(option)
                        }
                    }
                    .pickerStyle(.menu)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(.black.opacity(0.25))
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                .frame(maxWidth: .infinity)

                Button("실행") {
                    currentScreen = .camera
                    Task {
                        detectionService.refreshModelOptions()
                        await detectionService.applySelectedModel()
                        await detectionService.startIfNeeded()
                        await cameraManager.start()
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }
            .padding(20)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .padding()
        }
    }

    private var cameraView: some View {
        ZStack {
            CameraPreviewView(session: cameraManager.session)
                .ignoresSafeArea()

            GeometryReader { geometry in
                ForEach(detectionService.latestDetections) { detection in
                    let rect = convertedRect(
                        from: detection.boundingBox,
                        imageSize: detection.imageSize,
                        in: geometry.size
                    )

                    ZStack(alignment: .topLeading) {
                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .stroke(Color.green, lineWidth: 2)

                        Text("\(detection.localizedLabel) \(Int(detection.confidence * 100))%")
                            .font(.caption2.bold())
                            .foregroundStyle(.white)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 4)
                            .background(Color.green.opacity(0.9))
                            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                            .offset(x: 4, y: 4)
                    }
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)
                }
            }
            .allowsHitTesting(false)

            VStack {
                HStack {
                    Text("Frames: \(cameraManager.frameCounter)")
                        .font(.caption.monospacedDigit())
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial)
                        .clipShape(Capsule())

                    if detectionService.isModelReady {
                        Text("AI 준비됨")
                            .font(.caption)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(.green.opacity(0.85))
                            .foregroundStyle(.white)
                            .clipShape(Capsule())

                        if let modelName = detectionService.activeModelName {
                            Text(modelName)
                                .font(.caption2)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 6)
                                .background(.black.opacity(0.6))
                                .foregroundStyle(.white)
                                .clipShape(Capsule())
                        }

                        Text("렌즈: \(cameraManager.activeLens.title)")
                            .font(.caption2)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 6)
                            .background(.black.opacity(0.6))
                            .foregroundStyle(.white)
                            .clipShape(Capsule())
                    }

                    Spacer()
                }

                if detectionService.isModelReady {
                    HStack {
                        Text(detectionService.livePerformanceLog)
                            .font(.caption2.monospacedDigit())
                            .lineLimit(2)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .background(.ultraThinMaterial)
                            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        Spacer()
                    }
                    .padding(.top, 8)
                }

                if !detectionService.topDetectedLabels.isEmpty {
                    HStack {
                        Text("탐지: \(detectionService.topDetectedLabels.joined(separator: ", "))")
                            .font(.subheadline.weight(.semibold))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(.ultraThinMaterial)
                            .clipShape(Capsule())
                        Spacer()
                    }
                    .padding(.top, 8)
                }

                Spacer()

                if let message = statusMessage {
                    VStack(spacing: 10) {
                        Text(message)
                            .font(.callout)
                            .multilineTextAlignment(.center)

                        if cameraManager.authorizationState == .denied {
                            Button("설정 열기") {
                                if let url = URL(string: UIApplication.openSettingsURLString) {
                                    openURL(url)
                                }
                            }
                            .buttonStyle(.borderedProminent)
                        }
                    }
                    .padding(14)
                    .frame(maxWidth: .infinity)
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                }

                HStack(spacing: 10) {
                    Button(cameraManager.isRunning ? "실행 중" : "실행") {
                        Task {
                            await detectionService.startIfNeeded()
                            await cameraManager.start()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(cameraManager.isRunning)

                    Button("멈춤") {
                        cameraManager.stop()
                    }
                    .buttonStyle(.bordered)
                    .disabled(!cameraManager.isRunning)

                    Button("메인메뉴") {
                        cameraManager.stop()
                        currentScreen = .menu
                    }
                    .buttonStyle(.bordered)
                }

                if !cameraManager.availableLenses.isEmpty {
                    Picker("렌즈", selection: $cameraManager.selectedLens) {
                        ForEach(cameraManager.availableLenses) { lens in
                            Text(lens.title).tag(lens)
                        }
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: cameraManager.selectedLens) { _, newLens in
                        Task {
                            await cameraManager.switchLens(to: newLens)
                        }
                    }
                }

                Picker("모델", selection: $detectionService.selectedModelOption) {
                    ForEach(detectionService.availableModelOptions, id: \.self) { option in
                        Text(option).tag(option)
                    }
                }
                .pickerStyle(.menu)
                .onChange(of: detectionService.selectedModelOption) { _, _ in
                    Task {
                        await detectionService.applySelectedModel()
                    }
                }
            }
            .padding()

            if let detectionError = detectionService.statusMessage {
                VStack {
                    Spacer()
                    Text(detectionError)
                        .font(.footnote)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .frame(maxWidth: .infinity)
                        .background(.black.opacity(0.6))
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                }
                .padding(.horizontal)
                .padding(.bottom, 90)
            }
        }
        .onAppear {
            detectionService.refreshModelOptions()
            cameraManager.setFrameHandler { pixelBuffer in
                detectionService.handleFrame(pixelBuffer)
            }
        }
        .onDisappear {
            cameraManager.clearFrameHandler()
        }
    }

    private var statusMessage: String? {
        switch cameraManager.authorizationState {
        case .authorized:
            return nil
        case .notDetermined, .requesting:
            return "카메라 권한을 요청하는 중입니다."
        case .denied:
            return "카메라 접근이 거부되었습니다. 설정에서 카메라 권한을 허용해 주세요."
        case .restricted:
            return "이 기기에서는 카메라 접근이 제한되어 있습니다."
        case .failed(let message):
            return message
        }
    }

    private func convertedRect(from normalizedRect: CGRect, imageSize: CGSize, in viewSize: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else {
            return .zero
        }

        let imageX = normalizedRect.minX * imageSize.width
        let imageY = (1 - normalizedRect.maxY) * imageSize.height
        let imageWidth = normalizedRect.width * imageSize.width
        let imageHeight = normalizedRect.height * imageSize.height

        let scale = max(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
        let scaledImageWidth = imageSize.width * scale
        let scaledImageHeight = imageSize.height * scale
        let xOffset = (scaledImageWidth - viewSize.width) / 2
        let yOffset = (scaledImageHeight - viewSize.height) / 2

        let x = imageX * scale - xOffset
        let y = imageY * scale - yOffset
        let width = imageWidth * scale
        let height = imageHeight * scale

        return CGRect(x: x, y: y, width: width, height: height)
    }
}

#Preview {
    ContentView()
}

# yoloVision

iPhone 후면 카메라로 주변 객체를 실시간으로 인식하고, 한국어 음성으로 안내하는 **보행 보조 iOS 앱**입니다. 온디바이스 YOLO(CoreML) 추론을 사용하므로 네트워크 없이 기기에서만 동작합니다.

## 주요 기능

- 후면 카메라 실시간 프리뷰 (광각 / 초광각 / 망원 렌즈 전환)
- Vision + CoreML 기반 YOLO 객체 탐지 및 바운딩 박스 오버레이
- 여러 YOLO 모델 번들(yolov8n, yolo11s/m/l/x) 중 선택 / 자동 우선순위 로딩
- 보행 관련 COCO 클래스(사람/자동차/자전거/버스/트럭/신호등 등)만 필터링해 탐지
- LiDAR 지원 Pro 기기에서 depth map을 함께 수집해 객체 **거리(m)** 추정
- **한국어 TTS 음성 안내**: 방향(왼쪽/정면/오른쪽)과 객체명을 함께 안내
  - 거리 추정이 가능하면 `정면 2.1미터 앞에 사람` 형태로 음성 안내
  - 직렬 발화로 겹침 방지, 최소 발화 간격 / 중복 문장 억제 / 라벨별 쿨다운
- 실시간 추론 FPS·지연시간 성능 로그

## 요구 사항

- Xcode 26.3 이상
- iOS 26.2 이상 실기기 (카메라·CoreML은 시뮬레이터에서 제한적)
- 거리 안내(LiDAR depth)는 Pro 계열 iPhone/iPad Pro에서만 동작
- Apple Developer 서명 계정

## 프로젝트 구조

```
yoloVision/
├── yoloVisionApp.swift            # 앱 엔트리
├── ContentView.swift              # 메뉴/카메라 화면 및 오버레이 UI
├── Domain/Models/
│   └── DetectedObject.swift       # 탐지 결과 모델
├── Features/
│   ├── Camera/
│   │   ├── CameraManager.swift    # 권한·세션·렌즈 전환·RGB+Depth 프레임 콜백
│   │   └── CameraPreviewView.swift
│   ├── Detection/
│   │   ├── YOLOModelProvider.swift   # 번들 모델 탐색/로딩
│   │   └── DetectionService.swift    # 추론·후처리·음성 안내 트리거
│   └── Speech/
│       └── SpeechService.swift    # 한국어 TTS(직렬 큐/중복 억제)
└── Resources/ML/                  # CoreML 모델(.mlpackage) — git 미추적
```

> `.mlpackage` / `.mlmodelc` 모델 파일은 용량 때문에 `.gitignore`로 제외됩니다. 클론 후 아래 절차로 직접 생성/배치해야 빌드됩니다.

## 모델 준비 (macOS)

PyTorch YOLO 가중치를 CoreML로 변환해 앱 번들에 넣습니다.

```bash
# 1) 가상환경
python3 -m venv .venv
source .venv/bin/activate

# 2) 패키지 설치
pip install --upgrade pip
pip install ultralytics coremltools

# 3) CoreML export (NMS 포함 필수)
yolo export model=yolov8n.pt format=coreml nms=True imgsz=640
yolo export model=yolo11s.pt format=coreml nms=True imgsz=640
yolo export model=yolo11m.pt format=coreml nms=True imgsz=960
yolo export model=yolo11l.pt format=coreml nms=True imgsz=960
yolo export model=yolo11x.pt format=coreml nms=True imgsz=960
```

생성된 `*.mlpackage`를 `yoloVision/Resources/ML/`에 복사합니다.

> **중요:** `nms=True`로 export해야 Vision이 `VNRecognizedObjectObservation`(박스+라벨)을 반환합니다. NMS 없이 export하면 출력 형식이 달라 박스/라벨이 비게 되며, 앱이 안내 메시지로 알려 줍니다.

모델 인식 우선순위(자동 모드): `yolo11x → yolo11l → yolo11m → yolo11s → yolo11n → yolov8x … → yolov8n`.

## 실행

1. `yoloVision.xcodeproj`를 Xcode로 엽니다.
2. 서명 팀(Signing & Capabilities)을 본인 계정으로 설정합니다.
3. 실기기를 선택하고 실행합니다.
4. 메뉴에서 모델과 음성 안내 여부를 선택한 뒤 **실행**을 누릅니다.
5. 카메라 권한을 허용하면 탐지 박스와 음성 안내가 시작됩니다.

## 동작 개요

```
카메라 프레임 → CameraManager → DetectionService(Vision+CoreML)
   → 박스/라벨 오버레이(ContentView)
   → 보행 우선 객체 선택 → SpeechService(한국어 TTS)
```

## 로드맵

- [x] Phase 1. 카메라 기반 (권한/프리뷰/프레임)
- [x] Phase 2. Vision + CoreML 실시간 탐지·오버레이
- [x] Phase 3. 한국어 TTS 안내 (중복 억제·쿨다운)
- [ ] Phase 4. 통합/성능/접근성 (VoiceOver 강화, 장시간 발열·메모리 점검)

자세한 설계는 [`PLAN.md`](PLAN.md)를 참고하세요.

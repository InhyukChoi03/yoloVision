# yoloVision 실행 계획 (Phase 1~4)

이 문서는 프로젝트 내부에서 바로 확인하기 위한 설계 중심 계획 문서입니다.

## 1) 현재 코드 상태 요약

- 앱 엔트리
  - `yoloVisionApp.swift`에서 `ContentView`를 루트로 사용.

- UI 흐름
  - `ContentView.swift`에서 메뉴 화면 -> 카메라 화면 전환 구조 구현.
  - 메뉴에서 모델 선택(`자동/수동`) 후 실행.
  - 카메라 화면에서:
    - 프리뷰 표시
    - 탐지 박스 오버레이 표시
    - 프레임 카운터/모델 준비 상태/렌즈 상태 표시
    - 렌즈 변경, 모델 변경, 시작/정지 버튼 제공
    - 권한/오류 메시지 표시

- 카메라 모듈
  - `Features/Camera/CameraManager.swift`:
    - 카메라 권한 요청/상태 관리
    - `AVCaptureSession` 구성
    - 렌즈 탐색(광각/초광각/망원) 및 전환
    - 프레임 콜백 핸들러 제공
  - `Features/Camera/CameraPreviewView.swift`:
    - `AVCaptureVideoPreviewLayer` 기반 프리뷰 표시

- 탐지 모듈
  - `Features/Detection/YOLOModelProvider.swift`:
    - 번들 내 YOLO 모델 이름 우선순위 탐색
    - `.mlmodelc`, `.mlpackage` 로딩 지원
    - 선택 모델/자동 모델 로딩
  - `Features/Detection/DetectionService.swift`:
    - `VNCoreMLRequest` 생성 및 프레임 추론
    - 최소 추론 간격(`0.12s`)으로 부하 제어
    - confidence threshold 적용
    - 주요 라벨 우선 처리(person/chair 등)
    - 한국어 라벨 매핑
    - 최신 박스/상위 라벨/상태 메시지 퍼블리시

- 도메인 모델
  - `Domain/Models/DetectedObject.swift`:
    - 라벨/신뢰도/바운딩박스/이미지크기/타임스탬프 구조체 정의

- 현재 미구현/보완 필요
  - 한국어 TTS 음성 큐(겹침 방지/중복 억제)
  - 탐지 후처리 고도화(NMS 후 안정화 규칙, 쿨다운 정책 구체화)
  - 접근성(VoiceOver 상태 안내), 장시간 성능/발열 점검

## 2) Phase 실행 계획

### Phase 1. 카메라 기반
- 목표: 실행 즉시 후면 카메라 안정 표시 + 프레임 수급 보장
- 완료 기준:
  - 카메라 권한 흐름 정상
  - 카메라 프리뷰/프레임 카운터 정상 동작

### Phase 2. Vision + CoreML
- 목표: YOLO 추론 안정화 및 결과 품질 개선
- 작업:
  - 모델 선택/전환 안정화
  - threshold 튜닝(오탐/미탐 균형)
  - 동일 객체 반복 알림 완화 규칙(시간 기반 쿨다운)
- 완료 기준:
  - 실시간 탐지와 오버레이가 안정적으로 유지

### Phase 3. 한국어 TTS 파이프라인
- 목표: 안내 음성 품질 확보
- 작업:
  - `SpeechQueueManager` 구현(ko-KR)
  - 최소 발화 간격/중복 문장 억제/직렬 큐 처리
  - 인터럽트 후 복구
- 완료 기준:
  - 겹침 없이 자연스러운 안내

### Phase 4. 통합/성능/접근성
- 목표: 실제 보행 시나리오 안정성 확보
- 작업:
  - Camera -> Detection -> Speech end-to-end 정합
  - Main/UI 스레드와 추론/음성 스레드 분리 점검
  - VoiceOver 텍스트/상태 공지 강화
  - Instruments(Time Profiler/Memory/Energy) 점검
- 완료 기준:
  - 장시간 실행에서도 허용 가능한 성능/발열/메모리

## 3) 모델 설치/변환 가이드 (macOS)

1. 가상환경 생성/활성화
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`

2. 패키지 설치
   - `pip install --upgrade pip`
   - `pip install ultralytics coremltools`

3. CoreML export 예시
   - `yolo export model=yolo11s.pt format=coreml nms=True imgsz=640`
   - `yolo export model=yolo11m.pt format=coreml nms=True imgsz=960`
   - `yolo export model=yolo11l.pt format=coreml nms=True imgsz=960`
   - `yolo export model=yolo11x.pt format=coreml nms=True imgsz=960`
   - `yolo export model=yolov8n.pt format=coreml nms=True imgsz=640`

4. 결과 반영
   - export된 `.mlpackage`를 앱 번들 리소스 위치에 추가
  - 현재 프로젝트 구조 기준으로 `yoloVision/Resources/ML/`에 정리

## 4) 작업 원칙

- 우선순위: 안정성 > 실시간성 > 탐지 범위 확장
- 모델 변경 시 UI/추론 파이프라인 재초기화 동작을 명시적으로 유지
- 추론 실패/권한 실패 메시지는 사용자 행동 유도형 문구로 유지

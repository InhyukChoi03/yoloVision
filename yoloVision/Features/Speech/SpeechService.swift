import AVFoundation
import Combine
import Foundation

/// 한국어 안내 음성을 직렬로 재생하는 TTS 서비스.
/// - 같은 문장 반복 억제(중복 쿨다운)
/// - 최소 발화 간격으로 과다 발화 방지
/// - 말하는 중에는 새 문장을 무시(직렬 처리)해 겹침 방지
final class SpeechService: NSObject, ObservableObject {
    @Published var isEnabled: Bool = true {
        didSet {
            if !isEnabled {
                stop()
            }
        }
    }

    @Published private(set) var isSpeaking: Bool = false
    @Published private(set) var lastSpokenText: String?

    private let synthesizer = AVSpeechSynthesizer()
    private let language = "ko-KR"

    private var recentlySpoken: [String: Date] = [:]
    private let duplicateCooldown: TimeInterval = 3.0

    private var lastUtteranceStartedAt = Date.distantPast
    private let minUtteranceInterval: TimeInterval = 1.2

    private var audioSessionConfigured = false

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    /// 안내 문장을 큐에 넣는다.
    /// - Parameters:
    ///   - text: 읽을 문장
    ///   - force: 간격/중복/진행중 검사를 건너뛰고 즉시 발화(중요 경고용)
    func enqueue(_ text: String, force: Bool = false) {
        guard isEnabled else { return }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        let now = Date()

        if !force {
            if synthesizer.isSpeaking { return }
            if now.timeIntervalSince(lastUtteranceStartedAt) < minUtteranceInterval { return }
            if let last = recentlySpoken[trimmed], now.timeIntervalSince(last) < duplicateCooldown { return }
        }

        configureAudioSessionIfNeeded()

        if force, synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }

        let utterance = AVSpeechUtterance(string: trimmed)
        utterance.voice = AVSpeechSynthesisVoice(language: language)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.pitchMultiplier = 1.0
        utterance.postUtteranceDelay = 0.1

        lastUtteranceStartedAt = now
        recentlySpoken[trimmed] = now
        pruneRecentlySpoken(now: now)

        isSpeaking = true
        lastSpokenText = trimmed
        synthesizer.speak(utterance)
    }

    func stop() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        recentlySpoken.removeAll()
        lastUtteranceStartedAt = .distantPast
        isSpeaking = false
    }

    private func pruneRecentlySpoken(now: Date) {
        let window = duplicateCooldown * 2
        recentlySpoken = recentlySpoken.filter { now.timeIntervalSince($0.value) < window }
    }

    private func configureAudioSessionIfNeeded() {
        guard !audioSessionConfigured else { return }

        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .voicePrompt, options: [.duckOthers, .mixWithOthers])
            try session.setActive(true)
            audioSessionConfigured = true
        } catch {
            audioSessionConfigured = false
        }
    }
}

extension SpeechService: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = false
        }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = false
        }
    }
}

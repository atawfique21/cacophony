import CacophonyRouting
import Accelerate
import AVFoundation
import CoreMedia
import Foundation
import Synchronization

/// How much audio sits between the driver and the speakers.
struct LatencySnapshot {
    /// Seconds of audio waiting in the ring buffer.
    let ring: Double
    /// Seconds the enqueued timeline runs ahead of the synchronizer clock.
    let lead: Double

    /// Seconds between the driver delivering audio and AirPlay playing it.
    var total: Double { ring + lead }
}

/// Plays a live interleaved Float32 stream through AVSampleBufferAudioRenderer,
/// which is the pipeline that can be attached to an (private) AVOutputContext
/// for AirPlay 2 group playback.
///
/// AirPlay plays each sample at its presentation time, so audio from the
/// driver plays after the time it waits in the ring plus how far the enqueued
/// timeline runs ahead of the synchronizer clock. The output holds that total
/// at `targetDelay`:
///
/// - Short driver stalls need nothing: the renderer has the whole delay
///   queued, and the late audio refills it.
/// - While the driver is idle, silence keeps the timeline on target, so audio
///   resumes at the same delay.
/// - Drift and jitter are corrected one 20 ms chunk at a time, dropping a
///   chunk or inserting silence where the audio is quiet.
/// - Backlog after a long stall is dropped at once.
///
/// The driver reports the same delay to macOS, which is what lets players
/// keep video in sync.
final class RendererOutput {
    let renderer = AVSampleBufferAudioRenderer()
    private let synchronizer = AVSampleBufferRenderSynchronizer()
    private let ring: RingBuffer
    private let format: AVAudioFormat
    private let queue = DispatchQueue(label: "cacophony.renderer")
    private var timer: DispatchSourceTimer?
    private var formatDescription: CMAudioFormatDescription?
    private var silence: CMBlockBuffer?
    private var started = false

    /// Presentation time of the next chunk, in frames.
    private let timelineFrames = Atomic<Int64>(0)
    private let liveFrames = Atomic<Int64>(0)

    /// Driver frames handed to the renderer, excluding inserted silence.
    /// Used to detect a stuck connection.
    var enqueuedFrames: Int64 { liveFrames.load(ordering: .relaxed) }

    /// Timer ticks that found the renderer ready and the source ring empty.
    let ringEmptyTicks = Atomic<Int>(0)
    /// Silence frames inserted to hold the delay.
    let paddedFrames = Atomic<Int>(0)
    /// Driver frames dropped to hold the delay.
    let droppedFrames = Atomic<Int>(0)
    /// Chunk-level and bulk corrections made while audio was flowing.
    let corrections = Atomic<Int>(0)

    /// Seconds from the driver delivering audio to AirPlay playing it.
    let targetDelay: Double

    private let sampleRate: Double
    private let chunkFrames = 960  // 20 ms
    private var chunkSeconds: Double { Double(chunkFrames) / sampleRate }

    // Controller state, only touched on `queue`.
    private var smoothedError = 0.0
    private var lastCorrectionAt: TimeInterval = 0
    private var lastChunkPeak: Float = 1
    private var lastWritten = 0
    private var lastWriteAt: TimeInterval = 0

    /// Error that gets corrected at the next quiet chunk.
    private static let tolerance = 0.015
    /// Error that gets corrected even when the audio is loud.
    private static let forcedError = 0.04
    /// Backlog that gets dropped at once.
    private static let bulkError = 0.1
    /// Peak below which a chunk can be dropped or followed by silence unheard.
    private static let quietPeak: Float = 0.02
    private static let correctionInterval: TimeInterval = 0.5
    /// No driver audio for this long counts as idle. Shorter stalls recover on
    /// their own from the renderer's queued audio.
    private static let idleAfter: TimeInterval = 0.15
    /// The error jitters by about one driver packet (10 ms); smooth it over
    /// roughly 0.4 s before correcting.
    private static let smoothing = 0.05

    init(format: AVAudioFormat, ring: RingBuffer, targetDelay: Double) {
        self.format = format
        self.ring = ring
        self.targetDelay = targetDelay
        self.sampleRate = format.sampleRate
    }

    func start() throws {
        var asbd = format.streamDescription.pointee
        var description: CMAudioFormatDescription?
        try check(
            CMAudioFormatDescriptionCreate(
                allocator: kCFAllocatorDefault,
                asbd: &asbd,
                layoutSize: 0,
                layout: nil,
                magicCookieSize: 0,
                magicCookie: nil,
                extensions: nil,
                formatDescriptionOut: &description
            ),
            "CMAudioFormatDescriptionCreate"
        )
        formatDescription = description
        silence = makeBlock(fromRing: false)?.block

        CACRendererAddToSynchronizer(synchronizer, renderer)

        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now(), repeating: .milliseconds(20), leeway: .milliseconds(5))
        timer.setEventHandler { [weak self] in self?.pump() }
        timer.resume()
        self.timer = timer
    }

    func latencySnapshot() -> LatencySnapshot {
        LatencySnapshot(ring: ringSeconds(), lead: currentLead())
    }

    private func clockSeconds() -> Double {
        let clock = CMTimeGetSeconds(synchronizer.currentTime())
        return clock.isFinite ? clock : 0
    }

    private func currentLead() -> Double {
        Double(timelineFrames.load(ordering: .relaxed)) / sampleRate - clockSeconds()
    }

    private func ringSeconds() -> Double {
        Double(ring.framesAvailable) / sampleRate
    }

    private func pump() {
        let now = ProcessInfo.processInfo.systemUptime
        if !started {
            // Start with the whole delay buffered so the stream begins on target.
            guard ring.framesAvailable >= Int(targetDelay * sampleRate) else { return }
            synchronizer.setRate(1.0, time: .zero)
            started = true
            lastWriteAt = now
        }

        let written = ring.framesWritten
        if written != lastWritten {
            lastWritten = written
            lastWriteAt = now
        }
        let driverIdle = now - lastWriteAt > Self.idleAfter

        var lead = currentLead()
        if lead < 0 {
            // Starved for longer than the delay, e.g. while AirPlay reconnects.
            // Restart the timeline at the clock rather than enqueueing audio
            // stamped in the past.
            timelineFrames.store(Int64((clockSeconds() * sampleRate).rounded()), ordering: .relaxed)
            lead = 0
        }

        if CACRendererIsReady(renderer), ring.framesAvailable == 0 {
            ringEmptyTicks.wrappingAdd(1, ordering: .relaxed)
        }

        let error = ringSeconds() + lead - targetDelay
        smoothedError += (error - smoothedError) * Self.smoothing

        if error > Self.bulkError {
            // Backlog after a long stall: drop the oldest audio back to target.
            let dropped = ring.skip(frameCount: Int(error * sampleRate))
            droppedFrames.wrappingAdd(dropped, ordering: .relaxed)
            corrections.wrappingAdd(1, ordering: .relaxed)
            smoothedError = ringSeconds() + lead - targetDelay
        }

        // Live audio. Enqueueing moves audio from the ring to the renderer
        // without changing the total delay; capping the lead keeps any excess
        // in the ring, where it can be dropped.
        var dropNext = smoothedError > Self.tolerance && canCorrect(now)
        while lead < targetDelay + chunkSeconds,
              CACRendererIsReady(renderer),
              ring.framesAvailable >= chunkFrames {
            guard let chunk = makeBlock(fromRing: true) else { return }
            if dropNext, chunk.peak < Self.quietPeak || smoothedError > Self.forcedError {
                droppedFrames.wrappingAdd(chunk.frames, ordering: .relaxed)
                recordCorrection(now, change: -chunkSeconds)
                dropNext = false
                continue
            }
            guard enqueue(chunk.block) else { return }
            liveFrames.wrappingAdd(Int64(chunk.frames), ordering: .relaxed)
            lastChunkPeak = chunk.peak
            lead += chunkSeconds
        }

        if driverIdle {
            // Nothing is playing. Send any leftover audio (under one chunk,
            // zero-filled), then hold the timeline on target with silence.
            if ring.framesAvailable > 0, CACRendererIsReady(renderer),
               let chunk = makeBlock(fromRing: true), enqueue(chunk.block) {
                liveFrames.wrappingAdd(Int64(chunk.frames), ordering: .relaxed)
                lead += chunkSeconds
            }
            while lead + ringSeconds() < targetDelay - chunkSeconds / 2, CACRendererIsReady(renderer) {
                guard insertSilence() else { return }
                lead += chunkSeconds
            }
            smoothedError = ringSeconds() + lead - targetDelay
            lastChunkPeak = 0
        } else if smoothedError < -Self.tolerance, canCorrect(now),
                  lastChunkPeak < Self.quietPeak || smoothedError < -Self.forcedError,
                  CACRendererIsReady(renderer) {
            // Running short, usually from clock drift: add one chunk of silence
            // after a quiet chunk.
            guard insertSilence() else { return }
            recordCorrection(now, change: chunkSeconds)
        }
    }

    private func canCorrect(_ now: TimeInterval) -> Bool {
        now - lastCorrectionAt >= Self.correctionInterval
    }

    private func recordCorrection(_ now: TimeInterval, change: Double) {
        smoothedError += change
        lastCorrectionAt = now
        corrections.wrappingAdd(1, ordering: .relaxed)
    }

    private func insertSilence() -> Bool {
        guard let silence, enqueue(silence) else { return false }
        paddedFrames.wrappingAdd(chunkFrames, ordering: .relaxed)
        return true
    }

    /// One chunk of audio from the ring (zero-filled if short) or of silence.
    private func makeBlock(fromRing: Bool) -> (block: CMBlockBuffer, peak: Float, frames: Int)? {
        let channelCount = Int(format.channelCount)
        let sampleCount = chunkFrames * channelCount
        let byteCount = sampleCount * MemoryLayout<Float>.size

        var blockBuffer: CMBlockBuffer?
        guard CMBlockBufferCreateWithMemoryBlock(
            allocator: kCFAllocatorDefault,
            memoryBlock: nil,
            blockLength: byteCount,
            blockAllocator: kCFAllocatorDefault,
            customBlockSource: nil,
            offsetToData: 0,
            dataLength: byteCount,
            flags: kCMBlockBufferAssureMemoryNowFlag,
            blockBufferOut: &blockBuffer
        ) == kCMBlockBufferNoErr, let blockBuffer else { return nil }

        var lengthAtOffset = 0
        var totalLength = 0
        var dataPointer: UnsafeMutablePointer<CChar>?
        guard CMBlockBufferGetDataPointer(
            blockBuffer,
            atOffset: 0,
            lengthAtOffsetOut: &lengthAtOffset,
            totalLengthOut: &totalLength,
            dataPointerOut: &dataPointer
        ) == kCMBlockBufferNoErr,
        let dataPointer,
        lengthAtOffset >= byteCount
        else { return nil }

        let samples = UnsafeMutableRawPointer(dataPointer).assumingMemoryBound(to: Float.self)
        guard fromRing else {
            samples.update(repeating: 0, count: sampleCount)
            return (blockBuffer, 0, 0)
        }
        let frames = ring.read(into: samples, frameCount: chunkFrames)
        var peak: Float = 0
        vDSP_maxmgv(samples, 1, &peak, vDSP_Length(sampleCount))
        return (blockBuffer, peak, frames)
    }

    /// Enqueues one chunk at the end of the timeline.
    private func enqueue(_ block: CMBlockBuffer) -> Bool {
        guard let formatDescription else { return false }
        var sampleBuffer: CMSampleBuffer?
        let pts = CMTime(
            value: timelineFrames.load(ordering: .relaxed),
            timescale: CMTimeScale(sampleRate)
        )
        guard CMAudioSampleBufferCreateReadyWithPacketDescriptions(
            allocator: kCFAllocatorDefault,
            dataBuffer: block,
            formatDescription: formatDescription,
            sampleCount: chunkFrames,
            presentationTimeStamp: pts,
            packetDescriptions: nil,
            sampleBufferOut: &sampleBuffer
        ) == noErr, let sampleBuffer else { return false }

        CACRendererEnqueue(renderer, sampleBuffer)
        timelineFrames.wrappingAdd(Int64(chunkFrames), ordering: .relaxed)
        return true
    }

    func stop() {
        timer?.cancel()
        timer = nil
        queue.sync {}  // let an in-flight pump finish before the ring is reused
        synchronizer.setRate(0, time: .zero)
        CACRendererStopRequesting(renderer)
        CACRendererFlush(renderer)
    }
}

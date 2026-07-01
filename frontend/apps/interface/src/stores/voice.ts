/**
 * Voice store — availability + recorder slices.
 *
 * The recorder lives in the store (not a component) because the mic button
 * belongs to InputDock but transcripts must reach the session without a direct
 * import chain. WAV encoding decodes the browser's native MediaRecorder output
 * (webm / mp4 / ogg) via AudioContext.decodeAudioData, then writes a 44-byte
 * RIFF/WAVE header with interleaved 16-bit PCM samples.
 */
import { defineStore } from 'pinia';
import { ref } from 'vue';
import type { WakeLockHandle } from '@chalie/shared';
import { platform as adapter } from '@chalie/shared';
import { voice } from '../api/voice';
import { emit } from '../composables/useEventBus';
import { useSessionStore } from './session';

const POLL_INTERVAL_MS = 2000;
const MAX_POLL_MS = 60_000;

export const MAX_RECORD_MS = 10 * 60 * 1000; // 10 minutes

export type RecorderState = 'idle' | 'recording' | 'uploading';

export const useVoiceStore = defineStore('voice', () => {
  /** True once /voice/health returns status==='ok'. */
  const available = ref(false);
  /** True while the health poll is in flight (mic stays hidden until done). */
  const loading = ref(true);

  const recorderState = ref<RecorderState>('idle');

  // Internal refs — not reactive (raw objects, not observed by Vue).
  let _mediaRecorder: MediaRecorder | null = null;
  let _audioChunks: Blob[] = [];
  let _maxRecordTimer: ReturnType<typeof setTimeout> | null = null;
  let _wakeLock: WakeLockHandle | null = null;

  /**
   * Poll GET /voice/health until it terminates, then set `available`:
   *   'ok' → available, stop · 'unavailable' → hidden, stop
   *   'loading' → re-poll every 2s up to 60s · timeout/error → unavailable
   */
  async function checkAvailability(): Promise<void> {
    const deadline = Date.now() + MAX_POLL_MS;
    for (;;) {
      try {
        const data = await voice.health();
        if (data.status === 'ok') {
          available.value = true;
          loading.value = false;
          return;
        }
        if (data.status === 'unavailable') {
          available.value = false;
          loading.value = false;
          return;
        }
        // 'loading' — re-poll until deadline.
        if (Date.now() >= deadline) {
          available.value = false;
          loading.value = false;
          return;
        }
      } catch {
        available.value = false;
        loading.value = false;
        return;
      }
      await new Promise<void>((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
    }
  }

  /** True while a native on-device STT capture is in flight (Tauri only). */
  let _nativeSTT = false;

  /** Single entry point for the mic button. */
  async function toggleRecording(): Promise<void> {
    if (_nativeSTT) {
      _nativeSTT = false;
      recorderState.value = 'idle';
      try {
        await adapter.stopSTT();
      } catch (err) {
        console.debug('[voice] stopSTT:', err);
      }
      return;
    }
    if (recorderState.value === 'recording') {
      await _stopAndUpload();
      return;
    }
    if (recorderState.value !== 'idle') return; // 'uploading' — ignore clicks.

    // On the native runtime, capture on-device (server /voice/transcribe is
    // bypassed). Native partial/final results arrive as chalie:voice-transcript
    // window events emitted by the plugin, so we do NOT re-emit here.
    try {
      await adapter.startSTT();
      _nativeSTT = true;
      useSessionStore().errorMessage = null;
      recorderState.value = 'recording';
      return;
    } catch (err) {
      // STT_UNSUPPORTED (web) / permission-denied / no on-device model →
      // fall back to the MediaRecorder + /voice/transcribe path.
      console.debug('[voice] native STT unavailable, falling back:', err);
    }
    await _startRecording();
  }

  async function _startRecording(): Promise<void> {
    const session = useSessionStore();
    let stream: MediaStream;
    try {
      stream = await adapter.getUserMedia({ audio: true });
    } catch (err: unknown) {
      const name = err instanceof DOMException ? err.name : '';
      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        session.errorMessage = 'Microphone access denied';
      } else if (name === 'NotSupportedError') {
        // webPlatformAdapter maps non-secure context to NotSupportedError
        session.errorMessage = 'Microphone not available (requires HTTPS)';
      } else {
        session.errorMessage = 'Could not start recording';
      }
      // Mic is gated by `available`, but a race can reach here; leave disabled.
      available.value = false;
      return;
    }

    session.errorMessage = null;
    _audioChunks = [];

    // No mimeType constraint — we always convert to WAV before upload, so the
    // browser's default (webm/Chrome, mp4/iOS) is fine.
    _mediaRecorder = new MediaRecorder(stream);
    _mediaRecorder.ondataavailable = (e: BlobEvent) => {
      if (e.data.size > 0) _audioChunks.push(e.data);
    };

    _mediaRecorder.start(250);
    recorderState.value = 'recording';

    _wakeLock = adapter.createWakeLock();
    void _wakeLock.acquire();

    // Auto-stop at the 10-minute limit.
    _maxRecordTimer = setTimeout(() => {
      if (recorderState.value === 'recording') void _stopAndUpload();
    }, MAX_RECORD_MS);
  }

  async function _stopAndUpload(): Promise<void> {
    if (recorderState.value !== 'recording' || !_mediaRecorder) return;

    if (_maxRecordTimer !== null) {
      clearTimeout(_maxRecordTimer);
      _maxRecordTimer = null;
    }

    const recorder = _mediaRecorder;
    recorderState.value = 'uploading';
    void _wakeLock?.release();

    const chunks = await new Promise<Blob[]>((resolve) => {
      recorder.onstop = () => {
        recorder.stream.getTracks().forEach((t) => t.stop());
        // iOS Safari fires onstop before the last ondataavailable chunk;
        // a small defer lets the event queue flush first.
        setTimeout(() => resolve([..._audioChunks]), 100);
      };
      try {
        recorder.requestData();
      } catch (err) {
        // requestData() not supported on some browsers — recorder.stop()
        // still flushes the last chunk.
        console.debug('[voice] requestData unsupported:', err);
      }
      recorder.stop();
    });

    _mediaRecorder = null;

    if (chunks.length === 0) {
      recorderState.value = 'idle';
      return;
    }

    try {
      const mimeType = recorder.mimeType || 'audio/webm';
      const wav = await _convertToWav(chunks, mimeType);
      const text = await _transcribe(wav);
      if (text) {
        emit('chalie:voice-transcript', { text });
      }
    } catch (err: unknown) {
      console.error('[VoiceRecorder] STT error:', err);
      // Prefer server-supplied hint when voice deps are missing.
      useSessionStore().errorMessage =
        (err as { userMessage?: string }).userMessage ??
        (err instanceof Error ? err.message : 'Transcription failed');
    } finally {
      recorderState.value = 'idle';
    }
  }

  /**
   * Decode audio chunks (any format) and re-encode as 16-bit PCM WAV.
   * decodeAudioData handles webm, mp4, ogg — whatever the browser recorded.
   */
  async function _convertToWav(chunks: Blob[], mimeType: string): Promise<File> {
    const blob = new Blob(chunks, { type: mimeType });
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx = adapter.createAudioContext();
    const audioBuffer = await new Promise<AudioBuffer>((resolve, reject) => {
      audioCtx.decodeAudioData(arrayBuffer, resolve, reject);
    });
    void audioCtx.close();
    return _audioBufferToWav(audioBuffer);
  }

  /** Encode an AudioBuffer as a 16-bit PCM WAV File (44-byte RIFF/WAVE header). */
  function _audioBufferToWav(audioBuffer: AudioBuffer): File {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const numSamples = audioBuffer.length;
    const dataLength = numSamples * numChannels * 2; // 16-bit = 2 bytes/sample
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    const writeStr = (offset: number, str: string): void => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.codePointAt(i) ?? 0);
    };

    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);                            // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true); // byte rate
    view.setUint16(32, numChannels * 2, true);              // block align
    view.setUint16(34, 16, true);                           // bits per sample
    writeStr(36, 'data');
    view.setUint32(40, dataLength, true);

    const channels = Array.from<number, Float32Array>(
      { length: numChannels },
      (_, i) => audioBuffer.getChannelData(i),
    );
    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
      for (let c = 0; c < numChannels; c++) {
        const s = Math.max(-1, Math.min(1, channels[c][i] ?? 0));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        offset += 2;
      }
    }

    return new File([buffer], 'recording.wav', { type: 'audio/wav' });
  }

  /** Upload wav to /voice/transcribe; return the transcript text or null. */
  async function _transcribe(file: File): Promise<string | null> {
    const resp = await voice.transcribe(file);

    if (!resp.ok) {
      const body = await resp.json().catch(() => ({})) as {
        error?: string;
        reason?: string;
        hint?: string;
      };
      const err: Error & { userMessage?: string } = new Error(
        body.error ?? `STT error: ${resp.status}`,
      );
      if (body.reason === 'deps_missing') {
        err.userMessage = body.hint ?? body.error ?? 'Voice dependencies not installed';
      }
      throw err;
    }

    const data = await resp.json() as { text?: string };
    return data.text ?? null;
  }

  /** Release the mic track and reset state — call on unload / teardown. */
  function destroyRecorder(): void {
    if (_maxRecordTimer !== null) {
      clearTimeout(_maxRecordTimer);
      _maxRecordTimer = null;
    }
    if (_mediaRecorder && _mediaRecorder.state !== 'inactive') {
      try {
        _mediaRecorder.stop();
      } catch (err) {
        console.debug('[voice] recorder.stop on destroy:', err);
      }
    }
    if (_mediaRecorder?.stream) {
      _mediaRecorder.stream.getTracks().forEach((t) => t.stop());
    }
    _mediaRecorder = null;
    recorderState.value = 'idle';
    void _wakeLock?.release();
  }

  return {
    available,
    loading,
    checkAvailability,
    recorderState,
    toggleRecording,
    destroyRecorder,
    MAX_RECORD_MS,
  };
});

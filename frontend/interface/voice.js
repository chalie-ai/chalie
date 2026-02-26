/**
 * Voice I/O — Mic recording (STT) + TTS playback.
 *
 * Discovers the local voice service via /voice/health on init.
 * No configuration needed — voice is available when the service is running,
 * hidden when it's not.
 */

const _TTS_PATH = '/voice/synthesize';
const _STT_PATH = '/voice/transcribe';

export class VoiceIO {
  /**
   * @param {object} els — DOM elements
   * @param {HTMLElement} els.micBtn
   * @param {HTMLElement} els.recordingOverlay
   * @param {HTMLElement} els.stopRecordingBtn
   * @param {HTMLElement} els.audioPlayer
   * @param {HTMLElement} els.audioPlayPause
   * @param {HTMLElement} els.audioTime
   * @param {HTMLElement} els.audioClose
   * @param {HTMLElement} els.micError
   */
  constructor(els) {
    this._els = els;
    this._mediaRecorder = null;
    this._audioChunks = [];
    this._isRecording = false;
    this._currentAudio = null;
    this._micDisabled = false;
    this._speaking = false;
    this._available = false;

    this._bindEvents();
  }

  /**
   * Check if the voice service is running.
   * @returns {Promise<{tts: boolean, stt: boolean}>}
   */
  async init() {
    try {
      const res = await fetch('/voice/health', { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        this._available = true;
        return { tts: true, stt: true };
      }
    } catch (_) {
      // Voice service not available — graceful degradation
    }
    this._available = false;
    return { tts: false, stt: false };
  }

  // ---------------------------------------------------------------------------
  // Recording (STT)
  // ---------------------------------------------------------------------------

  async startRecording() {
    if (!this._available) return;
    if (this._isRecording || this._micDisabled) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this._handleMicError(new Error('getUserMedia not available — requires HTTPS'));
      return;
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      this._handleMicError(err);
      return;
    }

    this._audioChunks = [];

    // No mimeType constraint — we always convert to WAV before upload so the
    // browser's default (webm on Chrome, mp4 on iOS Safari) doesn't matter.
    this._mediaRecorder = new MediaRecorder(stream);

    this._mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) this._audioChunks.push(e.data);
    };

    this._mediaRecorder.start(250); // collect chunks every 250ms
    this._isRecording = true;
    this._els.micBtn.classList.add('recording');
    this._els.recordingOverlay.classList.remove('hidden');
  }

  /**
   * Stop recording and transcribe.
   * @returns {Promise<string|null>} transcribed text or null
   */
  async stopRecording() {
    if (!this._isRecording || !this._mediaRecorder) return null;

    const recorder = this._mediaRecorder;

    return new Promise((resolve) => {
      recorder.onstop = () => {
        recorder.stream.getTracks().forEach(t => t.stop());
        this._isRecording = false;
        this._els.micBtn.classList.remove('recording');
        this._els.recordingOverlay.classList.add('hidden');

        // iOS Safari can fire onstop before the final ondataavailable chunk.
        // A small defer lets the event queue flush first.
        setTimeout(async () => {
          if (this._audioChunks.length === 0) {
            resolve(null);
            return;
          }

          try {
            const mimeType = recorder.mimeType || 'audio/webm';
            // Convert to WAV via AudioContext — works regardless of whether the
            // browser recorded webm (Chrome) or mp4 (iOS Safari).
            const wav = await this._convertToWav(this._audioChunks, mimeType);
            const text = await this._transcribe(wav, 'recording.wav');
            resolve(text);
          } catch (err) {
            console.error('STT error:', err);
            resolve(null);
          }
        }, 100);
      };

      // Flush any buffered audio before stopping (helps on iOS Safari)
      try { recorder.requestData(); } catch (_) { /* not all browsers support it */ }
      recorder.stop();
    });
  }

  // ---------------------------------------------------------------------------
  // TTS Playback
  // ---------------------------------------------------------------------------

  /**
   * Speak text via TTS endpoint.
   * Must be called from a user gesture context (click handler).
   * @param {string} text
   */
  async speak(text) {
    if (!this._available) return;
    if (this._speaking) return;
    this._speaking = true;

    // Stop any current playback
    this._stopAudio();

    try {
      const response = await fetch(_TTS_PATH, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) throw new Error(`TTS error: ${response.status}`);

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);

      this._currentAudio = audio;
      this._els.audioPlayer.classList.remove('hidden');
      this._updatePlayPauseIcon(false);

      audio.addEventListener('timeupdate', () => {
        const m = Math.floor(audio.currentTime / 60);
        const s = Math.floor(audio.currentTime % 60);
        this._els.audioTime.textContent = `${m}:${s < 10 ? '0' : ''}${s}`;
      });

      audio.addEventListener('ended', () => {
        this._updatePlayPauseIcon(true);
        this._speaking = false;
      });

      audio.play();
      this._updatePlayPauseIcon(false);
    } catch (err) {
      console.error('TTS error:', err);
      this._speaking = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  _bindEvents() {
    // Note: stopRecordingBtn click is handled by app.js for transcribing UI flow

    // Audio player controls
    this._els.audioPlayPause.addEventListener('click', () => {
      if (!this._currentAudio) return;
      if (this._currentAudio.paused) {
        this._currentAudio.play();
        this._updatePlayPauseIcon(false);
      } else {
        this._currentAudio.pause();
        this._updatePlayPauseIcon(true);
      }
    });

    this._els.audioClose.addEventListener('click', () => {
      this._stopAudio();
    });

    // Listen for chalie:speak custom events
    document.addEventListener('chalie:speak', (e) => {
      this.speak(e.detail.text);
    });
  }

  _stopAudio() {
    if (this._currentAudio) {
      this._currentAudio.pause();
      if (this._currentAudio.src) {
        URL.revokeObjectURL(this._currentAudio.src);
      }
      this._currentAudio = null;
    }
    this._speaking = false;
    this._els.audioPlayer.classList.add('hidden');
    this._els.audioTime.textContent = '0:00';
  }

  _updatePlayPauseIcon(showPlay) {
    const btn = this._els.audioPlayPause;
    if (showPlay) {
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="5 3 19 12 5 21 5 3"></polygon>
      </svg>`;
    } else {
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="6" y="4" width="4" height="16"></rect>
        <rect x="14" y="4" width="4" height="16"></rect>
      </svg>`;
    }
  }

  _handleMicError(err) {
    console.error('Mic error:', err);
    this._micDisabled = true;
    this._els.micBtn.classList.add('disabled');
    this._els.micError.classList.remove('hidden');
  }

  /**
   * Decode audio chunks (any format) and re-encode as PCM WAV.
   * AudioContext.decodeAudioData handles webm, mp4, ogg — whatever the browser recorded.
   */
  async _convertToWav(chunks, mimeType) {
    const blob = new Blob(chunks, { type: mimeType });
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    audioCtx.close();
    return this._audioBufferToWav(audioBuffer);
  }

  _audioBufferToWav(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const numSamples = audioBuffer.length;
    const dataLength = numSamples * numChannels * 2; // 16-bit = 2 bytes/sample
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    const writeStr = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    // RIFF/WAVE header
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeStr(8, 'WAVE');
    // fmt chunk
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);                          // chunk size
    view.setUint16(20, 1, true);                           // PCM format
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true); // byte rate
    view.setUint16(32, numChannels * 2, true);             // block align
    view.setUint16(34, 16, true);                          // bits per sample
    // data chunk
    writeStr(36, 'data');
    view.setUint32(40, dataLength, true);

    // Interleave channels and write 16-bit PCM samples
    const channels = Array.from({ length: numChannels }, (_, i) => audioBuffer.getChannelData(i));
    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
      for (let c = 0; c < numChannels; c++) {
        const s = Math.max(-1, Math.min(1, channels[c][i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
      }
    }

    return new Blob([buffer], { type: 'audio/wav' });
  }

  async _transcribe(blob, filename = 'recording.wav') {
    const formData = new FormData();
    formData.append('file', blob, filename);

    const response = await fetch(_STT_PATH, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error(`STT error: ${response.status}`);

    const data = await response.json();
    return data.text || null;
  }
}

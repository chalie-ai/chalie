/**
 * VoiceRecorder — mic button + MediaRecorder → WAV → STT upload.
 *
 * Manages the #voiceRecBtn lifecycle:
 *   idle     → click → recording (red pulse)
 *   recording → click → uploading (spinner) → calls onTranscript(text)
 *
 * Mic permission error: shows inline error label below the chatbox,
 * disables the button.
 */

const _STT_PATH = '/voice/transcribe';
const _MAX_RECORD_MS = 10 * 60 * 1000; // 10 minutes

export class VoiceRecorder {
  /**
   * @param {{
   *   getHost: () => string,
   *   onTranscript: (text: string) => void,
   * }} opts
   */
  constructor({ getHost, onTranscript }) {
    this._getHost = getHost;
    this._onTranscript = onTranscript;

    this._btn = null;
    this._errorLabel = null;
    this._mediaRecorder = null;
    this._audioChunks = [];
    this._isRecording = false;
    this._maxRecordTimer = null;
  }

  /** Bind the mic button and error label after DOM is ready. */
  init() {
    this._btn = document.getElementById('voiceRecBtn');
    this._errorLabel = document.getElementById('voiceRecError');
    if (!this._btn) return;

    this._btn.addEventListener('click', () => this._handleClick());
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  async _handleClick() {
    if (this._isRecording) {
      await this._stopAndUpload();
    } else {
      await this._startRecording();
    }
  }

  async _startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this._showError('Microphone not available (requires HTTPS)');
      return;
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        this._showError('Microphone access denied');
      } else {
        this._showError('Could not start recording');
      }
      this._btn.disabled = true;
      return;
    }

    this._hideError();
    this._audioChunks = [];

    // No mimeType constraint — we always convert to WAV via AudioContext
    // before upload so the browser's default (webm/Chrome, mp4/iOS) is fine.
    this._mediaRecorder = new MediaRecorder(stream);
    this._mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) this._audioChunks.push(e.data);
    };

    this._mediaRecorder.start(250);
    this._isRecording = true;
    this._setButtonState('recording');

    // Auto-stop at 10-minute limit
    this._maxRecordTimer = setTimeout(() => {
      if (this._isRecording) this._stopAndUpload();
    }, _MAX_RECORD_MS);
  }

  async _stopAndUpload() {
    if (!this._isRecording || !this._mediaRecorder) return;

    clearTimeout(this._maxRecordTimer);
    this._maxRecordTimer = null;

    const recorder = this._mediaRecorder;
    this._isRecording = false;
    this._setButtonState('uploading');

    const chunks = await new Promise((resolve) => {
      recorder.onstop = () => {
        recorder.stream.getTracks().forEach(t => t.stop());
        // iOS Safari fires onstop before the last ondataavailable chunk;
        // a small defer lets the event queue flush first.
        setTimeout(() => resolve([...this._audioChunks]), 100);
      };
      try { recorder.requestData(); } catch (_) { /* not all browsers support it */ }
      recorder.stop();
    });

    if (chunks.length === 0) {
      this._setButtonState('idle');
      return;
    }

    try {
      const mimeType = recorder.mimeType || 'audio/webm';
      const wav = await this._convertToWav(chunks, mimeType);
      const text = await this._transcribe(wav);
      if (text) this._onTranscript(text);
    } catch (err) {
      console.error('[VoiceRecorder] STT error:', err);
      this._showError('Transcription failed');
    } finally {
      this._setButtonState('idle');
    }
  }

  _setButtonState(state) {
    if (!this._btn) return;
    this._btn.dataset.state = state;
    this._btn.setAttribute('aria-label',
      state === 'recording' ? 'Stop recording'
        : state === 'uploading' ? 'Uploading...'
          : 'Record voice message'
    );
  }

  _showError(msg) {
    if (this._errorLabel) {
      this._errorLabel.textContent = msg;
      this._errorLabel.classList.remove('hidden');
    }
  }

  _hideError() {
    if (this._errorLabel) {
      this._errorLabel.textContent = '';
      this._errorLabel.classList.add('hidden');
    }
  }

  _buildUrl(path) {
    const host = this._getHost?.();
    return host ? host.replace(/\/$/, '') + path : path;
  }

  /**
   * Decode audio chunks (any format) and re-encode as 16-bit PCM WAV.
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

  /**
   * Stop any active recording and release the mic track.
   * Called on page navigation via beforeunload.
   */
  destroy() {
    clearTimeout(this._maxRecordTimer);
    this._maxRecordTimer = null;
    if (this._mediaRecorder && this._mediaRecorder.state !== 'inactive') {
      try { this._mediaRecorder.stop(); } catch (_) {}
    }
    if (this._mediaRecorder?.stream) {
      this._mediaRecorder.stream.getTracks().forEach(t => t.stop());
    }
    this._mediaRecorder = null;
    this._isRecording = false;
    this._setButtonState('idle');
  }

  async _transcribe(blob) {
    const formData = new FormData();
    formData.append('file', blob, 'recording.wav');

    const response = await fetch(this._buildUrl(_STT_PATH), {
      method: 'POST',
      body: formData,
      credentials: 'same-origin',
    });

    if (!response.ok) throw new Error(`STT error: ${response.status}`);
    const data = await response.json();
    return data.text || null;
  }
}

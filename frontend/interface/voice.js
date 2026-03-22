/**
 * Voice I/O — Mic recording (STT) + TTS playback.
 *
 * Discovers the local voice service via /voice/health on init.
 * No configuration needed — voice is available when the service is running,
 * hidden when it's not.
 */
import { lsSet, showToast } from './utils.js';

const _TTS_PATH = '/voice/synthesize';
const _STT_PATH = '/voice/transcribe';
const _MAX_RECORD_MS = 10 * 60 * 1000; // 10 minutes

export class VoiceIO {
  /**
   * @param {() => string} [getHost] — returns the current backend host
   */
  constructor(getHost) {
    this._getHost = getHost;
    this._mediaRecorder = null;
    this._audioChunks = [];
    this._isRecording = false;
    this._currentAudio = null;
    this._speaking = false;
    this._available = false;
    this._audioCtx = null;
    this._analyser = null;
    this._analyserData = null;
    this._vizTarget = null;
    this._vizRaf = null;
  }

  /**
   * Unlock browser autoplay by creating/resuming an AudioContext + AnalyserNode.
   * Must be called from a user gesture (click/tap) before TTS playback.
   */
  unlockAudio() {
    if (!this._audioCtx) {
      this._audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      this._analyser = this._audioCtx.createAnalyser();
      this._analyser.fftSize = 256;
      this._analyser.smoothingTimeConstant = 0.82;
      this._analyserData = new Uint8Array(this._analyser.frequencyBinCount);
      this._analyser.connect(this._audioCtx.destination);
    }
    if (this._audioCtx.state === 'suspended') {
      this._audioCtx.resume();
    }
  }

  /**
   * Start the audio-reactive visualizer loop.
   * Sets --orb-energy (0-1) and --orb-bass (0-1) on the target element each frame.
   * @param {HTMLElement} orbEl — the orb element to drive
   */
  startVisualizer(orbEl) {
    this._vizTarget = orbEl;
    if (!this._vizRaf) this._vizLoop();
  }

  /** Stop the visualizer loop and reset custom properties. */
  stopVisualizer() {
    this._vizTarget = null;
    if (this._vizRaf) {
      cancelAnimationFrame(this._vizRaf);
      this._vizRaf = null;
    }
  }

  /** @private RAF loop — reads analyser data, sets CSS custom properties. */
  _vizLoop() {
    if (!this._vizTarget || !this._analyser) {
      this._vizRaf = null;
      return;
    }

    this._analyser.getByteFrequencyData(this._analyserData);

    const data = this._analyserData;
    const len = data.length;
    const bassEnd = Math.floor(len * 0.15);
    const midEnd = Math.floor(len * 0.5);

    let bass = 0, mid = 0, treble = 0;
    for (let i = 0; i < bassEnd; i++) bass += data[i];
    for (let i = bassEnd; i < midEnd; i++) mid += data[i];
    for (let i = midEnd; i < len; i++) treble += data[i];

    bass = bass / (bassEnd * 255);
    mid = mid / ((midEnd - bassEnd) * 255);
    treble = treble / ((len - midEnd) * 255);

    const energy = bass * 0.5 + mid * 0.35 + treble * 0.15;

    this._vizTarget.style.setProperty('--orb-energy', energy.toFixed(3));
    this._vizTarget.style.setProperty('--orb-bass', bass.toFixed(3));

    this._vizRaf = requestAnimationFrame(() => this._vizLoop());
  }

  _buildUrl(path) {
    const host = this._getHost?.();
    return host ? host.replace(/\/$/, '') + path : path;
  }

  /**
   * Check if the voice service is running.
   * @returns {Promise<{tts: boolean, stt: boolean}>}
   */
  async init() {
    try {
      const res = await fetch(this._buildUrl('/voice/health'), { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'ok') {
          this._available = true;
          return { tts: true, stt: true };
        }
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
    if (this._isRecording) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.error('getUserMedia not available — requires HTTPS');
      return;
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      console.error('Mic error:', err);
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

    // Auto-stop at 10 minutes — recording still goes through
    this._maxRecordTimer = setTimeout(() => {
      if (this._isRecording) {
        document.dispatchEvent(new CustomEvent('chalie:recording:maxed'));
      }
    }, _MAX_RECORD_MS);
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
        if (this._maxRecordTimer) { clearTimeout(this._maxRecordTimer); this._maxRecordTimer = null; }

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
   * Kicks off background synthesis on the server; individual WAV chunks
   * arrive as WebSocket messages (tts_chunk / tts_done) and are queued
   * for progressive playback.
   * @param {string} text
   */
  async speak(text) {
    if (!this._available) return;
    if (this._speaking) return;

    // Stop any current playback and reset stream state
    this.stopAudio();

    // Set _speaking AFTER stopAudio() (which resets it to false)
    this._speaking = true;
    this._chunkQueue = [];
    this._chunkTexts = [];
    this._chunkIdx = 0;
    this._streamDone = false;
    this._cumulativeTime = 0;

    try {
      const response = await fetch(this._buildUrl(_TTS_PATH), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) throw new Error(`TTS error: ${response.status}`);
      // Chunks will arrive via WebSocket — nothing more to do here
    } catch (err) {
      this._speaking = false;
      document.dispatchEvent(new CustomEvent('chalie:speak:error', { detail: { err } }));
    }
  }

  /**
   * Handle a TTS chunk pushed via WebSocket.
   * @param {{audio: string, index: number, total: number, text?: string}} data
   */
  handleTtsChunk(data) {
    if (!this._speaking) return;

    const raw = atob(data.audio);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

    const blob = new Blob([bytes], { type: 'audio/wav' });
    this._chunkQueue.push(URL.createObjectURL(blob));
    this._chunkTexts.push(data.text || '');

    // Start playback as soon as first chunk arrives
    if (this._chunkQueue.length === 1) this._playNextChunk();
  }

  /**
   * Handle TTS stream completion pushed via WebSocket.
   */
  handleTtsDone() {
    this._streamDone = true;
    this._checkStreamDone();
  }

  // ---------------------------------------------------------------------------
  // Chunk Navigation
  // ---------------------------------------------------------------------------

  /** Skip to next chunk. */
  skipForward() {
    if (!this._speaking || !this._currentAudio) return;
    this._cumulativeTime += this._currentAudio.currentTime;
    this._currentAudio.pause();
    URL.revokeObjectURL(this._chunkQueue[this._chunkIdx]);
    if (this._currentAudio.parentNode) this._currentAudio.parentNode.removeChild(this._currentAudio);
    this._chunkIdx++;
    this._playNextChunk();
  }

  /** Restart current chunk, or go to previous if near the start. */
  skipBack() {
    if (!this._speaking || !this._currentAudio) return;

    // If more than 2s in, just restart current chunk
    if (this._currentAudio.currentTime > 2) {
      this._currentAudio.currentTime = 0;
      return;
    }

    // Otherwise go to previous chunk
    if (this._chunkIdx > 0) {
      this._currentAudio.pause();
      if (this._currentAudio.parentNode) this._currentAudio.parentNode.removeChild(this._currentAudio);
      this._chunkIdx--;
      this._cumulativeTime = 0;
      this._playNextChunk();
    }
  }

  /** Toggle pause/resume on current chunk. Returns true if now playing. */
  togglePause() {
    if (!this._speaking || !this._currentAudio) return false;
    if (this._currentAudio.paused) {
      this._currentAudio.play();
      return true;
    } else {
      this._currentAudio.pause();
      return false;
    }
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  /** Play the next queued WAV chunk. */
  _playNextChunk() {
    if (this._chunkIdx >= this._chunkQueue.length) {
      this._checkStreamDone();
      return;
    }

    const url = this._chunkQueue[this._chunkIdx];
    const text = this._chunkTexts[this._chunkIdx] || '';
    const audio = new Audio(url);
    // crossOrigin not needed for blob URLs but harmless
    document.body.appendChild(audio);
    this._currentAudio = audio;

    // Route through AnalyserNode for visualizer
    if (this._audioCtx && this._analyser) {
      try {
        const source = this._audioCtx.createMediaElementSource(audio);
        source.connect(this._analyser);
      } catch (_) { /* already connected or no context */ }
    }

    document.dispatchEvent(new CustomEvent('chalie:speak:chunk', { detail: { text, index: this._chunkIdx } }));

    audio.addEventListener('ended', () => {
      this._cumulativeTime += audio.duration;
      URL.revokeObjectURL(url);
      if (audio.parentNode) audio.parentNode.removeChild(audio);
      this._chunkIdx++;
      this._playNextChunk();
    });

    audio.play().catch(err => {
      this._speaking = false;
      document.dispatchEvent(new CustomEvent('chalie:speak:error', { detail: { err } }));
    });
  }

  /** Fire the done event once the stream is exhausted and all chunks played. */
  _checkStreamDone() {
    if (this._streamDone && this._chunkIdx >= this._chunkQueue.length) {
      this._speaking = false;
      document.dispatchEvent(new CustomEvent('chalie:speak:done'));
    }
  }

  stopAudio() {
    if (this._currentAudio) {
      this._currentAudio.pause();
      if (this._currentAudio.src) URL.revokeObjectURL(this._currentAudio.src);
      if (this._currentAudio.parentNode) this._currentAudio.parentNode.removeChild(this._currentAudio);
      this._currentAudio = null;
    }

    // Release any remaining queued blob URLs
    if (this._chunkQueue) {
      for (let i = (this._chunkIdx || 0) + 1; i < this._chunkQueue.length; i++) {
        URL.revokeObjectURL(this._chunkQueue[i]);
      }
    }
    this._chunkQueue = [];
    this._chunkIdx = 0;
    this._streamDone = false;
    this._cumulativeTime = 0;
    this._speaking = false;
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

    const response = await fetch(this._buildUrl(_STT_PATH), {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error(`STT error: ${response.status}`);

    const data = await response.json();
    return data.text || null;
  }
}


/**
 * Voice Mode — full-screen orb overlay, state machine, and media controls.
 *
 * Owns the UI layer that sits on top of VoiceIO. app.js creates this and
 * wires the callbacks for presence, rendering, and WebSocket send.
 */
export class VoiceMode {
  /**
   * @param {{ voice: VoiceIO }} deps
   */
  constructor({ voice }) {
    this._voice = voice;
    this._active = false;
    this._hasPlayback = false;
    this._lastVoiceText = null;

    // Callbacks set by app.js
    this._onSendMessage = null;    // (text) => void — send voice message
    this._onModeEnter = null;      // () => void
    this._onModeExit = null;       // () => void
  }

  /** Register callback fired when a voice message should be sent. */
  onSendMessage(cb) { this._onSendMessage = cb; }

  /** Register callback when voice mode is entered. */
  onModeEnter(cb) { this._onModeEnter = cb; }

  /** Register callback when voice mode is exited. */
  onModeExit(cb) { this._onModeExit = cb; }

  /** Whether voice mode is currently active. */
  get active() { return this._active; }

  /**
   * Bind all voice mode DOM events.  Must be called after DOM ready.
   */
  init() {
    document.getElementById('voiceModeBtn')?.addEventListener('click', () => {
      if (this._active) this.exitMode();
      else this.enterMode();
    });

    document.getElementById('voiceModeExit')?.addEventListener('click', () => {
      this.exitMode();
    });

    // Orb click — state machine
    document.getElementById('voiceOrb')?.addEventListener('click', async () => {
      // Unlock autoplay on every user gesture so TTS can play later
      this._voice.unlockAudio();

      const orb = document.getElementById('voiceOrb');
      const state = orb?.dataset.state;

      if (state === 'idle') {
        this._lastVoiceText = null;
        const sub = document.getElementById('voiceSubtitle');
        if (sub) { sub.textContent = ''; sub.classList.add('hidden'); }
        await this._voice.startRecording();
        if (this._voice._isRecording) this._setOrbState('listening');
      } else if (state === 'listening') {
        this._setOrbState('thinking');
        const text = await this._voice.stopRecording();
        if (text) {
          this._onSendMessage?.(text);
        } else {
          this._setOrbState('idle');
        }
      }
    });

    // Auto-stop at 10-minute limit
    document.addEventListener('chalie:recording:maxed', async () => {
      if (!this._active) return;
      showToast('10-min max per voice message');
      this._setOrbState('thinking');
      const text = await this._voice.stopRecording();
      if (text) {
        this._onSendMessage?.(text);
      } else {
        this._setOrbState('idle');
      }
    });

    // Subtitle: show current sentence during playback
    document.addEventListener('chalie:speak:chunk', (e) => {
      if (!this._active) return;
      const el = document.getElementById('voiceSubtitle');
      if (el) {
        el.textContent = e.detail.text;
        el.classList.remove('hidden');
      }
    });

    // TTS finished — back to idle (controls stay visible for re-listen)
    document.addEventListener('chalie:speak:done', () => {
      if (this._active) {
        this._setOrbState('idle');
        this._setPlayPauseIcon(false);
      }
    });
    document.addEventListener('chalie:speak:error', () => {
      if (this._active) {
        this._hasPlayback = false;
        this._setOrbState('idle');
        const sub = document.getElementById('voiceSubtitle');
        if (sub) { sub.textContent = ''; sub.classList.add('hidden'); }
      }
    });

    // Media controls
    document.getElementById('voiceCtrlForward')?.addEventListener('click', () => this._voice.skipForward());
    document.getElementById('voiceCtrlBack')?.addEventListener('click', () => this._voice.skipBack());
    document.getElementById('voiceCtrlPause')?.addEventListener('click', () => {
      this._voice.unlockAudio();

      // If playback finished and user presses play — replay from start
      if (!this._voice._speaking && this._hasPlayback && this._lastVoiceText) {
        this._setOrbState('speaking');
        this._setPlayPauseIcon(true);
        this._voice.speak(this._lastVoiceText);
        return;
      }

      const playing = this._voice.togglePause();
      this._setPlayPauseIcon(playing);
    });

    // Speak button on chat messages — enter voice mode and play
    document.addEventListener('chalie:speak-message', (e) => {
      const text = e.detail?.text;
      if (!text || !this._voice._available) return;
      this._voice.unlockAudio();
      this.enterMode();
      this._lastVoiceText = text;
      this._setOrbState('speaking');
      this._setPlayPauseIcon(true);
      this._voice.speak(text);
    });
  }

  enterMode() {
    if (!this._voice._available) return;
    this._active = true;
    lsSet('chalie_voice_mode', '1');

    document.getElementById('voiceModeOverlay')?.classList.remove('hidden');
    document.getElementById('conversationSpine')?.classList.add('hidden');
    document.querySelector('.input-dock')?.classList.add('hidden');
    document.getElementById('taskStrip')?.classList.add('hidden');
    document.getElementById('voiceModeBtn')?.classList.add('active');

    this._setOrbState('idle');
    this._onModeEnter?.();
  }

  exitMode() {
    this._active = false;
    lsSet('chalie_voice_mode', '');

    // Stop any active recording or playback
    if (this._voice._isRecording) this._voice.stopRecording();
    this._voice.stopAudio();

    document.getElementById('voiceModeOverlay')?.classList.add('hidden');
    document.getElementById('conversationSpine')?.classList.remove('hidden');
    document.querySelector('.input-dock')?.classList.remove('hidden');
    document.getElementById('voiceModeBtn')?.classList.remove('active');

    this._onModeExit?.();
  }

  /**
   * Called by app.js when a voice message response arrives with TTS text.
   * Sets the orb to speaking state and starts playback.
   */
  startSpeaking(text) {
    this._lastVoiceText = text;
    this._setOrbState('speaking');
    this._setPlayPauseIcon(true);
    this._voice.speak(text);
  }

  /** Set orb to error/idle after send failure. */
  setOrbIdle() {
    this._setOrbState('idle');
  }

  /** Set orb to "thinking" state. */
  setOrbThinking() {
    this._setOrbState('thinking');
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  _setOrbState(state) {
    const orb = document.getElementById('voiceOrb');
    const controls = document.getElementById('voiceModeControls');
    const label = document.getElementById('voiceModeLabel');

    if (orb) orb.dataset.state = state;
    if (label) {
      const labels = { idle: '', listening: 'Listening...', thinking: 'Thinking...', speaking: '' };
      label.textContent = labels[state] || '';
    }

    if (controls) {
      if (state === 'speaking') {
        this._hasPlayback = true;
        controls.classList.remove('hidden');
      } else if (state === 'listening' || state === 'thinking') {
        this._hasPlayback = false;
        controls.classList.add('hidden');
      }
    }

    // Audio-reactive visualizer
    if (state === 'speaking' && orb) {
      this._voice.startVisualizer(orb);
    } else {
      this._voice.stopVisualizer();
      if (orb) {
        orb.style.removeProperty('--orb-energy');
        orb.style.removeProperty('--orb-bass');
      }
    }
  }

  _setPlayPauseIcon(playing) {
    const btn = document.getElementById('voiceCtrlPause');
    if (!btn) return;
    btn.innerHTML = playing
      ? '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>'
      : '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
  }
}

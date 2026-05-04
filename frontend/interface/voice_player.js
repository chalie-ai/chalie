/**
 * VoicePlayer — per-message streaming audio overlay.
 *
 * Listens for `chalie:speak-message` from renderer.js speaker buttons.
 * On each event it POSTs /voice/synthesize and reads the response body
 * as an NDJSON stream — one `{index, total, audio}` line per WAV chunk.
 * Each chunk is decoded into an AudioBuffer and queued; buffers play in
 * order by chaining source.onended → _playNextChunk.
 *
 * iOS Safari autoplay: AudioContext.resume() runs synchronously inside
 * the click gesture (via the `chalie:speak-message` handler), satisfying
 * the user-activation requirement.
 *
 * Only one session plays at a time — opening a new message supersedes
 * the previous stream via `_openGeneration`.
 */

import { createWakeLock } from './utils.js';

const _TTS_PATH = '/voice/synthesize';
const _SKIP_SECONDS = 10;

export class VoicePlayer {
  /**
   * @param {{ getHost: () => string }} opts
   */
  constructor({ getHost }) {
    this._getHost = getHost;

    this._overlay = null;
    this._playBtn = null;
    this._bkBtn = null;
    this._ffBtn = null;
    this._closeBtn = null;
    this._progress = null;
    this._timeEl = null;
    this._errorEl = null;
    this._loadingEl = null;
    this._controlsEl = null;

    this._boundKeydown = null;

    // Generation counter so a stale stream from a superseded session is
    // ignored. Incremented on every _open() and _close().
    this._openGeneration = 0;
    this._fetchAbort = null;

    // AudioContext — created lazily on first open, reused after.
    this._audioCtx = null;

    // Per-session playback state
    this._bufferQueue = [];
    this._chunkIdx = 0;
    this._streamDone = false;
    this._currentSource = null;
    this._currentBuffer = null;
    this._sourceStartCtxTime = 0;
    this._sourceStartOffset = 0;
    this._paused = false;
    this._pausedOffset = 0;
    this._progressTimer = null;
    this._wakeLock = createWakeLock();
  }

  /** Bind overlay DOM + listeners. Must be called after DOM ready. */
  init() {
    this._overlay = document.getElementById('voicePlayerOverlay');
    if (!this._overlay) return;

    this._playBtn = this._overlay.querySelector('#vpPlayBtn');
    this._bkBtn = this._overlay.querySelector('#vpBkBtn');
    this._ffBtn = this._overlay.querySelector('#vpFfBtn');
    this._closeBtn = this._overlay.querySelector('#vpCloseBtn');
    this._progress = this._overlay.querySelector('#vpProgress');
    this._timeEl = this._overlay.querySelector('#vpTime');
    this._errorEl = this._overlay.querySelector('#vpError');
    this._loadingEl = this._overlay.querySelector('#vpLoading');
    this._controlsEl = this._overlay.querySelector('.voice-player__controls');

    this._playBtn?.addEventListener('click', () => this._togglePlayPause());
    this._bkBtn?.addEventListener('click', () => this._skipBack());
    this._ffBtn?.addEventListener('click', () => this._skipForward());
    this._closeBtn?.addEventListener('click', () => this._close());

    this._overlay.addEventListener('click', (e) => {
      if (e.target === this._overlay) this._close();
    });

    // Scrubber seeks within the current chunk only — cross-chunk seeking
    // isn't worth the complexity for a speaker-button UX.
    this._progress?.addEventListener('input', () => {
      if (!this._currentBuffer) return;
      const offset = Number.parseFloat(this._progress.value) || 0;
      this._stopCurrentSource();
      this._playBufferFrom(this._currentBuffer, offset);
    });

    // Speaker-button click fires here; handler runs inside the user gesture
    // so AudioContext.resume() is permitted on iOS Safari.
    document.addEventListener('chalie:speak-message', (e) => {
      const { text } = e.detail || {};
      if (text) this._open(text);
    });
  }

  // ---------------------------------------------------------------------------
  // Private — session lifecycle
  // ---------------------------------------------------------------------------

  async _open(text) {
    this._openGeneration += 1;
    const gen = this._openGeneration;

    if (this._fetchAbort) this._fetchAbort.abort();
    this._fetchAbort = new AbortController();

    this._resetPlayback();

    if (!this._audioCtx) {
      const Ctor = window.AudioContext || window.webkitAudioContext;
      this._audioCtx = new Ctor();
    }
    if (this._audioCtx.state === 'suspended') {
      this._audioCtx.resume().catch(() => {});
    }

    this._showOverlay();
    this._showLoading(true);
    this._showError(null);
    this._bindKeyboard();

    try {
      const host = this._getHost?.() || '';
      const url = host.replace(/\/$/, '') + _TTS_PATH;

      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ text }),
        signal: this._fetchAbort.signal,
      });

      if (gen !== this._openGeneration) return;
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        if (gen !== this._openGeneration) return;
        throw new Error(err.error || `HTTP ${resp.status}`);
      }

      const sawDone = await this._consumeStream(resp.body, gen);

      if (gen !== this._openGeneration) return;
      if (!sawDone) {
        this._showLoading(false);
        this._showError('Audio was cut short — try again');
        return;
      }
      this._streamDone = true;
      this._checkStreamDone();
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (gen !== this._openGeneration) return;
      this._showLoading(false);
      this._showError(err.message || 'Failed to synthesize');
      console.error('[VoicePlayer] synthesize error:', err);
    }
  }

  /**
   * Read `body` as an NDJSON stream of `{index, total, audio}` chunks plus
   * a final `{done: true, total}` sentinel. Decode each WAV inline, push
   * into the playback queue, and kick playback off when the first buffer
   * lands. Returns `true` only if the sentinel was received; `false`
   * means the server closed the stream early (client should show an
   * error rather than treat partial audio as success).
   *
   * @param {ReadableStream<Uint8Array>} body
   * @param {number} gen — generation captured at _open() time
   * @returns {Promise<boolean>}
   */
  async _consumeStream(body, gen) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let sawDone = false;

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (gen !== this._openGeneration) { reader.cancel().catch(() => {}); return false; }
        if (done) break;

        buf += decoder.decode(value, { stream: true });

        let nl;
        while ((nl = buf.indexOf('\n')) !== -1) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          if (await this._enqueueLine(line, gen)) sawDone = true;
          if (gen !== this._openGeneration) { reader.cancel().catch(() => {}); return false; }
        }
      }

      const tail = buf.trim();
      if (tail && await this._enqueueLine(tail, gen)) sawDone = true;
    } finally {
      try { reader.releaseLock(); } catch { /* already released */ }
    }
    return sawDone;
  }

  /**
   * Parse one NDJSON line. Returns `true` iff this line was the
   * `{done: true}` sentinel.
   */
  async _enqueueLine(line, gen) {
    let payload;
    try {
      payload = JSON.parse(line);
    } catch {
      console.warn('[VoicePlayer] dropping non-JSON line:', line);
      return false;
    }
    if (payload?.done) return true;

    const audioB64 = payload?.audio;
    if (!audioB64) return false;

    const raw = atob(audioB64);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

    let buffer;
    try {
      buffer = await this._decodeAudio(bytes.buffer);
    } catch (err) {
      // Malformed WAV or AudioContext gone — abort the fetch so the
      // server stops synthesising chunks the client will never play.
      this._fetchAbort?.abort();
      throw err;
    }
    if (gen !== this._openGeneration) return false;

    this._bufferQueue.push(buffer);

    if (!this._currentSource && !this._paused
        && this._chunkIdx < this._bufferQueue.length) {
      this._showLoading(false);
      this._playNextChunk();
    }
    return false;
  }

  /** Promise wrapper around AudioContext.decodeAudioData (callback form for iOS). */
  _decodeAudio(arrayBuffer) {
    return new Promise((resolve, reject) => {
      if (!this._audioCtx) { reject(new Error('AudioContext gone')); return; }
      this._audioCtx.decodeAudioData(arrayBuffer, resolve, reject);
    });
  }

  _close() {
    this._openGeneration += 1; // discard any in-flight stream
    if (this._fetchAbort) this._fetchAbort.abort();
    this._resetPlayback();
    if (this._overlay?.open) this._overlay.close();
    this._unbindKeyboard();
  }

  _resetPlayback() {
    this._stopCurrentSource();
    this._stopProgressTimer();
    this._bufferQueue = [];
    this._chunkIdx = 0;
    this._streamDone = false;
    this._currentBuffer = null;
    this._sourceStartCtxTime = 0;
    this._sourceStartOffset = 0;
    this._paused = false;
    this._pausedOffset = 0;
    this._setPlayIcon(false);
    if (this._progress) { this._progress.max = 0; this._progress.value = 0; }
    if (this._timeEl) this._timeEl.textContent = '0:00 / 0:00';
    this._showLoading(false);
    this._wakeLock.release();
  }

  _checkStreamDone() {
    if (this._streamDone
        && this._chunkIdx >= this._bufferQueue.length
        && !this._currentSource) {
      this._setPlayIcon(false);
      if (this._progress && this._currentBuffer) {
        this._progress.value = this._currentBuffer.duration || 0;
      }
      this._wakeLock.release();
    }
  }

  // ---------------------------------------------------------------------------
  // Private — playback
  // ---------------------------------------------------------------------------

  _playNextChunk() {
    if (this._chunkIdx >= this._bufferQueue.length) {
      this._checkStreamDone();
      return;
    }

    const buffer = this._bufferQueue[this._chunkIdx];
    this._currentBuffer = buffer;
    this._paused = false;
    this._pausedOffset = 0;

    if (this._progress) {
      this._progress.max = buffer.duration || 0;
      this._progress.value = 0;
    }

    this._playBufferFrom(buffer, 0);
  }

  _playBufferFrom(buffer, offset) {
    if (!this._audioCtx) return;

    // Browsers may auto-suspend the AudioContext between chunks (no recent
    // user gesture). resume() transitions state synchronously for scheduling
    // purposes — queued start() calls execute once the context is running.
    if (this._audioCtx.state === 'suspended') {
      this._audioCtx.resume().catch(() => {});
    }

    const source = this._audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(this._audioCtx.destination);

    source.onended = () => {
      if (source !== this._currentSource) return;
      this._currentSource = null;
      if (this._paused) return;
      this._chunkIdx += 1;
      this._playNextChunk();
    };

    try {
      source.start(0, offset);
    } catch (err) {
      console.error('[VoicePlayer] source.start failed:', err);
      this._showError('Playback error');
      return;
    }

    this._currentSource = source;
    this._currentBuffer = buffer;
    this._sourceStartCtxTime = this._audioCtx.currentTime;
    this._sourceStartOffset = offset;
    this._paused = false;
    this._setPlayIcon(true);
    this._startProgressTimer();
    this._wakeLock.acquire();
  }

  _stopCurrentSource() {
    if (!this._currentSource) return;
    this._currentSource.onended = null;
    try { this._currentSource.stop(); } catch (e) { console.warn('[VoicePlayer] source.stop failed:', e); }
    try { this._currentSource.disconnect(); } catch (e) { console.warn('[VoicePlayer] source.disconnect failed:', e); }
    this._currentSource = null;
  }

  _currentPosition() {
    if (this._paused) return this._pausedOffset;
    if (!this._currentSource || !this._audioCtx) return 0;
    return this._sourceStartOffset + (this._audioCtx.currentTime - this._sourceStartCtxTime);
  }

  _togglePlayPause() {
    if (!this._audioCtx) return;
    if (this._paused) {
      if (!this._currentBuffer) return;
      this._playBufferFrom(this._currentBuffer, this._pausedOffset);
      return;
    }
    if (!this._currentSource) {
      // Nothing playing — likely finished. Restart from first buffer.
      if (this._bufferQueue.length) {
        this._chunkIdx = 0;
        this._playNextChunk();
      }
      return;
    }
    this._pausedOffset = this._currentPosition();
    this._stopCurrentSource();
    this._paused = true;
    this._setPlayIcon(false);
    this._stopProgressTimer();
    this._wakeLock.release();
  }

  _skipForward() {
    if (!this._currentBuffer || !this._audioCtx) return;
    const pos = this._currentPosition();
    const target = pos + _SKIP_SECONDS;
    if (target < this._currentBuffer.duration) {
      this._stopCurrentSource();
      this._playBufferFrom(this._currentBuffer, target);
      return;
    }
    this._stopCurrentSource();
    this._chunkIdx += 1;
    this._playNextChunk();
  }

  _skipBack() {
    if (!this._currentBuffer || !this._audioCtx) return;
    const pos = this._currentPosition();
    const target = pos - _SKIP_SECONDS;
    if (target > 0) {
      this._stopCurrentSource();
      this._playBufferFrom(this._currentBuffer, target);
      return;
    }
    if (pos > 2) {
      this._stopCurrentSource();
      this._playBufferFrom(this._currentBuffer, 0);
      return;
    }
    if (this._chunkIdx > 0) {
      this._stopCurrentSource();
      this._chunkIdx -= 1;
      this._playNextChunk();
    } else {
      this._stopCurrentSource();
      this._playBufferFrom(this._currentBuffer, 0);
    }
  }

  // ---------------------------------------------------------------------------
  // Private — UI
  // ---------------------------------------------------------------------------

  _showOverlay() {
    if (this._overlay && !this._overlay.open) this._overlay.show();
  }

  _showLoading(loading) {
    this._loadingEl?.classList.toggle('hidden', !loading);
    this._controlsEl?.classList.toggle('hidden', loading);
  }

  _showError(msg) {
    if (!this._errorEl) return;
    if (msg) {
      this._errorEl.textContent = msg;
      this._errorEl.classList.remove('hidden');
    } else {
      this._errorEl.textContent = '';
      this._errorEl.classList.add('hidden');
    }
  }

  _setPlayIcon(playing) {
    if (!this._playBtn) return;
    this._playBtn.innerHTML = playing
      ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
           <rect x="6" y="4" width="4" height="16"></rect>
           <rect x="14" y="4" width="4" height="16"></rect>
         </svg>`
      : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
           <polygon points="5 3 19 12 5 21 5 3"></polygon>
         </svg>`;
    this._playBtn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
  }

  _startProgressTimer() {
    this._stopProgressTimer();
    this._progressTimer = setInterval(() => this._updateProgress(), 200);
  }

  _stopProgressTimer() {
    if (this._progressTimer) {
      clearInterval(this._progressTimer);
      this._progressTimer = null;
    }
  }

  _updateProgress() {
    const buf = this._currentBuffer;
    if (!buf) return;
    const cur = this._currentPosition();
    const dur = buf.duration || 0;
    if (this._progress && !Number.isNaN(dur)) {
      this._progress.max = dur;
      this._progress.value = Math.min(cur, dur);
    }
    if (this._timeEl) {
      this._timeEl.textContent = `${this._fmt(cur)} / ${this._fmt(dur)}`;
    }
  }

  _fmt(s) {
    if (Number.isNaN(s) || !Number.isFinite(s)) return '0:00';
    const m = Math.floor(s / 60);
    const sec = String(Math.floor(s % 60)).padStart(2, '0');
    return `${m}:${sec}`;
  }

  _bindKeyboard() {
    this._unbindKeyboard();
    this._boundKeydown = (e) => {
      if (!this._overlay?.open) return;
      if (e.key === 'Escape') { this._close(); return; }
      if (e.key === ' ' || e.key === 'Space') { e.preventDefault(); this._togglePlayPause(); return; }
      if (e.key === 'ArrowRight') { this._skipForward(); return; }
      if (e.key === 'ArrowLeft') this._skipBack();
    };
    document.addEventListener('keydown', this._boundKeydown);
  }

  _unbindKeyboard() {
    if (this._boundKeydown) {
      document.removeEventListener('keydown', this._boundKeydown);
      this._boundKeydown = null;
    }
  }
}

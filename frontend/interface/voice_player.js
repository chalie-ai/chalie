/**
 * VoicePlayer — per-message audio overlay.
 *
 * Listens for `chalie:speak-message` from renderer.js speaker buttons. Each
 * event POSTs /voice/synthesize, awaits the full WAV blob, decodes it once
 * into a single AudioBuffer, and plays it via the Web Audio API. Skip/scrub
 * controls work against that single buffer.
 *
 * iOS Safari autoplay: AudioContext.resume() runs synchronously inside the
 * click handler (via the `chalie:speak-message` dispatch), satisfying the
 * user-activation requirement.
 *
 * Only one session plays at a time — opening a new message supersedes the
 * previous fetch via `_openGeneration`.
 */

import { createWakeLock } from './utils.js';

const _TTS_PATH = '/voice/synthesize';
const _SKIP_SECONDS = 10;
// Cold-start retry policy: TTS models can take 10-30s to load on first call.
// We auto-retry while the server signals `reason: loading` so the user
// doesn't have to click the speaker button repeatedly.
const _LOADING_MAX_RETRIES = 6;
const _LOADING_DEFAULT_DELAY_MS = 3000;

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

    // Generation counter so a stale fetch from a superseded session is
    // ignored. Incremented on every _open() and _close().
    this._openGeneration = 0;
    this._fetchAbort = null;

    // AudioContext — created lazily on first open, reused after.
    this._audioCtx = null;

    // Per-session playback state
    this._buffer = null;
    this._currentSource = null;
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

    this._progress?.addEventListener('input', () => {
      if (!this._buffer) return;
      const offset = Number.parseFloat(this._progress.value) || 0;
      this._stopCurrentSource();
      this._playBufferFrom(this._buffer, offset);
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
      const Ctor = globalThis.AudioContext || globalThis.webkitAudioContext;
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

      const resp = await this._fetchWithLoadingRetry(url, text, gen);
      if (gen !== this._openGeneration) return;
      if (!resp) return; // aborted or stale

      const arrayBuf = await resp.arrayBuffer();
      if (gen !== this._openGeneration) return;

      const buffer = await this._decodeAudio(arrayBuf);
      if (gen !== this._openGeneration) return;

      this._buffer = buffer;
      this._showLoading(false);
      if (this._progress) {
        this._progress.max = buffer.duration || 0;
        this._progress.value = 0;
      }
      this._playBufferFrom(buffer, 0);
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (gen !== this._openGeneration) return;
      this._showLoading(false);
      this._showError(err.message || 'Failed to synthesize');
      console.error('[VoicePlayer] synthesize error:', err);
    }
  }

  /**
   * POST text to /voice/synthesize with cold-start retry.
   *
   * The server returns 503 + `{reason: 'loading'}` while Kokoro/Moonshine
   * are still warming up. We honour `Retry-After` (or fall back to a default
   * delay) and re-issue the request until the models report ready, the user
   * supersedes the session, or we hit the retry cap.
   *
   * Returns the successful `Response`, or `null` if the session was aborted
   * or superseded mid-wait. Throws on any non-loading error so the caller's
   * catch block can surface it to the user.
   */
  async _fetchWithLoadingRetry(url, text, gen) {
    for (let attempt = 0; attempt <= _LOADING_MAX_RETRIES; attempt += 1) {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ text }),
        signal: this._fetchAbort.signal,
      });
      if (gen !== this._openGeneration) return null;
      if (resp.ok) return resp;

      const err = await resp.clone().json().catch(() => ({}));
      // deps_missing → surface the install hint verbatim. The speaker button
      // SHOULD already be hidden by the /voice/health poller in app.js, but a
      // stale page load (interface fetched before voice install completed)
      // can still hit this path.
      if (err.reason === 'deps_missing') {
        throw new Error(err.hint || err.error || 'Voice dependencies not installed');
      }
      if (err.reason === 'models_missing') {
        throw new Error(err.hint || err.error || 'Voice models not installed');
      }
      if (err.reason === 'loading' && attempt < _LOADING_MAX_RETRIES) {
        const retryAfter = Number.parseFloat(resp.headers.get('Retry-After') || '');
        const delayMs = Number.isFinite(retryAfter) && retryAfter > 0
          ? retryAfter * 1000
          : _LOADING_DEFAULT_DELAY_MS;
        await this._sleepAbortable(delayMs);
        if (gen !== this._openGeneration) return null;
        continue;
      }
      throw new Error(err.error || `HTTP ${resp.status}`);
    }
    throw new Error('Voice models still loading — please retry shortly');
  }

  /** Resolve after `ms`, or reject immediately if the active fetch aborts. */
  _sleepAbortable(ms) {
    return new Promise((resolve, reject) => {
      const signal = this._fetchAbort?.signal;
      const t = setTimeout(() => {
        signal?.removeEventListener('abort', onAbort);
        resolve();
      }, ms);
      const onAbort = () => {
        clearTimeout(t);
        const err = new Error('aborted');
        err.name = 'AbortError';
        reject(err);
      };
      if (signal?.aborted) { onAbort(); return; }
      signal?.addEventListener('abort', onAbort, { once: true });
    });
  }

  /** Promise wrapper around AudioContext.decodeAudioData (callback form for iOS). */
  _decodeAudio(arrayBuffer) {
    return new Promise((resolve, reject) => {
      if (!this._audioCtx) { reject(new Error('AudioContext gone')); return; }
      this._audioCtx.decodeAudioData(arrayBuffer, resolve, reject);
    });
  }

  _close() {
    this._openGeneration += 1; // discard any in-flight fetch
    if (this._fetchAbort) this._fetchAbort.abort();
    this._resetPlayback();
    if (this._overlay?.open) this._overlay.close();
    this._unbindKeyboard();
  }

  _resetPlayback() {
    this._stopCurrentSource();
    this._stopProgressTimer();
    this._buffer = null;
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

  // ---------------------------------------------------------------------------
  // Private — playback
  // ---------------------------------------------------------------------------

  _playBufferFrom(buffer, offset) {
    if (!this._audioCtx) return;

    // Browsers may auto-suspend the AudioContext after period of inactivity.
    // resume() transitions state synchronously for scheduling purposes —
    // queued start() calls execute once the context is running.
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
      // Natural end-of-playback: stop the timer, leave the buffer in place
      // so the user can press play to replay from the beginning.
      this._setPlayIcon(false);
      this._stopProgressTimer();
      if (this._progress && buffer) {
        this._progress.value = buffer.duration || 0;
      }
      this._wakeLock.release();
    };

    try {
      source.start(0, offset);
    } catch (err) {
      console.error('[VoicePlayer] source.start failed:', err);
      this._showError('Playback error');
      return;
    }

    this._currentSource = source;
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
    try { this._currentSource.stop(); } catch (stopErr) { console.warn('[VoicePlayer] source.stop failed:', stopErr); }
    try { this._currentSource.disconnect(); } catch (disconnectErr) { console.warn('[VoicePlayer] source.disconnect failed:', disconnectErr); }
    this._currentSource = null;
  }

  _currentPosition() {
    if (this._paused) return this._pausedOffset;
    if (!this._currentSource || !this._audioCtx) return 0;
    return this._sourceStartOffset + (this._audioCtx.currentTime - this._sourceStartCtxTime);
  }

  _togglePlayPause() {
    if (!this._audioCtx || !this._buffer) return;
    if (this._paused) {
      this._playBufferFrom(this._buffer, this._pausedOffset);
      return;
    }
    if (!this._currentSource) {
      // Finished playing — clicking play restarts from the beginning.
      this._playBufferFrom(this._buffer, 0);
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
    if (!this._buffer || !this._audioCtx) return;
    const pos = this._currentPosition();
    const target = pos + _SKIP_SECONDS;
    if (target < this._buffer.duration) {
      this._stopCurrentSource();
      this._playBufferFrom(this._buffer, target);
      return;
    }
    // Past end — stop, sit at the end so play restarts from 0.
    this._stopCurrentSource();
    this._stopProgressTimer();
    this._paused = false;
    this._pausedOffset = 0;
    if (this._progress) this._progress.value = this._buffer.duration || 0;
    this._setPlayIcon(false);
    this._wakeLock.release();
  }

  _skipBack() {
    if (!this._buffer || !this._audioCtx) return;
    const pos = this._currentPosition();
    const target = Math.max(0, pos - _SKIP_SECONDS);
    this._stopCurrentSource();
    this._playBufferFrom(this._buffer, target);
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
    const buf = this._buffer;
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
      const t = e.target;
      const editable = t && (
        t.tagName === 'INPUT'
        || t.tagName === 'TEXTAREA'
        || t.isContentEditable
      );
      if (editable) return;
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

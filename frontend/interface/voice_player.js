/**
 * VoicePlayer — per-message audio overlay player.
 *
 * Listens for `chalie:speak-message` custom events dispatched by renderer.js.
 * On each event: fetches the full audio WAV blob from /voice/synthesize via
 * a single HTTP request, creates an object URL, and plays it via a pre-unlocked
 * <audio> element created synchronously inside the speaker-button click handler.
 *
 * iOS Safari autoplay: renderer.js creates `new Audio()` and calls `.play()`
 * SYNCHRONOUSLY inside the click handler (before any await), unlocking the
 * audio context for iOS. That element is passed as `event.detail.audio` and
 * reused here — so `.play()` on the fetched blob is permitted by iOS even
 * though it occurs after an async fetch.
 *
 * Race / concurrency: each `_open()` increments `_openGeneration`. Any
 * in-flight fetch from a prior open is aborted via AbortController, and
 * every `await` checks the generation counter before proceeding. This
 * ensures only the most-recently requested message plays.
 *
 * Only one audio plays at a time — opening a new one aborts any previous fetch
 * and releases the previous object URL.
 */

const _TTS_PATH = '/voice/synthesize';

export class VoicePlayer {
  /**
   * @param {{ getHost: () => string }} opts
   */
  constructor({ getHost }) {
    this._getHost = getHost;

    this._overlay = null;
    this._titleEl = null;
    this._playBtn = null;
    this._bkBtn = null;
    this._ffBtn = null;
    this._closeBtn = null;
    this._progress = null;
    this._timeEl = null;
    this._errorEl = null;
    this._loadingEl = null;
    this._controlsEl = null;

    this._audio = null;
    this._objectUrl = null;
    this._boundKeydown = null;

    // Race-safety: generation counter + AbortController
    this._openGeneration = 0;
    this._fetchAbort = null;
  }

  /** Bind overlay DOM and speak-message listener. Must be called after DOM ready. */
  init() {
    this._overlay = document.getElementById('voicePlayerOverlay');
    if (!this._overlay) return;

    this._titleEl = this._overlay.querySelector('.voice-player__title');
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
    this._bkBtn?.addEventListener('click', () => this._seek(-10));
    this._ffBtn?.addEventListener('click', () => this._seek(+10));
    this._closeBtn?.addEventListener('click', () => this._close());

    // Backdrop click closes
    this._overlay.addEventListener('click', (e) => {
      if (e.target === this._overlay) this._close();
    });

    this._progress?.addEventListener('input', () => {
      if (this._audio) this._audio.currentTime = Number.parseFloat(this._progress.value);
    });

    // Listen for speak-message events from renderer.js speaker buttons.
    // event.detail carries { text, audio } — `audio` is the iOS-unlocked element.
    document.addEventListener('chalie:speak-message', (e) => {
      const { text, audio } = e.detail || {};
      if (text) this._open(text, audio);
    });
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  async _open(text, unlockedAudio) {
    // Increment generation so any concurrent open can detect it was superseded.
    this._openGeneration += 1;
    const gen = this._openGeneration;

    // Abort any in-flight fetch from a previous open.
    if (this._fetchAbort) this._fetchAbort.abort();
    this._fetchAbort = new AbortController();

    // Release previous audio and reset loading UI.
    this._releaseAudio();

    this._showOverlay();
    this._showLoading(true);
    this._showError(null);

    // Truncate title for display — take first 80 chars of the text
    if (this._titleEl) {
      this._titleEl.textContent = text.length > 80 ? text.slice(0, 77) + '...' : text;
    }

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

      // Another open was triggered while we were awaiting — bail out.
      if (gen !== this._openGeneration) return;

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        if (gen !== this._openGeneration) return;
        throw new Error(err.error || `HTTP ${resp.status}`);
      }

      const blob = await resp.blob();
      if (gen !== this._openGeneration) return;

      this._objectUrl = URL.createObjectURL(blob);

      // Reuse the iOS-unlocked Audio element if provided; otherwise create fresh.
      const audio = unlockedAudio || new Audio();
      this._audio = audio;

      audio.addEventListener('loadedmetadata', () => {
        if (gen !== this._openGeneration) return;
        this._showLoading(false);
        this._updateProgress();
        if (this._progress) {
          this._progress.max = audio.duration;
          this._progress.value = 0;
        }
        // `.play()` is allowed on this element because it was unlocked inside
        // the original click gesture (renderer.js created it synchronously).
        audio.play().catch((err) => {
          console.warn('[VoicePlayer] Autoplay blocked:', err);
          this._setPlayIcon(false);
        });
      });

      audio.addEventListener('timeupdate', () => this._updateProgress());
      audio.addEventListener('play', () => this._setPlayIcon(true));
      audio.addEventListener('pause', () => this._setPlayIcon(false));
      audio.addEventListener('ended', () => {
        this._setPlayIcon(false);
        if (this._progress) this._progress.value = audio.duration || 0;
      });
      audio.addEventListener('error', () => {
        if (gen !== this._openGeneration) return;
        this._showLoading(false);
        this._showError('Playback error');
      });

      audio.src = this._objectUrl;
      audio.load();

    } catch (err) {
      if (err.name === 'AbortError') return; // Superseded — silently discard
      if (gen !== this._openGeneration) return;
      this._showLoading(false);
      this._showError(err.message || 'Failed to load audio');
      console.error('[VoicePlayer] fetch error:', err);
    }

    this._bindKeyboard();
  }

  _close() {
    // Cancel any in-flight fetch before releasing audio.
    if (this._fetchAbort) this._fetchAbort.abort();
    this._releaseAudio();
    if (this._overlay?.open) this._overlay.close();
    this._unbindKeyboard();
  }

  _releaseAudio() {
    if (this._audio) {
      this._audio.pause();
      this._audio.src = '';
      this._audio = null;
    }
    if (this._objectUrl) {
      URL.revokeObjectURL(this._objectUrl);
      this._objectUrl = null;
    }
    this._setPlayIcon(false);
    if (this._progress) this._progress.value = 0;
    if (this._timeEl) this._timeEl.textContent = '0:00 / 0:00';
    // Reset loading UI so re-opening a new message starts from a clean state.
    this._showLoading(false);
  }

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

  _togglePlayPause() {
    if (!this._audio) return;
    if (this._audio.paused) {
      this._audio.play().catch((err) => this._showError(err.message));
    } else {
      this._audio.pause();
    }
  }

  _seek(delta) {
    if (!this._audio) return;
    const dur = this._audio.duration || 0;
    this._audio.currentTime = Math.max(0, Math.min(dur, this._audio.currentTime + delta));
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

  _updateProgress() {
    const audio = this._audio;
    if (!audio) return;
    const cur = audio.currentTime || 0;
    const dur = audio.duration || 0;
    if (this._progress && !Number.isNaN(dur)) {
      this._progress.max = dur;
      this._progress.value = cur;
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
      if (e.key === 'ArrowRight') { this._seek(+10); return; }
      if (e.key === 'ArrowLeft') this._seek(-10);
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

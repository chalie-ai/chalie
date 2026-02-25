/**
 * Ambient Behavioral Sensor — Passive-only observer.
 *
 * Collects activity state, tab focus, visibility, typing cadence, online state,
 * audio device changes, media playback state, and interruption count.
 * Never initiates network requests — heartbeat pulls snapshot() every 5min.
 */

const IDLE_MS = 2 * 60 * 1000;   // 2 min → idle
const AWAY_MS = 10 * 60 * 1000;  // 10 min → away
const CADENCE_WINDOW = 10_000;    // 10s rolling window for CPS
const CADENCE_BUFFER_SIZE = 10;   // ring buffer entries
const MEDIA_POLL_INTERVAL = 30_000; // poll mediaSession every 30s

export class AmbientSensor {
  constructor() {
    this._pageLoadTime = Date.now();
    this._lastActivityAt = Date.now();
    this._tabFocused = document.hasFocus();
    this._tabFocusedSince = Date.now();
    this._visibility = document.visibilityState;
    this._online = navigator.onLine;
    this._audioDeviceChanged = false;
    this._mediaPlaying = false;
    this._interruptionCount = 0;
    this._wasIdle = false;
    this._lastResponseAt = null;

    // Typing cadence ring buffer: [{ts, chars}]
    this._cadenceBuffer = [];
    this._cadenceInputLen = 0;

    this._bindListeners();
    this._startMediaPoll();
  }

  _bindListeners() {
    const activity = { passive: true };

    // Activity detection
    const onActivity = () => {
      const wasIdle = this._isIdle();
      this._lastActivityAt = Date.now();
      if (wasIdle) {
        this._interruptionCount++;
        this._wasIdle = false;
      }
    };
    window.addEventListener('mousemove', onActivity, activity);
    window.addEventListener('keydown', onActivity, activity);
    window.addEventListener('touchstart', onActivity, activity);
    window.addEventListener('scroll', onActivity, activity);

    // Tab focus
    window.addEventListener('focus', () => {
      this._tabFocused = true;
      this._tabFocusedSince = Date.now();
    });
    window.addEventListener('blur', () => {
      this._tabFocused = false;
    });

    // Visibility
    document.addEventListener('visibilitychange', () => {
      this._visibility = document.visibilityState;
    });

    // Online state
    window.addEventListener('online', () => { this._online = true; });
    window.addEventListener('offline', () => { this._online = false; });

    // Audio device changes
    if (navigator.mediaDevices?.addEventListener) {
      navigator.mediaDevices.addEventListener('devicechange', () => {
        this._audioDeviceChanged = true;
      });
    }
  }

  _startMediaPoll() {
    if (!navigator.mediaSession) return;
    this._mediaPollTimer = setInterval(() => {
      this._mediaPlaying = navigator.mediaSession.playbackState === 'playing';
    }, MEDIA_POLL_INTERVAL);
  }

  /**
   * Bind typing cadence tracking to a textarea/input element.
   * Records chars-per-second in a ring buffer — zero keystrokes captured.
   */
  bindTypingInput(el) {
    if (!el) return;
    this._cadenceInputLen = el.value.length;
    el.addEventListener('input', () => {
      const now = Date.now();
      const newLen = el.value.length;
      const chars = Math.abs(newLen - this._cadenceInputLen);
      this._cadenceInputLen = newLen;
      if (chars > 0) {
        this._cadenceBuffer.push({ ts: now, chars });
        if (this._cadenceBuffer.length > CADENCE_BUFFER_SIZE) {
          this._cadenceBuffer.shift();
        }
      }
    }, { passive: true });
  }

  /**
   * Record timestamp of last Chalie response (for interaction tempo).
   */
  recordResponse() {
    this._lastResponseAt = Date.now();
  }

  _isIdle() {
    return (Date.now() - this._lastActivityAt) >= IDLE_MS;
  }

  _getActivityState() {
    const elapsed = Date.now() - this._lastActivityAt;
    if (elapsed >= AWAY_MS) return 'away';
    if (elapsed >= IDLE_MS) return 'idle';
    return 'active';
  }

  _getTypingCps() {
    const now = Date.now();
    const cutoff = now - CADENCE_WINDOW;
    const recent = this._cadenceBuffer.filter(e => e.ts >= cutoff);
    if (recent.length < 2) return null;

    const totalChars = recent.reduce((sum, e) => sum + e.chars, 0);
    const span = (recent[recent.length - 1].ts - recent[0].ts) / 1000;
    if (span <= 0) return null;
    return Math.round((totalChars / span) * 10) / 10; // 1 decimal
  }

  /**
   * Return a flat dict of current behavioral state for heartbeat inclusion.
   */
  snapshot() {
    const snap = {
      activity: this._getActivityState(),
      idle_ms: Date.now() - this._lastActivityAt,
      tab_focused: this._tabFocused,
      tab_focused_since: this._tabFocusedSince,
      visibility: this._visibility,
      session_duration_ms: Date.now() - this._pageLoadTime,
      online: this._online,
      audio_device_changed: this._audioDeviceChanged,
      media_playing: this._mediaPlaying,
      interruption_count: this._interruptionCount,
    };

    const cps = this._getTypingCps();
    if (cps !== null) snap.typing_cps = cps;

    if (this._lastResponseAt) {
      snap.last_response_at = this._lastResponseAt;
    }

    // Reset per-snapshot counters
    this._interruptionCount = 0;
    this._audioDeviceChanged = false;

    return snap;
  }

  /**
   * Clean up timers.
   */
  destroy() {
    if (this._mediaPollTimer) {
      clearInterval(this._mediaPollTimer);
    }
  }
}

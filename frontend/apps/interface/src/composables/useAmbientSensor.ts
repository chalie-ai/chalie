/**
 * Ambient Behavioral Sensor — passive-only observer (singleton, created once on
 * first import). Never initiates network requests; heartbeat pulls snapshot().
 */

const IDLE_MS = 2 * 60 * 1000;        // 2 min → idle
const AWAY_MS = 10 * 60 * 1000;       // 10 min → away
const CADENCE_WINDOW = 10_000;         // 10s rolling window for CPS
const CADENCE_BUFFER_SIZE = 10;        // ring buffer entries
const MEDIA_POLL_INTERVAL = 30_000;    // poll mediaSession every 30s

/** One entry in the typing cadence ring buffer. */
interface CadenceEntry {
  ts: number;
  chars: number;
}

/** The flat behavioral snapshot sent with every heartbeat. */
export interface AmbientSnapshot {
  activity: 'active' | 'idle' | 'away';
  idle_ms: number;
  tab_focused: boolean;
  tab_focused_since: number;
  visibility: DocumentVisibilityState;
  session_duration_ms: number;
  online: boolean;
  audio_device_changed: boolean;
  media_playing: boolean;
  interruption_count: number;
  typing_cps?: number;
  last_response_at?: number;
}

/** Public API surface exposed by the singleton. */
export interface AmbientSensorApi {
  snapshot(): AmbientSnapshot;
  bindTypingInput(el: HTMLInputElement | HTMLTextAreaElement): void;
  recordResponse(): void;
  destroy(): void;
}

const _pageLoadTime = Date.now();
let _lastActivityAt = Date.now();
let _tabFocused = document.hasFocus();
let _tabFocusedSince = Date.now();
let _visibility: DocumentVisibilityState = document.visibilityState;
let _online = navigator.onLine;
let _audioDeviceChanged = false;
let _mediaPlaying = false;
let _interruptionCount = 0;
let _lastResponseAt: number | null = null;

const _cadenceBuffer: CadenceEntry[] = [];
let _cadenceInputLen = 0;
let _mediaPollTimer: ReturnType<typeof setInterval> | null = null;

function _getActivityState(): 'active' | 'idle' | 'away' {
  const elapsed = Date.now() - _lastActivityAt;
  if (elapsed >= AWAY_MS) return 'away';
  if (elapsed >= IDLE_MS) return 'idle';
  return 'active';
}

function _getTypingCps(): number | null {
  const now = Date.now();
  const cutoff = now - CADENCE_WINDOW;
  const recent = _cadenceBuffer.filter(e => e.ts >= cutoff);
  if (recent.length < 2) return null;

  const totalChars = recent.reduce((sum, e) => sum + e.chars, 0);
  const span = (recent[recent.length - 1].ts - recent[0].ts) / 1000;
  if (span <= 0) return null;
  return Math.round((totalChars / span) * 10) / 10;
}

function _bindListeners(): void {
  const passive = { passive: true };

  const onActivity = () => {
    const wasIdle = (Date.now() - _lastActivityAt) >= IDLE_MS;
    _lastActivityAt = Date.now();
    if (wasIdle) _interruptionCount++;
  };

  window.addEventListener('mousemove', onActivity, passive);
  window.addEventListener('keydown', onActivity, passive);
  window.addEventListener('touchstart', onActivity, passive);
  window.addEventListener('scroll', onActivity, passive);

  window.addEventListener('focus', () => {
    _tabFocused = true;
    _tabFocusedSince = Date.now();
  });
  window.addEventListener('blur', () => {
    _tabFocused = false;
  });

  document.addEventListener('visibilitychange', () => {
    _visibility = document.visibilityState;
  });

  window.addEventListener('online', () => { _online = true; });
  window.addEventListener('offline', () => { _online = false; });

  if (navigator.mediaDevices) {
    navigator.mediaDevices.addEventListener('devicechange', () => {
      _audioDeviceChanged = true;
    });
  }
}

function _startMediaPoll(): void {
  if (!navigator.mediaSession) return;
  _mediaPollTimer = setInterval(() => {
    _mediaPlaying = navigator.mediaSession.playbackState === 'playing';
  }, MEDIA_POLL_INTERVAL);
}

// Initialise once at module load.
_bindListeners();
_startMediaPoll();

function snapshot(): AmbientSnapshot {
  const snap: AmbientSnapshot = {
    activity: _getActivityState(),
    idle_ms: Date.now() - _lastActivityAt,
    tab_focused: _tabFocused,
    tab_focused_since: _tabFocusedSince,
    visibility: _visibility,
    session_duration_ms: Date.now() - _pageLoadTime,
    online: _online,
    audio_device_changed: _audioDeviceChanged,
    media_playing: _mediaPlaying,
    interruption_count: _interruptionCount,
  };

  const cps = _getTypingCps();
  if (cps !== null) snap.typing_cps = cps;

  if (_lastResponseAt !== null) snap.last_response_at = _lastResponseAt;

  // Counters are consumed on read — reset after each snapshot.
  _interruptionCount = 0;
  _audioDeviceChanged = false;

  return snap;
}

/** Track typing cadence on an input — counts chars only, never keystrokes. */
function bindTypingInput(el: HTMLInputElement | HTMLTextAreaElement): void {
  _cadenceInputLen = el.value.length;
  el.addEventListener('input', () => {
    const now = Date.now();
    const newLen = el.value.length;
    const chars = Math.abs(newLen - _cadenceInputLen);
    _cadenceInputLen = newLen;
    if (chars > 0) {
      _cadenceBuffer.push({ ts: now, chars });
      if (_cadenceBuffer.length > CADENCE_BUFFER_SIZE) _cadenceBuffer.shift();
    }
  }, { passive: true });
}

function recordResponse(): void {
  _lastResponseAt = Date.now();
}

function destroy(): void {
  if (_mediaPollTimer !== null) clearInterval(_mediaPollTimer);
  _mediaPollTimer = null;
}

const _sensorApi: AmbientSensorApi = { snapshot, bindTypingInput, recordResponse, destroy };

/** Returns the singleton sensor — safe to call outside setup(). */
export function useAmbientSensor(): AmbientSensorApi {
  return _sensorApi;
}

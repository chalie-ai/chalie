<template>
  <!-- Non-modal dialog (legacy: _overlay.show(), not showModal()) -->
  <dialog ref="dialogEl" class="voice-player-overlay" @click.self="_close">
    <!-- Loading state -->
    <div v-if="uiState === 'loading'" class="voice-player__loading">
      <span class="voice-player__spinner" aria-label="Loading audio…" />
    </div>

    <!-- Error state -->
    <p v-if="errorMsg" class="voice-player__error">{{ errorMsg }}</p>

    <!-- Playback controls — hidden while loading -->
    <div v-show="uiState !== 'loading'" class="voice-player__controls">
      <button
        ref="playBtnEl"
        class="voice-player__btn voice-player__btn--play"
        :aria-label="isPlaying ? 'Pause' : 'Play'"
        @click="_togglePlayPause"
      >
        <!-- Pause icon -->
        <svg v-if="isPlaying" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="6" y="4" width="4" height="16" />
          <rect x="14" y="4" width="4" height="16" />
        </svg>
        <!-- Play icon -->
        <svg v-else width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3" />
        </svg>
      </button>

      <input
        ref="progressEl"
        class="voice-player__progress"
        type="range"
        :min="0"
        :max="duration"
        :value="progressValue"
        step="0.01"
        aria-label="Playback position"
        @input="_onScrub"
      />

      <span class="voice-player__time">{{ timeLabel }}</span>

      <button
        ref="closeBtnEl"
        class="voice-player__btn voice-player__btn--close"
        aria-label="Close player"
        @click="_close"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </button>
    </div>
  </dialog>
</template>

<script setup lang="ts">
/**
 * VoicePlayerDialog — singleton non-modal TTS player.
 *
 * Port of frontend/interface/voice_player.js VoicePlayer.
 *
 * Design decision: player state is kept in local refs (not the Pinia voice
 * store) because the player is a self-contained singleton dialog that never
 * needs to expose playback state to other components. The store owns the
 * recorder (which must cross component boundaries to deliver transcripts),
 * but the player only talks to itself and the event bus.
 *
 * Constants are ported verbatim from voice_player.js:
 *   _LOADING_MAX_RETRIES   = 6
 *   _LOADING_DEFAULT_DELAY_MS = 3000
 */

import { ref, onMounted, onBeforeUnmount } from 'vue';
import { webPlatformAdapter } from '@chalie/shared';
import type { WakeLockHandle } from '@chalie/shared';
import { on } from '../../composables/useEventBus';
import { voice } from '../../api/voice';

// ── Constants (port verbatim) ─────────────────────────────────────────────────

const _LOADING_MAX_RETRIES = 6;
const _LOADING_DEFAULT_DELAY_MS = 3000;

// ── Template refs ─────────────────────────────────────────────────────────────

const dialogEl = ref<HTMLDialogElement | null>(null);
const playBtnEl = ref<HTMLButtonElement | null>(null);
const progressEl = ref<HTMLInputElement | null>(null);

// ── Reactive UI state ─────────────────────────────────────────────────────────

/** 'idle' = controls visible, 'loading' = spinner visible */
const uiState = ref<'idle' | 'loading'>('idle');
const errorMsg = ref<string | null>(null);
const isPlaying = ref(false);
const duration = ref(0);
const progressValue = ref(0);
const timeLabel = ref('0:00 / 0:00');

// ── Internal (non-reactive) playback state ────────────────────────────────────

// Generation counter: incremented on every open() and close() so stale
// fetches from superseded sessions are discarded (port of _openGeneration).
let _openGeneration = 0;
let _fetchAbort: AbortController | null = null;

// AudioContext — created lazily on first open, reused thereafter.
let _audioCtx: AudioContext | null = null;

// Per-session playback state
let _buffer: AudioBuffer | null = null;
let _currentSource: AudioBufferSourceNode | null = null;
let _sourceStartCtxTime = 0;
let _sourceStartOffset = 0;
let _paused = false;
let _pausedOffset = 0;
let _progressTimer: ReturnType<typeof setInterval> | null = null;
let _wakeLock: WakeLockHandle | null = null;

// Keyboard handler ref for add/removeEventListener symmetry
let _boundKeydown: ((e: KeyboardEvent) => void) | null = null;

// ── Lifecycle ─────────────────────────────────────────────────────────────────

let _unsubSpeak: (() => void) | null = null;

onMounted(() => {
  _unsubSpeak = on('chalie:speak-message', ({ text }) => {
    if (text) void _open(text);
  });
});

onBeforeUnmount(() => {
  _unsubSpeak?.();
  _close();
  _unbindKeyboard();
});

// ── Session lifecycle ─────────────────────────────────────────────────────────

async function _open(text: string): Promise<void> {
  _openGeneration += 1;
  const gen = _openGeneration;

  // Abort any in-flight fetch from a previous session
  _fetchAbort?.abort();
  _fetchAbort = new AbortController();

  _resetPlayback();

  if (!_audioCtx) {
    _audioCtx = webPlatformAdapter.createAudioContext();
  }
  if (_audioCtx.state === 'suspended') {
    void _audioCtx.resume();
  }

  _showDialog();
  uiState.value = 'loading';
  errorMsg.value = null;
  _bindKeyboard();

  try {
    const resp = await _fetchWithLoadingRetry(text, gen);
    if (gen !== _openGeneration || !resp) return;

    const arrayBuf = await resp.arrayBuffer();
    if (gen !== _openGeneration) return;

    const buffer = await _decodeAudio(arrayBuf);
    if (gen !== _openGeneration) return;

    _buffer = buffer;
    uiState.value = 'idle';
    duration.value = buffer.duration || 0;
    progressValue.value = 0;
    _playBufferFrom(buffer, 0);
  } catch (err: unknown) {
    if ((err as { name?: string }).name === 'AbortError') return;
    if (gen !== _openGeneration) return;
    uiState.value = 'idle';
    errorMsg.value =
      err instanceof Error ? err.message : 'Failed to synthesize';
    console.error('[VoicePlayer] synthesize error:', err);
  }
}

/**
 * POST text to /voice/synthesize with cold-start retry.
 *
 * Returns the successful Response, or null if the session was aborted or
 * superseded mid-wait. Throws on any non-loading error.
 *
 * Port of voice_player.js._fetchWithLoadingRetry — constants and logic
 * are identical.
 */
async function _fetchWithLoadingRetry(
  text: string,
  gen: number,
): Promise<Response | null> {
  for (let attempt = 0; attempt <= _LOADING_MAX_RETRIES; attempt += 1) {
    const resp = await voice.speak(text);
    if (gen !== _openGeneration) return null;
    if (resp.ok) return resp;

    const err = await resp.clone().json().catch(() => ({})) as {
      error?: string;
      reason?: string;
      hint?: string;
    };

    if (err.reason === 'deps_missing') {
      throw new Error(err.hint ?? err.error ?? 'Voice dependencies not installed');
    }
    if (err.reason === 'models_missing') {
      throw new Error(err.hint ?? err.error ?? 'Voice models not installed');
    }
    if (err.reason === 'loading' && attempt < _LOADING_MAX_RETRIES) {
      const retryAfter = Number.parseFloat(resp.headers.get('Retry-After') ?? '');
      const delayMs =
        Number.isFinite(retryAfter) && retryAfter > 0
          ? retryAfter * 1000
          : _LOADING_DEFAULT_DELAY_MS;
      await _sleepAbortable(delayMs);
      if (gen !== _openGeneration) return null;
      continue;
    }
    throw new Error(err.error ?? `HTTP ${resp.status}`);
  }
  throw new Error('Voice models still loading — please retry shortly');
}

/** Resolve after `ms`, or reject with AbortError if the active AbortController fires. */
function _sleepAbortable(ms: number): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const signal = _fetchAbort?.signal ?? null;
    const t = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(t);
      const e = new DOMException('aborted', 'AbortError');
      reject(e);
    };
    if (signal?.aborted) { onAbort(); return; }
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/** Promise wrapper around AudioContext.decodeAudioData (callback form for iOS compat). */
function _decodeAudio(arrayBuffer: ArrayBuffer): Promise<AudioBuffer> {
  return new Promise<AudioBuffer>((resolve, reject) => {
    if (!_audioCtx) { reject(new Error('AudioContext gone')); return; }
    _audioCtx.decodeAudioData(arrayBuffer, resolve, reject);
  });
}

function _close(): void {
  _openGeneration += 1; // discard any in-flight fetch
  _fetchAbort?.abort();
  _resetPlayback();
  const d = dialogEl.value;
  if (d?.open) d.close();
  _unbindKeyboard();
}

function _resetPlayback(): void {
  _stopCurrentSource();
  _stopProgressTimer();
  _buffer = null;
  _sourceStartCtxTime = 0;
  _sourceStartOffset = 0;
  _paused = false;
  _pausedOffset = 0;
  isPlaying.value = false;
  uiState.value = 'idle';
  duration.value = 0;
  progressValue.value = 0;
  timeLabel.value = '0:00 / 0:00';
  void _wakeLock?.release();
}

// ── Playback ──────────────────────────────────────────────────────────────────

function _playBufferFrom(buffer: AudioBuffer, offset: number): void {
  if (!_audioCtx) return;

  // Browsers may auto-suspend the AudioContext after a period of inactivity.
  if (_audioCtx.state === 'suspended') {
    void _audioCtx.resume();
  }

  const source = _audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(_audioCtx.destination);

  source.onended = () => {
    if (source !== _currentSource) return;
    _currentSource = null;
    if (_paused) return;
    // Natural end-of-playback: stop the timer, leave the buffer so the user
    // can press play to restart from the beginning.
    isPlaying.value = false;
    _stopProgressTimer();
    progressValue.value = buffer.duration || 0;
    void _wakeLock?.release();
  };

  try {
    source.start(0, offset);
  } catch (err) {
    console.error('[VoicePlayer] source.start failed:', err);
    errorMsg.value = 'Playback error';
    return;
  }

  _currentSource = source;
  _sourceStartCtxTime = _audioCtx.currentTime;
  _sourceStartOffset = offset;
  _paused = false;
  isPlaying.value = true;
  _startProgressTimer();

  if (!_wakeLock) _wakeLock = webPlatformAdapter.createWakeLock();
  void _wakeLock.acquire();
}

function _stopCurrentSource(): void {
  if (!_currentSource) return;
  _currentSource.onended = null;
  try { _currentSource.stop(); } catch { /* ignore */ }
  try { _currentSource.disconnect(); } catch { /* ignore */ }
  _currentSource = null;
}

function _currentPosition(): number {
  if (_paused) return _pausedOffset;
  if (!_currentSource || !_audioCtx) return 0;
  return _sourceStartOffset + (_audioCtx.currentTime - _sourceStartCtxTime);
}

function _togglePlayPause(): void {
  if (!_audioCtx || !_buffer) return;
  if (_paused) {
    _playBufferFrom(_buffer, _pausedOffset);
    return;
  }
  if (!_currentSource) {
    // Finished playing — clicking play restarts from the beginning.
    _playBufferFrom(_buffer, 0);
    return;
  }
  _pausedOffset = _currentPosition();
  _stopCurrentSource();
  _paused = true;
  isPlaying.value = false;
  _stopProgressTimer();
  void _wakeLock?.release();
}

function _onScrub(e: Event): void {
  if (!_buffer) return;
  const offset = Number.parseFloat((e.target as HTMLInputElement).value) || 0;
  _stopCurrentSource();
  _playBufferFrom(_buffer, offset);
}

// ── Progress timer ────────────────────────────────────────────────────────────

function _startProgressTimer(): void {
  _stopProgressTimer();
  _progressTimer = setInterval(() => _updateProgress(), 200);
}

function _stopProgressTimer(): void {
  if (_progressTimer !== null) {
    clearInterval(_progressTimer);
    _progressTimer = null;
  }
}

function _updateProgress(): void {
  const buf = _buffer;
  if (!buf) return;
  const cur = _currentPosition();
  const dur = buf.duration || 0;
  if (!Number.isNaN(dur)) {
    progressValue.value = Math.min(cur, dur);
  }
  timeLabel.value = `${_fmt(cur)} / ${_fmt(dur)}`;
}

function _fmt(s: number): string {
  if (Number.isNaN(s) || !Number.isFinite(s)) return '0:00';
  const m = Math.floor(s / 60);
  const sec = String(Math.floor(s % 60)).padStart(2, '0');
  return `${m}:${sec}`;
}

// ── Dialog helpers ────────────────────────────────────────────────────────────

function _showDialog(): void {
  const d = dialogEl.value;
  if (d && !d.open) d.show(); // non-modal, matches legacy
}

// ── Keyboard ──────────────────────────────────────────────────────────────────

function _bindKeyboard(): void {
  _unbindKeyboard();
  _boundKeydown = (e: KeyboardEvent) => {
    const d = dialogEl.value;
    if (!d?.open) return;
    if (e.key === 'Escape') { _close(); return; }
    const t = e.target as HTMLElement | null;
    const editable =
      t !== null &&
      (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
    if (editable) return;
    if (e.key === ' ' || e.key === 'Space') { e.preventDefault(); _togglePlayPause(); }
  };
  document.addEventListener('keydown', _boundKeydown);
}

function _unbindKeyboard(): void {
  if (_boundKeydown) {
    document.removeEventListener('keydown', _boundKeydown);
    _boundKeydown = null;
  }
}
</script>

<style scoped lang="scss">
.voice-player-overlay {
  position: fixed;
  top: 1.5rem;
  left: 50%;
  transform: translateX(-50%);
  z-index: 900;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 0.75rem;
  padding: 0.75rem 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.18);
  color: var(--text);
  min-width: 280px;

  /* <dialog> resets */
  margin: 0;
  max-width: none;
  max-height: none;
  outline: none;

  &::backdrop {
    background: transparent;
  }
}

/* A closed <dialog> must stay hidden. The author `display: flex` above
   overrides the UA `dialog:not([open]) { display: none }` rule, so without
   this guard the player renders permanently over the input dock even though
   it was never opened. Re-gate visibility on the native `open` attribute so
   it shows only after _showDialog()'s d.show() and hides again on _close(). */
.voice-player-overlay:not([open]) {
  display: none;
}

.voice-player__loading {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 2rem;
  width: 100%;
}

.voice-player__spinner {
  display: inline-block;
  width: 1.25rem;
  height: 1.25rem;
  border: 2px solid var(--border);
  border-top-color: var(--violet);
  border-radius: 50%;
  animation: vp-spin 0.7s linear infinite;
}

@keyframes vp-spin {
  to { transform: rotate(360deg); }
}

.voice-player__error {
  color: var(--error);
  font-size: 0.8125rem;
  margin: 0;
  max-width: 22rem;
}

.voice-player__controls {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  width: 100%;
}

.voice-player__btn {
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: none;
  cursor: pointer;
  color: var(--text);
  border-radius: 50%;
  padding: 0.25rem;
  flex-shrink: 0;
  transition: color 0.15s;

  &:hover {
    color: var(--violet);
  }

  &--play {
    width: 2.25rem;
    height: 2.25rem;
    background: var(--violet);
    color: #fff;
    border-radius: 50%;

    &:hover {
      color: #fff;
      opacity: 0.9;
    }
  }

  &--close {
    color: var(--text-tertiary);
    margin-left: 0.25rem;
  }
}

.voice-player__progress {
  flex: 1;
  min-width: 0;
  accent-color: var(--violet);
  height: 4px;
  cursor: pointer;
}

.voice-player__time {
  font-size: 0.75rem;
  color: var(--text-secondary);
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
  flex-shrink: 0;
}
</style>

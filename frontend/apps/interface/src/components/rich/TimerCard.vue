<script setup lang="ts">
/**
 * TimerCard — Vue 3 SFC port of frontend/interface/rich_media/timer.js
 *
 * Ported from timer.js with two sanctioned changes:
 *   1. The legacy MutationObserver DOM-detach guard is replaced with Vue's
 *      onBeforeUnmount.
 *   2. The expiry alarm now auto-silences ALARM_AUTO_STOP_MS after it starts,
 *      so an unattended timer no longer bleeps forever (legacy rang until the
 *      user pressed Stop). The card stays on "Done"; only the audio and the
 *      ring's attention pulse stop. See onExpire / silenceAlarm.
 * Everything else — ring math, stale-reload guard, alarm scheduling, pause /
 * resume / stop state machine — is reproduced exactly.
 */
import { ref, computed, onMounted, onBeforeUnmount } from 'vue';
import { webPlatformAdapter } from '@chalie/shared';

// ── Payload contract ───────────────────────────────────────────────────────

export interface TimerPayload {
  /** Opaque card identifier. */
  id: string | number;
  /** Display title. Matches TimerAbility's `{title, duration_seconds}` payload. */
  title: string;
  /** Timer duration in seconds. Matches TimerAbility's `duration_seconds`. */
  duration_seconds: number;
  /**
   * ISO-8601 wall-clock instant at which the timer was started.
   * Grafted onto the payload by TimerAbility.enrich_rich_payload; the LLM
   * never supplies it. Parsed via Date.parse — same as legacy parseStartedAt().
   */
  started_at: string;
}

const props = defineProps<{ payload: TimerPayload; synthesis?: string }>();

// ── Constants (identical to timer.js) ─────────────────────────────────────

/** pathLength / stroke-dasharray value for the SVG progress ring. */
const RING_PATH_LENGTH = 100;

/**
 * Grace window for stale-reload: if we open a conversation containing a timer
 * that already finished more than this ago, render "Ended" silently rather
 * than blasting the alarm on every page load.
 */
const STALE_RELOAD_GRACE_MS = 60 * 1_000;

/**
 * How long the expiry alarm bleeps before it auto-silences. The legacy timer
 * rang until the user pressed Stop; an unattended timer would bleep forever.
 * After this window the audio stops and the ring drops its attention pulse,
 * while the card stays on "Done".
 */
const ALARM_AUTO_STOP_MS = 30 * 1_000;

// ── Helpers (ported 1:1 from timer.js) ────────────────────────────────────

function parseStartedAt(s: string | undefined | null): number | null {
  if (!s) return null;
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : null;
}

function pad2(n: number): string {
  return n < 10 ? `0${n}` : String(n);
}

function formatRemaining(totalSeconds: number): string {
  const h = Math.floor(totalSeconds / 3_600);
  const m = Math.floor((totalSeconds % 3_600) / 60);
  const s = totalSeconds % 60;
  if (h > 0) return `${h}:${pad2(m)}:${pad2(s)}`;
  return `${pad2(m)}:${pad2(s)}`;
}

function formatHMS(d: Date): string {
  return d.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

// ── Alarm (Web Audio API via platform adapter) ─────────────────────────────

interface AlarmHandle {
  stop(): void;
}

function startAlarm(): AlarmHandle {
  let ctx: AudioContext;
  try {
    ctx = webPlatformAdapter.createAudioContext();
  } catch {
    return { stop() {} };
  }

  let burstTimer: ReturnType<typeof setTimeout> | null = null;
  let alive = true;

  /**
   * Schedule one 3-beep burst. Each beep is a sine oscillator @ 880 Hz.
   * Beeps are spaced 0.22 s apart; each lasts 0.16 s.
   * Gain ramps: 0 → 0.18 in 0.02 s (attack), holds, → 0 in 0.04 s (release).
   * The entire burst repeats every 1 400 ms (via setTimeout).
   */
  const burst = (): void => {
    if (!alive) return;
    const now = ctx.currentTime;
    for (let i = 0; i < 3; i++) {
      const osc: OscillatorNode = ctx.createOscillator();
      const gain: GainNode = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = 880;
      const start = now + i * 0.22;
      const end = start + 0.16;
      gain.gain.setValueAtTime(0, start);
      gain.gain.linearRampToValueAtTime(0.18, start + 0.02);
      gain.gain.setValueAtTime(0.18, end - 0.04);
      gain.gain.linearRampToValueAtTime(0, end);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(start);
      osc.stop(end);
    }
    burstTimer = setTimeout(burst, 1_400);
  };

  burst();

  return {
    stop(): void {
      alive = false;
      if (burstTimer !== null) {
        clearTimeout(burstTimer);
        burstTimer = null;
      }
      try {
        void ctx.close();
      } catch {
        /* already closed */
      }
    },
  };
}

// ── Derived initial values ─────────────────────────────────────────────────

const title = (props.payload.title || props.synthesis || 'Timer').toString();
const totalSeconds = Number(props.payload.duration_seconds) || 0;
const startedAtMs = parseStartedAt(props.payload.started_at);
const isInvalid = totalSeconds <= 0 || startedAtMs == null;

const totalMs = isInvalid ? 0 : totalSeconds * 1_000;
const initialEndsAtMs = isInvalid ? 0 : startedAtMs! + totalMs;

/**
 * Stale-reload guard — if the timer finished more than STALE_RELOAD_GRACE_MS
 * ago (e.g. conversation reloaded long after the timer rang), render "Ended"
 * quietly instead of blasting the alarm. Exact condition from timer.js line 52.
 */
const isStale = !isInvalid && Date.now() - initialEndsAtMs > STALE_RELOAD_GRACE_MS;

// ── Reactive state ─────────────────────────────────────────────────────────

type CardState = 'running' | 'paused' | 'done' | 'stopped' | 'stale' | 'error';

const state = ref<CardState>(
  isInvalid ? 'error'
    : isStale ? 'stale'
    : 'running',
);

/** SVG ring dashoffset (0 = empty, RING_PATH_LENGTH = full). */
const ringOffset = ref<number>(RING_PATH_LENGTH);

/** Human-readable countdown/status line. */
const timeHtml = ref<string>('');

/**
 * True once the expiry alarm has auto-silenced (after ALARM_AUTO_STOP_MS).
 * Drives `.timer-card--silenced`, which drops the ring's attention pulse.
 */
const silenced = ref<boolean>(false);

// ── Mutable runtime (not reactive — not rendered directly) ─────────────────

/** Wall-clock ms at which the timer expires; recomputed on resume. */
let endsAtMs: number = initialEndsAtMs;

/** When paused, holds the remaining ms so resume can reconstruct endsAtMs. */
let pausedRemainingMs: number | null = null;

let displayId: ReturnType<typeof setInterval> | null = null;
let alarmId: ReturnType<typeof setTimeout> | null = null;
let alarm: AlarmHandle | null = null;

/** Pending auto-silence timer scheduled by onExpire; cleared on stop/unmount. */
let autoStopId: ReturnType<typeof setTimeout> | null = null;

// ── Ring helper ────────────────────────────────────────────────────────────

/**
 * Set ring progress in [0, 1].
 * progress=0 → empty (dashoffset=RING_PATH_LENGTH).
 * progress=1 → full (dashoffset=0).
 * Matches legacy: `fill.style.strokeDashoffset = RING_PATH_LENGTH * (1 - p)`.
 */
function setProgress(p: number): void {
  const clamped = Math.max(0, Math.min(1, p));
  ringOffset.value = RING_PATH_LENGTH * (1 - clamped);
}

// ── Display update ─────────────────────────────────────────────────────────

/** Recompute ring + time line. Called every 1 s and on pause/resume. */
function update(): void {
  const remainingMs = pausedRemainingMs ?? endsAtMs - Date.now();
  const remainingSeconds = Math.max(0, Math.ceil(remainingMs / 1_000));
  setProgress(Math.min(1, 1 - remainingSeconds / totalSeconds));
  if (remainingMs > 0) {
    timeHtml.value =
      `<b>${formatRemaining(remainingSeconds)}</b> remaining · ends ${formatHMS(new Date(endsAtMs))}`;
  }
}

// ── Expiry ─────────────────────────────────────────────────────────────────

function onExpire(): void {
  alarmId = null;
  if (displayId !== null) {
    clearInterval(displayId);
    displayId = null;
  }
  timeHtml.value = '<b>Done</b>';
  state.value = 'done';
  setProgress(1);
  alarm = startAlarm();
  // Auto-silence the alarm after the grace window so an unattended timer does
  // not bleep indefinitely. Pressing Stop earlier cancels this via cleanup().
  autoStopId = setTimeout(silenceAlarm, ALARM_AUTO_STOP_MS);
}

/**
 * Silence the expiry alarm: stop the audio, cancel the auto-stop timer, and
 * flip `silenced` so the ring drops its attention pulse (card stays on "Done").
 * Idempotent — safe whether fired by the auto-stop timer or called again.
 */
function silenceAlarm(): void {
  if (autoStopId !== null) {
    clearTimeout(autoStopId);
    autoStopId = null;
  }
  if (alarm !== null) {
    alarm.stop();
    alarm = null;
  }
  silenced.value = true;
}

// ── Cleanup (replaces MutationObserver from timer.js) ─────────────────────

/** Tears down all timers and the AudioContext. Called on unmount and stop. */
function cleanup(): void {
  if (displayId !== null) {
    clearInterval(displayId);
    displayId = null;
  }
  if (alarmId !== null) {
    clearTimeout(alarmId);
    alarmId = null;
  }
  if (autoStopId !== null) {
    clearTimeout(autoStopId);
    autoStopId = null;
  }
  if (alarm !== null) {
    alarm.stop();
    alarm = null;
  }
}

// ── Pause / Resume / Stop ──────────────────────────────────────────────────

function onPause(): void {
  if (state.value !== 'running') return;
  pausedRemainingMs = Math.max(0, endsAtMs - Date.now());
  if (alarmId !== null) {
    clearTimeout(alarmId);
    alarmId = null;
  }
  // Stop alarm if it somehow started while transitioning
  if (alarm !== null) {
    alarm.stop();
    alarm = null;
  }
  state.value = 'paused';
  update();
}

function onResume(): void {
  if (state.value !== 'paused' || pausedRemainingMs === null) return;
  endsAtMs = Date.now() + pausedRemainingMs;
  alarmId = setTimeout(onExpire, pausedRemainingMs);
  pausedRemainingMs = null;
  state.value = 'running';
  update();
}

function onStop(): void {
  cleanup();
  state.value = 'stopped';
  timeHtml.value = '<b>Stopped</b>';
  setProgress(0);
}

// ── Derived template helpers ───────────────────────────────────────────────

const isPauseDisabled = computed<boolean>(
  () => state.value !== 'running' && state.value !== 'paused',
);

const isStopDisabled = computed<boolean>(
  () => state.value === 'stopped' || state.value === 'stale' || state.value === 'error',
);

// ── Lifecycle ──────────────────────────────────────────────────────────────

onMounted((): void => {
  if (isInvalid || isStale) {
    // Stale: ring full + disabled. Error: nothing scheduled.
    if (isStale) {
      timeHtml.value = '<b>Ended</b>';
      setProgress(1);
    }
    return;
  }

  // Schedule alarm + display loop. If we land already past endsAtMs but within
  // the grace window (isStale=false but remainingMs<=0), fire onExpire
  // synchronously so Done renders immediately — exactly as timer.js does.
  const remainingMs = endsAtMs - Date.now();
  if (remainingMs <= 0) {
    onExpire();
  } else {
    alarmId = setTimeout(onExpire, remainingMs);
    update();
    displayId = setInterval(update, 1_000);
  }
});

onBeforeUnmount((): void => {
  cleanup();
});
</script>

<template>
  <!-- Error: invalid payload -->
  <div
    v-if="state === 'error'"
    class="rich-card timer-card timer-card--error"
  >
    Invalid timer payload
  </div>

  <!-- Normal card (running / paused / done / stopped / stale) -->
  <div
    v-else
    class="rich-card timer-card"
    :class="{
      'timer-card--done': state === 'done' || state === 'stale',
      'timer-card--stopped': state === 'stopped',
      'timer-card--silenced': silenced,
    }"
  >
    <!-- Progress ring (56 × 56 px; SVG viewBox 36 × 36; radius 16; pathLength 100) -->
    <div class="timer-card__ring" aria-hidden="true">
      <svg viewBox="0 0 36 36">
        <circle
          class="timer-card__ring-track"
          cx="18"
          cy="18"
          r="16"
        />
        <circle
          class="timer-card__ring-fill"
          cx="18"
          cy="18"
          r="16"
          :pathLength="RING_PATH_LENGTH"
          :stroke-dasharray="RING_PATH_LENGTH"
          :stroke-dashoffset="ringOffset"
        />
      </svg>
    </div>

    <!-- Title + countdown line -->
    <div class="timer-card__info">
      <h4 class="timer-card__title">{{ title }}</h4>
      <!-- eslint-disable-next-line vue/no-v-html -->
      <div class="timer-card__time" v-html="timeHtml" />
    </div>

    <!-- Pause/Resume + Stop buttons -->
    <div class="timer-card__actions">
      <!-- Pause / Resume toggle -->
      <button
        type="button"
        class="timer-card__btn"
        :title="state === 'paused' ? 'Resume' : 'Pause'"
        :aria-label="state === 'paused' ? 'Resume' : 'Pause'"
        :disabled="isPauseDisabled"
        @click="state === 'paused' ? onResume() : onPause()"
      >
        <!-- Pause icon -->
        <svg
          v-if="state !== 'paused'"
          viewBox="0 0 24 24"
          fill="currentColor"
          aria-hidden="true"
        >
          <rect x="6" y="5" width="4" height="14" />
          <rect x="14" y="5" width="4" height="14" />
        </svg>
        <!-- Play / Resume icon -->
        <svg
          v-else
          viewBox="0 0 24 24"
          fill="currentColor"
          aria-hidden="true"
        >
          <polygon points="6,4 20,12 6,20" />
        </svg>
      </button>

      <!-- Stop button -->
      <button
        type="button"
        class="timer-card__btn"
        title="Stop"
        aria-label="Stop"
        :disabled="isStopDisabled"
        @click="onStop"
      >
        <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <rect x="6" y="6" width="12" height="12" rx="1" />
        </svg>
      </button>
    </div>
  </div>
</template>

<style scoped lang="scss">
/*
 * Card-specific rules only. Base chrome (.rich-card, __divider, __synthesis,
 * card-enter animation) is declared globally in base_card.css — not repeated here.
 */

.rich-card.timer-card {
  display: grid;
  grid-template-columns: 56px 1fr auto;
  gap: 16px;
  align-items: center;
  border: none;
  box-shadow: none;
}

/* ── Progress ring ──────────────────────────────────────────────────────── */

.timer-card__ring {
  position: relative;
  width: 56px;
  height: 56px;

  svg {
    width: 100%;
    height: 100%;
    transform: rotate(-90deg);
  }
}

.timer-card__ring-track {
  fill: none;
  stroke: var(--border, color-mix(in oklab, var(--violet) 18%, transparent));
  stroke-width: 3;
}

.timer-card__ring-fill {
  fill: none;
  stroke: var(--violet, #8a5cff);
  stroke-width: 3;
  stroke-linecap: round;
  transition: stroke-dashoffset 0.4s linear;
}

.timer-card--done .timer-card__ring-fill {
  stroke: var(--amber, #ffb257);
  animation: timer-ring-pulse 1s ease-in-out infinite;
}

@keyframes timer-ring-pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.45; }
}

/*
 * Auto-silenced alarm: keep the amber "Done" ring but stop the attention
 * pulse once the audio has auto-stopped (ALARM_AUTO_STOP_MS after expiry).
 * Compound `--done--silenced` selector (specificity 0-3-0) deliberately
 * outranks the `--done` pulse rule (0-2-0) so it wins by specificity, not by
 * source order — a silenced card always carries both classes.
 */
.timer-card--done.timer-card--silenced .timer-card__ring-fill {
  animation: none;
}

/* ── Title + time ────────────────────────────────────────────────────────── */

.timer-card__info {
  min-width: 0;
}

.timer-card__title {
  font-size: 0.96rem;
  font-weight: 500;
  letter-spacing: -0.005em;
  margin: 0 0 2px;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.timer-card__time {
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.78rem;
  color: var(--text-tertiary);
  letter-spacing: 0.04em;

  :deep(b) {
    color: var(--text-secondary);
    font-weight: 500;
  }
}

/* ── Action buttons ─────────────────────────────────────────────────────── */

.timer-card__actions {
  display: flex;
  gap: 6px;
}

.timer-card__btn {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: 1px solid var(--border, color-mix(in oklab, var(--violet) 22%, transparent));
  background: transparent;
  color: var(--text-secondary);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: color 160ms ease, border-color 160ms ease;
  padding: 0;

  &:hover:not(:disabled) {
    color: var(--text-primary);
    border-color: color-mix(in oklab, var(--violet) 50%, var(--border));
  }

  &:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  svg {
    width: 12px;
    height: 12px;
    display: block;
  }
}

/* ── Stopped + error variants ───────────────────────────────────────────── */

.timer-card--stopped {
  opacity: 0.65;
}

.timer-card--error {
  display: block;
  color: var(--text-tertiary);
  font-style: italic;
}
</style>

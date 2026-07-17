<script setup lang="ts">
/**
 * TimerCard — two intentional divergences from the legacy timer:
 *   1. MutationObserver DOM-detach guard replaced with Vue onBeforeUnmount.
 *   2. The expiry alarm auto-silences ALARM_AUTO_STOP_MS after it starts so an
 *      unattended timer no longer bleeps forever (card stays "Done"; only audio
 *      + ring pulse stop). See onExpire / silenceAlarm.
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { Pause, Play, Square } from '@lucide/vue';
import { webPlatformAdapter } from '@chalie/shared';
import { formatClock, formatDuration } from '../../utils/time';

export interface TimerPayload {
  id: string | number;
  title: string;
  duration_seconds: number;
  /** Wall-clock start instant grafted on by the backend; LLM never supplies it. */
  started_at: string;
}

const props = defineProps<{ payload: TimerPayload; synthesis?: string }>();

/** pathLength / stroke-dasharray value for the SVG progress ring. */
const RING_PATH_LENGTH = 100;

/** Stale-reload grace: a timer that finished longer ago than this renders
 *  "Ended" silently instead of blasting the alarm on every page load. */
const STALE_RELOAD_GRACE_MS = 60 * 1_000;

/** How long the expiry alarm bleeps before auto-silencing (card stays "Done";
 *  audio + ring pulse stop) so an unattended timer doesn't bleep forever. */
const ALARM_AUTO_STOP_MS = 30 * 1_000;

function parseStartedAt(s: string | undefined | null): number | null {
  if (!s) return null;
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : null;
}

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

  /** Schedule one 3-beep sine burst, then repeat every 1 400 ms via setTimeout. */
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
        ctx.close();
      } catch {
        /* already closed */
      }
    },
  };
}

const title = (props.payload.title || props.synthesis || 'Timer').toString();
const totalSeconds = Number(props.payload.duration_seconds) || 0;
const startedAtMs = parseStartedAt(props.payload.started_at);
const isInvalid = totalSeconds <= 0 || startedAtMs == null;

const totalMs = isInvalid ? 0 : totalSeconds * 1_000;
const initialEndsAtMs = isInvalid ? 0 : startedAtMs! + totalMs;

const isStale = !isInvalid && Date.now() - initialEndsAtMs > STALE_RELOAD_GRACE_MS;

type CardState = 'running' | 'paused' | 'done' | 'stopped' | 'stale' | 'error';

function initialCardState(): CardState {
  if (isInvalid) return 'error';
  if (isStale) return 'stale';
  return 'running';
}
const state = ref<CardState>(initialCardState());

/** SVG ring dashoffset (0 = empty, RING_PATH_LENGTH = full). */
const ringOffset = ref<number>(RING_PATH_LENGTH);

const timeHtml = ref<string>('');

/** True once the alarm auto-silenced; drives `.timer-card--silenced`. */
const silenced = ref<boolean>(false);

// Mutable runtime below is not reactive (not rendered directly).

/** Wall-clock ms at which the timer expires; recomputed on resume. */
let endsAtMs: number = initialEndsAtMs;

/** When paused, holds the remaining ms so resume can reconstruct endsAtMs. */
let pausedRemainingMs: number | null = null;

let displayId: ReturnType<typeof setInterval> | null = null;
let alarmId: ReturnType<typeof setTimeout> | null = null;
let alarm: AlarmHandle | null = null;

/** Pending auto-silence timer scheduled by onExpire; cleared on stop/unmount. */
let autoStopId: ReturnType<typeof setTimeout> | null = null;

/** Set ring progress in [0, 1]: 0 → empty, 1 → full. */
function setProgress(p: number): void {
  const clamped = Math.max(0, Math.min(1, p));
  ringOffset.value = RING_PATH_LENGTH * (1 - clamped);
}

/** Recompute ring + time line. Called every 1 s and on pause/resume. */
function update(): void {
  const remainingMs = pausedRemainingMs ?? endsAtMs - Date.now();
  const remainingSeconds = Math.max(0, Math.ceil(remainingMs / 1_000));
  setProgress(Math.min(1, 1 - remainingSeconds / totalSeconds));
  if (remainingMs > 0) {
    timeHtml.value = `<b>${formatDuration(remainingSeconds)}</b> remaining · ends ${formatClock(new Date(endsAtMs), { seconds: true })}`;
  }
}

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
  // Auto-silence after the grace window; pressing Stop earlier cancels via cleanup().
  autoStopId = setTimeout(silenceAlarm, ALARM_AUTO_STOP_MS);
}

/** Stop the alarm audio + cancel auto-stop; flip `silenced`. Idempotent. */
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

const isPauseDisabled = computed<boolean>(
  () => state.value !== 'running' && state.value !== 'paused',
);

const isStopDisabled = computed<boolean>(
  () => state.value === 'stopped' || state.value === 'stale' || state.value === 'error',
);

onMounted((): void => {
  if (isInvalid || isStale) {
    // Stale: ring full + disabled. Error: nothing scheduled.
    if (isStale) {
      timeHtml.value = '<b>Ended</b>';
      setProgress(1);
    }
    return;
  }

  // If already past endsAtMs but within the grace window, fire onExpire
  // synchronously so Done renders immediately.
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
  <div v-if="state === 'error'" class="rich-card timer-card timer-card--error">
    Invalid timer payload
  </div>

  <div
    v-else
    class="rich-card timer-card"
    :class="{
      'timer-card--done': state === 'done' || state === 'stale',
      'timer-card--stopped': state === 'stopped',
      'timer-card--silenced': silenced,
    }"
  >
    <div class="timer-card__ring" aria-hidden="true">
      <svg viewBox="0 0 36 36">
        <circle class="timer-card__ring-track" cx="18" cy="18" r="16" />
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

    <div class="timer-card__info">
      <h4 class="timer-card__title">{{ title }}</h4>
      <!-- eslint-disable-next-line vue/no-v-html -->
      <div class="timer-card__time" v-html="timeHtml" />
    </div>

    <div class="timer-card__actions">
      <button
        type="button"
        class="timer-card__btn"
        :title="state === 'paused' ? 'Resume' : 'Pause'"
        :aria-label="state === 'paused' ? 'Resume' : 'Pause'"
        :disabled="isPauseDisabled"
        @click="state === 'paused' ? onResume() : onPause()"
      >
        <Pause v-if="state !== 'paused'" fill="currentColor" aria-hidden="true" />
        <Play v-else fill="currentColor" aria-hidden="true" />
      </button>

      <button
        type="button"
        class="timer-card__btn"
        title="Stop"
        aria-label="Stop"
        :disabled="isStopDisabled"
        @click="onStop"
      >
        <Square fill="currentColor" aria-hidden="true" />
      </button>
    </div>
  </div>
</template>

<style scoped lang="scss">
/* Card-specific rules only; base .rich-card chrome lives globally in base_card.css. */

.rich-card.timer-card {
  display: grid;
  grid-template-columns: 56px 1fr auto;
  gap: 16px;
  align-items: center;
  border: none;
  box-shadow: none;
}

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
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.45;
  }
}

/*
 * Compound `--done.--silenced` (specificity 0-3-0) deliberately outranks the
 * `--done` pulse rule (0-2-0) by specificity, not source order — a silenced
 * card always carries both classes — so it cancels the pulse while keeping amber.
 */
.timer-card--done.timer-card--silenced .timer-card__ring-fill {
  animation: none;
}

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
  transition:
    color 160ms ease,
    border-color 160ms ease;
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

.timer-card--stopped {
  opacity: 0.65;
}

.timer-card--error {
  display: block;
  color: var(--text-tertiary);
  font-style: italic;
}
</style>

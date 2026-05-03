/**
 * Timer card module — live countdown.
 *
 * Implements the rich-media card contract: exports render(payload, synthesis, root).
 * Renders a 56px progress ring + title + "MM:SS remaining · ends HH:MM" line +
 * pause/play and stop buttons. When the countdown reaches zero, an alarm beep
 * loop plays via the Web Audio API until the user clicks stop.
 *
 * Payload shape (post-parser):
 *   { title, duration_seconds, started_at }  // started_at is ISO 8601 UTC.
 *
 * `started_at` is NOT produced by TimerAbility — the LLM never sees it. It is
 * grafted onto the payload by RichMediaParser from the tool_calls row's
 * created_at column. This keeps the wall-clock invisible to the model while
 * still giving the FE a reload-safe anchor.
 *
 * Reload-safe: the remaining time is computed from `now() - started_at` on
 * every tick, so refreshing the page resumes the countdown from real elapsed.
 * Pause is in-memory only — refreshing while paused will resume the timer.
 */

const RING_PATH_LENGTH = 100; // matches `pathLength="100"` on the SVG circle.
const ICON_PAUSE = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="5" width="4" height="14"/><rect x="14" y="5" width="4" height="14"/></svg>';
const ICON_PLAY = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><polygon points="6,4 20,12 6,20"/></svg>';
const ICON_STOP = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>';

// Grace window: if the first render lands within this many ms of the natural
// end, treat the timer as "just finished now" and let the alarm play. Past
// that window the timer is considered stale (page reload of an old turn) —
// render Done silently so we don't blare alarms on every conversation reload.
const STALE_RELOAD_GRACE_MS = 60 * 1000;

export function render(payload, synthesis, root) {
  const title = (payload.title || synthesis || 'Timer').toString();
  const totalSeconds = Number(payload.duration_seconds) || 0;
  const startedAtMs = parseStartedAt(payload.started_at);
  if (totalSeconds <= 0 || startedAtMs == null) {
    appendError(root, 'Invalid timer payload');
    return;
  }

  const card = document.createElement('div');
  card.className = 'rich-card timer-card';

  const ring = buildRing();
  card.appendChild(ring.el);

  const info = document.createElement('div');
  info.className = 'timer-card__info';

  const titleEl = document.createElement('h4');
  titleEl.className = 'timer-card__title';
  titleEl.textContent = title;
  info.appendChild(titleEl);

  const timeEl = document.createElement('div');
  timeEl.className = 'timer-card__time';
  info.appendChild(timeEl);
  card.appendChild(info);

  const actions = document.createElement('div');
  actions.className = 'timer-card__actions';
  const pauseBtn = makeBtn('Pause', ICON_PAUSE);
  const stopBtn = makeBtn('Stop', ICON_STOP);
  actions.appendChild(pauseBtn);
  actions.appendChild(stopBtn);
  card.appendChild(actions);

  root.appendChild(card);

  const state = {
    totalSeconds,
    startedAtMs,
    pausedAtMs: null,
    accumulatedPausedMs: 0,
    intervalId: null,
    audioCtx: null,
    alarmTimer: null,
    alarmPlaying: false,
    stopped: false,
    domObserver: null,
  };

  // Stale-reload guard: a finished timer reloaded from /conversation/recent
  // would otherwise fire its alarm on every page open. If we're already past
  // the natural end by more than the grace window, render the Done state
  // silently and skip the tick loop entirely.
  const initialElapsedMs = Date.now() - startedAtMs;
  const overshootMs = initialElapsedMs - totalSeconds * 1000;
  if (overshootMs > STALE_RELOAD_GRACE_MS) {
    timeEl.innerHTML = '<b>Ended</b>';
    card.classList.add('timer-card--done');
    pauseBtn.disabled = true;
    stopBtn.disabled = true;
    ring.setProgress(1);
    return;
  }

  pauseBtn.addEventListener('click', () => {
    if (state.stopped) return;
    if (state.pausedAtMs == null) {
      state.pausedAtMs = Date.now();
      pauseBtn.innerHTML = ICON_PLAY;
      pauseBtn.title = 'Resume';
    } else {
      state.accumulatedPausedMs += Date.now() - state.pausedAtMs;
      state.pausedAtMs = null;
      pauseBtn.innerHTML = ICON_PAUSE;
      pauseBtn.title = 'Pause';
    }
  });

  stopBtn.addEventListener('click', () => {
    cleanup(state);
    card.classList.add('timer-card--stopped');
    pauseBtn.disabled = true;
    stopBtn.disabled = true;
    timeEl.innerHTML = '<b>Stopped</b>';
    ring.setProgress(0);
  });

  const tick = () => {
    if (state.stopped) return;
    const elapsedMs = effectiveElapsedMs(state);
    const remainingSeconds = Math.max(0, state.totalSeconds - Math.floor(elapsedMs / 1000));
    const progress = Math.max(0, Math.min(1, 1 - remainingSeconds / state.totalSeconds));
    ring.setProgress(progress);

    if (remainingSeconds > 0) {
      // Account for any in-progress pause so the displayed end-time freezes
      // alongside the countdown rather than ticking forward during a hold.
      const currentPauseMs = state.pausedAtMs != null ? Date.now() - state.pausedAtMs : 0;
      const endsAt = new Date(
        state.startedAtMs + state.accumulatedPausedMs + currentPauseMs + state.totalSeconds * 1000,
      );
      timeEl.innerHTML = `<b>${formatRemaining(remainingSeconds)}</b> remaining · ends ${formatHM(endsAt)}`;
    } else {
      timeEl.innerHTML = '<b>Done</b>';
      card.classList.add('timer-card--done');
      if (!state.alarmPlaying) {
        startAlarm(state);
      }
      // Stop the 1Hz tick once the countdown is exhausted — the alarm has its
      // own self-rescheduling setTimeout chain. Without this clearInterval the
      // tick would keep firing forever on every completed timer in the DOM.
      if (state.intervalId != null) {
        clearInterval(state.intervalId);
        state.intervalId = null;
      }
    }
  };

  tick();
  state.intervalId = setInterval(tick, 1000);

  // Lifecycle cleanup: if the card element is detached (conversation cleared,
  // history reloaded, SPA nav) while alarm/interval are live, the closures
  // would otherwise keep an AudioContext + setTimeout chain alive forever.
  state.domObserver = new MutationObserver(() => {
    if (!document.contains(card)) {
      cleanup(state);
    }
  });
  state.domObserver.observe(document.body, { childList: true, subtree: true });
}

function cleanup(state) {
  state.stopped = true;
  stopAlarm(state);
  if (state.intervalId != null) {
    clearInterval(state.intervalId);
    state.intervalId = null;
  }
  if (state.domObserver) {
    state.domObserver.disconnect();
    state.domObserver = null;
  }
}

function effectiveElapsedMs(state) {
  if (state.pausedAtMs != null) {
    return state.pausedAtMs - state.startedAtMs - state.accumulatedPausedMs;
  }
  return Date.now() - state.startedAtMs - state.accumulatedPausedMs;
}

function buildRing() {
  const wrap = document.createElement('div');
  wrap.className = 'timer-card__ring';
  wrap.innerHTML = `
    <svg viewBox="0 0 36 36" aria-hidden="true">
      <circle class="timer-card__ring-track" cx="18" cy="18" r="16"></circle>
      <circle class="timer-card__ring-fill" cx="18" cy="18" r="16"
              pathLength="${RING_PATH_LENGTH}"
              stroke-dasharray="${RING_PATH_LENGTH}"
              stroke-dashoffset="${RING_PATH_LENGTH}"></circle>
    </svg>`;
  const fill = wrap.querySelector('.timer-card__ring-fill');
  return {
    el: wrap,
    setProgress(p) {
      fill.style.strokeDashoffset = `${RING_PATH_LENGTH * (1 - p)}`;
    },
  };
}

function makeBtn(title, svg) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'timer-card__btn';
  btn.title = title;
  btn.setAttribute('aria-label', title);
  btn.innerHTML = svg;
  return btn;
}

function parseStartedAt(s) {
  if (!s) return null;
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : null;
}

function formatRemaining(totalSeconds) {
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  if (h > 0) return `${h}:${pad2(m)}:${pad2(s)}`;
  return `${pad2(m)}:${pad2(s)}`;
}

function formatHM(d) {
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
}

function pad2(n) {
  return n < 10 ? `0${n}` : String(n);
}

function appendError(root, msg) {
  const card = document.createElement('div');
  card.className = 'rich-card timer-card timer-card--error';
  card.textContent = msg;
  root.appendChild(card);
}

// ── Alarm (Web Audio API) ──────────────────────────────────────────────────

function startAlarm(state) {
  state.alarmPlaying = true;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    state.audioCtx = new Ctx();
    const playBurst = () => {
      if (!state.alarmPlaying || !state.audioCtx) return;
      const ctx = state.audioCtx;
      const now = ctx.currentTime;
      // Three short 880Hz beeps per burst.
      for (let i = 0; i < 3; i++) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
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
      state.alarmTimer = setTimeout(playBurst, 1400);
    };
    playBurst();
  } catch (e) {
    // Audio is best-effort; failing to play does not break the card.
    state.alarmPlaying = false;
  }
}

function stopAlarm(state) {
  state.alarmPlaying = false;
  if (state.alarmTimer != null) {
    clearTimeout(state.alarmTimer);
    state.alarmTimer = null;
  }
  if (state.audioCtx) {
    try { state.audioCtx.close(); } catch (e) { /* already closed */ }
    state.audioCtx = null;
  }
}

/**
 * Timer card module — live countdown.
 *
 * Renders a 56px progress ring + title + "MM:SS remaining · ends HH:MM:SS"
 * line + pause/resume and stop buttons. When the countdown reaches zero, an
 * alarm beep loop plays via the Web Audio API until the user clicks Stop.
 *
 * Payload shape (post-parser):
 *   { title, duration_seconds, started_at }
 *
 * `started_at` is grafted onto the payload by TimerAbility.enrich_rich_payload
 * from the tool_calls row's created_at — the LLM never sees it.
 *
 * The single source of truth is `endsAtMs` (real wall-clock end time). Display
 * reads it; a single setTimeout fires the alarm at it. Pause stashes the
 * remaining ms and clears the alarm; Resume recomputes endsAtMs from now() +
 * remaining and reschedules. Reload-safe: nothing is stored in JS memory that
 * isn't reconstructible from `started_at + duration_seconds`.
 */

const RING_PATH_LENGTH = 100;
const ICON_PAUSE = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="5" width="4" height="14"/><rect x="14" y="5" width="4" height="14"/></svg>';
const ICON_PLAY = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><polygon points="6,4 20,12 6,20"/></svg>';
const ICON_STOP = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>';

// Grace window for stale-reload: if we open a conversation containing a timer
// that already finished more than this ago, render "Ended" silently rather
// than blasting the alarm on every page load.
const STALE_RELOAD_GRACE_MS = 60 * 1000;

export function render(payload, synthesis, root) {
  const title = (payload.title || synthesis || 'Timer').toString();
  const totalSeconds = Number(payload.duration_seconds) || 0;
  const startedAtMs = parseStartedAt(payload.started_at);
  if (totalSeconds <= 0 || startedAtMs == null) {
    appendError(root, 'Invalid timer payload');
    return;
  }

  const card = buildCard(title);
  root.appendChild(card.el);

  const totalMs = totalSeconds * 1000;
  let endsAtMs = startedAtMs + totalMs;
  let pausedRemainingMs = null;
  let displayId = null;
  let alarmId = null;
  let alarm = null;
  let observer = null;

  // Stale-reload guard — past the grace window, render quiet Ended.
  if (Date.now() - endsAtMs > STALE_RELOAD_GRACE_MS) {
    card.timeEl.innerHTML = '<b>Ended</b>';
    card.el.classList.add('timer-card--done');
    card.pauseBtn.disabled = true;
    card.stopBtn.disabled = true;
    card.ring.setProgress(1);
    return;
  }

  const update = () => {
    const remainingMs = pausedRemainingMs ?? endsAtMs - Date.now();
    const remainingSeconds = Math.max(0, Math.ceil(remainingMs / 1000));
    card.ring.setProgress(Math.min(1, 1 - remainingSeconds / totalSeconds));
    if (remainingMs > 0) {
      card.timeEl.innerHTML =
        `<b>${formatRemaining(remainingSeconds)}</b> remaining · ends ${formatHMS(new Date(endsAtMs))}`;
    }
  };

  const onExpire = () => {
    alarmId = null;
    if (displayId != null) { clearInterval(displayId); displayId = null; }
    card.timeEl.innerHTML = '<b>Done</b>';
    card.el.classList.add('timer-card--done');
    card.ring.setProgress(1);
    alarm = startAlarm();
  };

  const cleanup = () => {
    if (displayId != null) { clearInterval(displayId); displayId = null; }
    if (alarmId != null) { clearTimeout(alarmId); alarmId = null; }
    if (alarm) { alarm.stop(); alarm = null; }
    if (observer) { observer.disconnect(); observer = null; }
  };

  card.pauseBtn.addEventListener('click', () => {
    if (pausedRemainingMs == null) {
      pausedRemainingMs = Math.max(0, endsAtMs - Date.now());
      if (alarmId != null) { clearTimeout(alarmId); alarmId = null; }
      card.pauseBtn.innerHTML = ICON_PLAY;
      card.pauseBtn.title = 'Resume';
    } else {
      endsAtMs = Date.now() + pausedRemainingMs;
      alarmId = setTimeout(onExpire, pausedRemainingMs);
      pausedRemainingMs = null;
      card.pauseBtn.innerHTML = ICON_PAUSE;
      card.pauseBtn.title = 'Pause';
    }
    update();
  });

  card.stopBtn.addEventListener('click', () => {
    cleanup();
    card.el.classList.add('timer-card--stopped');
    card.pauseBtn.disabled = true;
    card.stopBtn.disabled = true;
    card.timeEl.innerHTML = '<b>Stopped</b>';
    card.ring.setProgress(0);
  });

  // Schedule alarm + display loop. If we land already past endsAtMs but within
  // the grace window, fire onExpire synchronously so Done renders immediately.
  const remainingMs = endsAtMs - Date.now();
  if (remainingMs <= 0) {
    onExpire();
  } else {
    alarmId = setTimeout(onExpire, remainingMs);
    update();
    displayId = setInterval(update, 1000);
  }

  // DOM-detach cleanup — without this the AudioContext + setTimeout chain
  // would outlive the card on conversation clear / SPA nav.
  observer = new MutationObserver(() => {
    if (!document.contains(card.el)) cleanup();
  });
  observer.observe(document.body, { childList: true, subtree: true });
}

// ── DOM construction ───────────────────────────────────────────────────────

function buildCard(title) {
  const el = document.createElement('div');
  el.className = 'rich-card timer-card';

  const ring = buildRing();
  el.appendChild(ring.el);

  const info = document.createElement('div');
  info.className = 'timer-card__info';
  const titleEl = document.createElement('h4');
  titleEl.className = 'timer-card__title';
  titleEl.textContent = title;
  info.appendChild(titleEl);
  const timeEl = document.createElement('div');
  timeEl.className = 'timer-card__time';
  info.appendChild(timeEl);
  el.appendChild(info);

  const actions = document.createElement('div');
  actions.className = 'timer-card__actions';
  const pauseBtn = makeBtn('Pause', ICON_PAUSE);
  const stopBtn = makeBtn('Stop', ICON_STOP);
  actions.appendChild(pauseBtn);
  actions.appendChild(stopBtn);
  el.appendChild(actions);

  return { el, ring, timeEl, pauseBtn, stopBtn };
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
      fill.style.strokeDashoffset = `${RING_PATH_LENGTH * (1 - Math.max(0, Math.min(1, p)))}`;
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

function appendError(root, msg) {
  const card = document.createElement('div');
  card.className = 'rich-card timer-card timer-card--error';
  card.textContent = msg;
  root.appendChild(card);
}

// ── Formatting ─────────────────────────────────────────────────────────────

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

function formatHMS(d) {
  return d.toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  });
}

function pad2(n) {
  return n < 10 ? `0${n}` : String(n);
}

// ── Alarm (Web Audio API) ──────────────────────────────────────────────────

function startAlarm() {
  const Ctx = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!Ctx) return { stop() {} };
  let ctx;
  try { ctx = new Ctx(); } catch { return { stop() {} }; }

  let timer = null;
  let alive = true;

  const burst = () => {
    if (!alive) return;
    const now = ctx.currentTime;
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
    timer = setTimeout(burst, 1400);
  };

  burst();

  return {
    stop() {
      alive = false;
      if (timer != null) { clearTimeout(timer); timer = null; }
      try { ctx.close(); } catch { /* already closed */ }
    },
  };
}

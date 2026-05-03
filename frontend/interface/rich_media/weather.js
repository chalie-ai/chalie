/**
 * Weather card module — ambient sky variant.
 *
 * Renders a hero canvas (gradient sky tinted by local time + condition,
 * drifting clouds, sun/moon, skyline silhouette), a readout overlay
 * (big temperature top-left, location + day/time top-right, narrative
 * caption bottom-left), and an 8-hour rail at the bottom.
 *
 * Implements the rich-media card contract: render(payload, synthesis, root).
 * Payload is the WeatherAbility dict; the new fields it depends on
 * (`hourly`, `sunrise`, `sunset`) are tolerated as missing — the rail
 * simply hides if `hourly` is empty, and the gradient falls back to a
 * plain day/night split when sunrise/sunset are absent.
 */

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

// ── Time-of-day phase ───────────────────────────────────────────────────────

/**
 * Pick a sky phase (dawn/day/sunset/night) from the current local hour and
 * the daily sunrise/sunset times. Falls back to a coarse day/night split
 * when astro times are missing.
 *
 * @param {Date}    now      - Current local time (already adjusted for location)
 * @param {string}  sunrise  - ISO local string e.g. "2026-05-03T06:14"
 * @param {string}  sunset   - ISO local string e.g. "2026-05-03T20:08"
 * @returns {'dawn'|'day'|'sunset'|'night'}
 */
function pickPhase(now, sunrise, sunset) {
  const sr = parseLocalISO(sunrise);
  const ss = parseLocalISO(sunset);
  if (!sr || !ss) return now.getHours() >= 6 && now.getHours() < 20 ? 'day' : 'night';

  const t = now.getTime();
  const ms = 60 * 60 * 1000;
  if (t >= sr.getTime() - ms && t < sr.getTime() + ms) return 'dawn';
  if (t >= ss.getTime() - ms && t < ss.getTime() + ms) return 'sunset';
  if (t >= sr.getTime() && t < ss.getTime()) return 'day';
  return 'night';
}

/** Parse an Open-Meteo local ISO ("YYYY-MM-DDTHH:MM") into a Date. */
function parseLocalISO(s) {
  if (!s) return null;
  const d = new Date(s);
  return isNaN(d.getTime()) ? null : d;
}

// ── Card construction ───────────────────────────────────────────────────────

export function render(payload, synthesis, root) {
  const card = document.createElement('div');
  card.className = 'rich-card weather-card';

  const now = new Date();
  const phase = pickPhase(now, payload.sunrise, payload.sunset);
  card.dataset.phase = phase;

  card.appendChild(buildSky(payload, synthesis, now, phase));

  const rail = buildRail(payload.hourly || [], now);
  if (rail) card.appendChild(rail);

  root.appendChild(card);
}

function buildSky(payload, synthesis, now, phase) {
  const sky = document.createElement('div');
  sky.className = 'weather-card__sky';

  const sun = document.createElement('div');
  sun.className = 'weather-card__sun';
  sky.appendChild(sun);

  for (let i = 1; i <= 2; i++) {
    const cloud = document.createElement('div');
    cloud.className = `weather-card__cloud weather-card__cloud--${i}`;
    sky.appendChild(cloud);
  }

  const skyline = document.createElement('div');
  skyline.className = 'weather-card__skyline';
  sky.appendChild(skyline);

  sky.appendChild(buildOverlay(payload, synthesis, now, phase));
  return sky;
}

function buildOverlay(payload, synthesis, now, phase) {
  const overlay = document.createElement('div');
  overlay.className = 'weather-card__overlay';

  const readout = document.createElement('div');
  readout.className = 'weather-card__readout';

  const temp = document.createElement('div');
  temp.className = 'weather-card__temp';
  const t = Math.round(payload.temperature_c ?? 0);
  temp.innerHTML = `${t}<sup>°</sup>`;
  readout.appendChild(temp);

  const loc = document.createElement('div');
  loc.className = 'weather-card__loc';
  const locLine = document.createElement('div');
  locLine.textContent = payload.location || '—';
  const timeLine = document.createElement('div');
  timeLine.textContent = `${DAYS[now.getDay()]} · ${formatHM(now)}`;
  loc.appendChild(locLine);
  loc.appendChild(timeLine);
  readout.appendChild(loc);

  overlay.appendChild(readout);

  const caption = document.createElement('div');
  caption.className = 'weather-card__caption';
  if (synthesis) {
    caption.innerHTML = synthesis;
  } else {
    caption.textContent = buildFallbackCaption(payload, phase);
  }
  overlay.appendChild(caption);

  return overlay;
}

function buildFallbackCaption(payload, phase) {
  const cond = (payload.condition || '').toLowerCase();
  const feels = payload.feels_like_c == null ? '' : ` Feels like ${Math.round(payload.feels_like_c)}°.`;
  if (phase === 'sunset') return `Golden hour in ${payload.location || 'your area'}.${feels}`;
  if (phase === 'dawn') return `Sun's coming up over ${payload.location || 'your area'}.${feels}`;
  if (phase === 'night') return `Quiet night in ${payload.location || 'your area'}.${feels}`;
  return `${payload.condition || 'Steady weather'} in ${payload.location || 'your area'}.${feels}`;
}

function buildRail(hourly, now) {
  if (!hourly || hourly.length === 0) return null;

  const rail = document.createElement('div');
  rail.className = 'weather-card__rail';

  const temps = hourly.map(h => h.temp_c).filter(t => t != null);
  const min = temps.length ? Math.min(...temps) : 0;
  const max = temps.length ? Math.max(...temps) : 1;
  const span = Math.max(1, max - min);
  const peakTemp = max;
  const currentHour = now.getHours();

  for (const h of hourly) {
    const cell = document.createElement('div');
    cell.className = 'weather-card__hour';
    if (h.hour === currentHour) cell.classList.add('weather-card__hour--cur');
    if (h.temp_c === peakTemp) cell.classList.add('weather-card__hour--peak');

    const hourLabel = document.createElement('span');
    hourLabel.className = 'weather-card__hour-label';
    hourLabel.textContent = String(h.hour);
    cell.appendChild(hourLabel);

    const tempEl = document.createElement('b');
    tempEl.className = 'weather-card__hour-temp';
    tempEl.textContent = `${h.temp_c}°`;
    cell.appendChild(tempEl);

    const bar = document.createElement('div');
    bar.className = 'weather-card__hour-bar';
    const pct = 35 + Math.round(((h.temp_c - min) / span) * 65);
    bar.style.width = `${pct}%`;
    cell.appendChild(bar);

    rail.appendChild(cell);
  }
  return rail;
}

function formatHM(d) {
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
}

/**
 * Weather card module.
 *
 * Implements the rich-media card contract: exports render(payload, synthesis, root).
 * The card adapts to the existing WeatherAbility dict shape — no field renames.
 * Icon key is derived on the frontend from (condition, is_daylight, is_raining, is_clear).
 * Temperature count-up animates on first render (400ms ease-out).
 */

// ── Icon resolution ──────────────────────────────────────────────────────────

const ICON_BASE = new URL('./icons/weather/', import.meta.url).href;

/**
 * Derive the SVG icon filename from condition flags.
 * Order of priority: thunder → snow → rain → fog → clear → partly-cloudy → cloudy.
 *
 * @param {string} condition  - Condition string (e.g. "Partly cloudy", "Heavy rain")
 * @param {boolean} isDaylight
 * @param {boolean} isRaining
 * @param {boolean} isClear
 * @returns {string} SVG file path
 */
function iconPath(condition, isDaylight, isRaining, isClear) {
  const cond = (condition || '').toLowerCase();

  if (cond.includes('thunder')) {
    return ICON_BASE + 'thunderstorm.svg';
  }
  if (cond.includes('snow') || cond.includes('blizzard') || cond.includes('sleet')) {
    return ICON_BASE + 'snow.svg';
  }
  if (isRaining || cond.includes('rain') || cond.includes('drizzle') || cond.includes('shower')) {
    return ICON_BASE + 'rain.svg';
  }
  if (cond.includes('fog') || cond.includes('mist') || cond.includes('haze')) {
    return ICON_BASE + 'fog.svg';
  }
  if (isClear || cond.includes('clear') || cond.includes('sunny') || cond.includes('mainly clear')) {
    return isDaylight ? ICON_BASE + 'clear-day.svg' : ICON_BASE + 'clear-night.svg';
  }
  if (cond.includes('partly')) {
    return isDaylight ? ICON_BASE + 'partly-cloudy-day.svg' : ICON_BASE + 'partly-cloudy-night.svg';
  }
  return ICON_BASE + 'cloudy.svg';
}

// ── Count-up animation ───────────────────────────────────────────────────────

/**
 * Animate a numeric count-up from 0 to target over durationMs.
 *
 * @param {HTMLElement} el       - Element whose textContent is mutated
 * @param {number}      target   - Target numeric value
 * @param {string}      suffix   - Unit suffix (e.g. "°C")
 * @param {number}      [duration=400]
 */
function countUp(el, target, suffix, duration = 400) {
  const startTime = performance.now();
  const startVal = 0;

  function step(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease-out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(startVal + (target - startVal) * eased);
    el.textContent = current + suffix;
    if (progress < 1) {
      requestAnimationFrame(step);
    }
  }

  requestAnimationFrame(step);
}

// ── Card DOM construction ────────────────────────────────────────────────────

/**
 * Format a direction + speed string. Returns "" if wind is falsy.
 */
function windStr(dir, kmh) {
  if (!kmh) return '';
  const speed = Math.round(kmh);
  return dir ? `${dir} ${speed} km/h` : `${speed} km/h`;
}

/**
 * Build and mount the weather card DOM into root.
 *
 * @param {object}      payload   - Parsed WeatherAbility dict
 * @param {string}      synthesis - LLM synthesis text (displayed at bottom)
 * @param {HTMLElement} root      - Container element supplied by registry caller
 */
export function render(payload, synthesis, root) {
  const card = document.createElement('div');
  card.className = 'rich-card weather-card';

  // ── Location ───────────────────────────────────────────────────────────────
  const locationEl = document.createElement('div');
  locationEl.className = 'weather-card__location';
  locationEl.textContent = payload.location || '';
  card.appendChild(locationEl);

  // ── Main row: icon + temp + condition ─────────────────────────────────────
  const mainEl = document.createElement('div');
  mainEl.className = 'weather-card__main';

  const iconEl = document.createElement('div');
  iconEl.className = 'weather-card__icon';
  const img = document.createElement('img');
  img.src = iconPath(payload.condition, payload.is_daylight, payload.is_raining, payload.is_clear);
  img.alt = payload.condition || 'Weather icon';
  iconEl.appendChild(img);
  mainEl.appendChild(iconEl);

  const tempBlock = document.createElement('div');
  tempBlock.className = 'weather-card__temp-block';

  const tempEl = document.createElement('div');
  tempEl.className = 'weather-card__temp';
  const tempTarget = Math.round(payload.temperature_c ?? 0);
  tempEl.textContent = '0°C';
  tempBlock.appendChild(tempEl);

  const condEl = document.createElement('div');
  condEl.className = 'weather-card__condition';
  condEl.textContent = payload.condition || '';
  tempBlock.appendChild(condEl);

  mainEl.appendChild(tempBlock);
  card.appendChild(mainEl);

  // ── Detail pills ──────────────────────────────────────────────────────────
  const details = [];
  if (payload.feels_like_c != null) {
    details.push(`Feels ${Math.round(payload.feels_like_c)}°`);
  }
  if (payload.humidity_pct != null) {
    details.push(`${payload.humidity_pct}%`);
  }
  const wind = windStr(payload.wind_direction, payload.wind_kmh);
  if (wind) details.push(wind);

  if (details.length > 0) {
    const detailsEl = document.createElement('div');
    detailsEl.className = 'weather-card__details';
    for (const text of details) {
      const pill = document.createElement('span');
      pill.className = 'weather-card__detail-pill';
      pill.textContent = text;
      detailsEl.appendChild(pill);
    }
    card.appendChild(detailsEl);
  }

  // ── Forecast ──────────────────────────────────────────────────────────────
  const hasForecast = payload.forecast_tomorrow_condition
    || payload.forecast_tomorrow_max_c != null
    || payload.forecast_tomorrow_min_c != null;

  if (hasForecast) {
    const divider = document.createElement('div');
    divider.className = 'rich-card__divider';
    card.appendChild(divider);

    const forecastEl = document.createElement('div');
    forecastEl.className = 'weather-card__forecast';

    const label = document.createElement('span');
    label.className = 'weather-card__forecast-label';
    label.textContent = 'Tomorrow';
    forecastEl.appendChild(label);

    if (payload.forecast_tomorrow_max_c != null || payload.forecast_tomorrow_min_c != null) {
      const tempRange = document.createElement('span');
      tempRange.className = 'weather-card__forecast-temp';
      const hi = payload.forecast_tomorrow_max_c != null ? Math.round(payload.forecast_tomorrow_max_c) : '–';
      const lo = payload.forecast_tomorrow_min_c != null ? Math.round(payload.forecast_tomorrow_min_c) : '–';
      tempRange.textContent = `${hi}°/${lo}°`;
      forecastEl.appendChild(tempRange);
    }

    if (payload.forecast_tomorrow_condition) {
      const cond = document.createElement('span');
      cond.className = 'weather-card__forecast-condition';
      cond.textContent = payload.forecast_tomorrow_condition;
      forecastEl.appendChild(cond);
    }

    if (payload.forecast_tomorrow_precip_chance_pct != null) {
      const rain = document.createElement('span');
      rain.className = 'weather-card__forecast-rain';
      rain.textContent = `${payload.forecast_tomorrow_precip_chance_pct}%`;
      forecastEl.appendChild(rain);
    }

    card.appendChild(forecastEl);
  }

  // ── Synthesis (LLM interpretation) ───────────────────────────────────────
  if (synthesis) {
    const synDivider = document.createElement('div');
    synDivider.className = 'rich-card__divider';
    card.appendChild(synDivider);

    const synEl = document.createElement('div');
    synEl.className = 'rich-card__synthesis';
    synEl.textContent = synthesis;
    card.appendChild(synEl);
  }

  root.appendChild(card);

  // Start count-up after mount (rAF ensures layout is done)
  requestAnimationFrame(() => {
    countUp(tempEl, tempTarget, '°C');
  });
}

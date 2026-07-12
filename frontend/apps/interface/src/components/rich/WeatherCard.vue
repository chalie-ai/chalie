<script setup lang="ts">
import { computed } from 'vue';
import { renderMarkup } from '../../composables/useMarkup';
import { DAYS, parseDate, formatClock } from '../../utils/time';

export interface WeatherPayload {
  location?: string;
  temperature_c?: number;
  feels_like_c?: number;
  condition?: string;
  sunrise?: string;
  sunset?: string;
  hourly?: Array<{
    hour: number;
    temp_c: number;
  }>;
}

const props = defineProps<{ payload: WeatherPayload; synthesis?: string }>();

const now = new Date();

/** Sky phase from local hour + sunrise/sunset; coarse day/night when astro times missing. */
function pickPhase(
  now: Date,
  sunrise: string | undefined,
  sunset: string | undefined,
): 'dawn' | 'day' | 'sunset' | 'night' {
  const sr = parseDate(sunrise);
  const ss = parseDate(sunset);
  if (!sr || !ss) {
    return now.getHours() >= 6 && now.getHours() < 20 ? 'day' : 'night';
  }

  const t = now.getTime();
  const ms = 60 * 60 * 1000;
  if (t >= sr.getTime() - ms && t < sr.getTime() + ms) return 'dawn';
  if (t >= ss.getTime() - ms && t < ss.getTime() + ms) return 'sunset';
  if (t >= sr.getTime() && t < ss.getTime()) return 'day';
  return 'night';
}

const phase = computed<'dawn' | 'day' | 'sunset' | 'night'>(() =>
  pickPhase(now, props.payload.sunrise, props.payload.sunset),
);

const roundedTemp = computed<number>(() => Math.round(props.payload.temperature_c ?? 0));

const dayLabel = computed<string>(() => DAYS[now.getDay()]);

const timeLabel = computed<string>(() => formatClock(now));

const captionHtml = computed<string>(() => (props.synthesis ? renderMarkup(props.synthesis) : ''));

const fallbackCaption = computed<string>(() => {
  if (props.synthesis) return '';
  const loc = props.payload.location || 'your area';
  const feels =
    props.payload.feels_like_c == null
      ? ''
      : ` Feels like ${Math.round(props.payload.feels_like_c)}°.`;
  if (phase.value === 'sunset') return `Golden hour in ${loc}.${feels}`;
  if (phase.value === 'dawn') return `Sun's coming up over ${loc}.${feels}`;
  if (phase.value === 'night') return `Quiet night in ${loc}.${feels}`;
  return `${props.payload.condition || 'Steady weather'} in ${loc}.${feels}`;
});

interface HourCell {
  hour: number;
  temp_c: number;
  isCurrent: boolean;
  isPeak: boolean;
  barWidth: number;
}

const hourCells = computed<HourCell[]>(() => {
  const hourly = props.payload.hourly;
  if (!hourly || hourly.length === 0) return [];

  const temps = hourly.map((h) => h.temp_c).filter((t) => t != null);
  const min = temps.length ? Math.min(...temps) : 0;
  const max = temps.length ? Math.max(...temps) : 1;
  const span = Math.max(1, max - min);
  const currentHour = now.getHours();

  return hourly.map((h) => ({
    hour: h.hour,
    temp_c: h.temp_c,
    isCurrent: h.hour === currentHour,
    isPeak: h.temp_c === max,
    barWidth: 35 + Math.round(((h.temp_c - min) / span) * 65),
  }));
});
</script>

<template>
  <div class="rich-card weather-card" :data-phase="phase">
    <div class="weather-card__sky">
      <div class="weather-card__sun" aria-hidden="true" />

      <div class="weather-card__cloud weather-card__cloud--1" aria-hidden="true" />
      <div class="weather-card__cloud weather-card__cloud--2" aria-hidden="true" />

      <div class="weather-card__skyline" aria-hidden="true" />

      <div class="weather-card__overlay">
        <div class="weather-card__readout">
          <div class="weather-card__temp">{{ roundedTemp }}<sup>°</sup></div>

          <div class="weather-card__loc">
            <div>{{ payload.location || '—' }}</div>
            <div>{{ dayLabel }} · {{ timeLabel }}</div>
          </div>
        </div>

        <!-- Caption: synthesis (HTML) or fallback plain text -->
        <div v-if="synthesis" class="weather-card__caption" v-html="captionHtml" />
        <div v-else class="weather-card__caption">{{ fallbackCaption }}</div>
      </div>
    </div>

    <div v-if="hourCells.length" class="weather-card__rail">
      <div
        v-for="cell in hourCells"
        :key="cell.hour"
        class="weather-card__hour"
        :class="{
          'weather-card__hour--cur': cell.isCurrent,
          'weather-card__hour--peak': cell.isPeak,
        }"
      >
        <b class="weather-card__hour-temp">{{ cell.temp_c }}°</b>
        <div class="weather-card__hour-track" aria-hidden="true">
          <div class="weather-card__hour-bar" :style="{ height: cell.barWidth * 0.28 + 'px' }" />
        </div>
        <span class="weather-card__hour-label">{{ cell.hour }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
/* Card-specific rules only; base .rich-card chrome lives globally in base_card.css. */

/* padding:0 — the sky canvas IS the card body. */
.rich-card.weather-card {
  padding: 0;
  overflow: hidden;
  width: 100%;
  max-width: 100%;
  background: transparent;
  border: 1px solid var(--border, color-mix(in oklab, var(--violet) 22%, transparent));
}

.weather-card__sky {
  position: relative;
  min-height: 220px;
  overflow: hidden;
  isolation: isolate;
}

/* Phase-driven gradients. Fallback (no phase) is night. */
.weather-card[data-phase='day'] .weather-card__sky {
  background: linear-gradient(180deg, #4a8cd6 0%, #6ea7e0 40%, #a9c8eb 80%, #f0d59f 100%);
}
.weather-card[data-phase='dawn'] .weather-card__sky {
  background: linear-gradient(180deg, #1a2240 0%, #4a4070 35%, #c87a5a 75%, #ffc97a 100%);
}
.weather-card[data-phase='sunset'] .weather-card__sky {
  background: linear-gradient(180deg, #1a2240 0%, #2a3560 35%, #c75a3f 75%, #ffb070 100%);
}
.weather-card[data-phase='night'] .weather-card__sky {
  background: linear-gradient(180deg, #06080e 0%, #121a30 50%, #1f2a4a 100%);
}

.weather-card__sky::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 60% 40% at 70% 30%, rgba(255, 200, 140, 0.3), transparent 60%),
    radial-gradient(ellipse 80% 50% at 20% 85%, rgba(138, 92, 255, 0.2), transparent 60%);
  pointer-events: none;
}

.weather-card__sun {
  position: absolute;
  right: 14%;
  top: 38%;
  width: 90px;
  height: 90px;
  border-radius: 50%;
  filter: blur(2px);
  opacity: 0.9;
  animation: weather-sun-bob 12s ease-in-out infinite alternate;
  pointer-events: none;

  .weather-card[data-phase='day'] & {
    background: radial-gradient(circle, #fff4cc 0%, #ffd972 60%, transparent 80%);
  }

  .weather-card[data-phase='dawn'] &,
  .weather-card[data-phase='sunset'] & {
    background: radial-gradient(circle, #ffd28c 0%, #ff9b50 60%, transparent 80%);
  }

  .weather-card[data-phase='night'] & {
    width: 70px;
    height: 70px;
    background: radial-gradient(circle, #f4f1ff 0%, #b8b3d6 55%, transparent 80%);
    filter: blur(1px);
    opacity: 0.85;
  }
}

@keyframes weather-sun-bob {
  from {
    transform: translateY(0);
  }
  to {
    transform: translateY(8px);
  }
}

.weather-card__cloud {
  position: absolute;
  background: rgba(255, 255, 255, 0.18);
  filter: blur(14px);
  border-radius: 50%;
  animation: weather-cloud-drift 30s linear infinite;
  pointer-events: none;

  &--1 {
    width: 220px;
    height: 60px;
    top: 30%;
    left: -25%;
    animation-duration: 38s;
  }

  &--2 {
    width: 160px;
    height: 48px;
    top: 50%;
    left: -25%;
    animation-duration: 26s;
    animation-delay: -10s;
    opacity: 0.55;
  }
}

@keyframes weather-cloud-drift {
  from {
    transform: translateX(0);
  }
  to {
    transform: translateX(160vw);
  }
}

.weather-card__skyline {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 36px;
  background: linear-gradient(to top, #06080e 0%, #06080e 30%, transparent 100%);
  pointer-events: none;

  &::before {
    content: '';
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 28px;
    background-image: linear-gradient(
      to right,
      transparent 5%,
      #0a0d18 5%,
      #0a0d18 8%,
      transparent 8%,
      transparent 14%,
      #0a0d18 14%,
      #0a0d18 18%,
      transparent 18%,
      transparent 24%,
      #0a0d18 24%,
      #0a0d18 30%,
      transparent 30%,
      transparent 38%,
      #0a0d18 38%,
      #0a0d18 41%,
      transparent 41%,
      transparent 50%,
      #0a0d18 50%,
      #0a0d18 56%,
      transparent 56%,
      transparent 64%,
      #0a0d18 64%,
      #0a0d18 68%,
      transparent 68%,
      transparent 76%,
      #0a0d18 76%,
      #0a0d18 82%,
      transparent 82%,
      transparent 90%,
      #0a0d18 90%,
      #0a0d18 95%,
      transparent 95%
    );
    background-size: 100% 100%;
    background-position: bottom;
    background-repeat: no-repeat;
    mask-image: linear-gradient(to top, black 60%, transparent 100%);
    -webkit-mask-image: linear-gradient(to top, black 60%, transparent 100%);
  }
}

/* In normal flow (not absolute) so the sky grows to fit a long caption instead
   of clipping it. min-height keeps the art visible when the caption is short. */
.weather-card__overlay {
  position: relative;
  min-height: 220px;
  display: grid;
  grid-template-rows: 1fr auto;
  padding: 18px 22px;
  color: white;
  z-index: 1;
}

.weather-card__readout {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.weather-card__temp {
  font-size: 3.75rem;
  font-weight: 250;
  letter-spacing: -0.03em;
  line-height: 0.9;
  font-variant-numeric: tabular-nums;

  sup {
    font-size: 1.625rem;
    font-weight: 300;
    vertical-align: super;
  }
}

.weather-card__loc {
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.75rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-weight: 600;
  text-align: right;
  margin-top: 4px;

  div:last-child {
    font-size: 0.6875rem;
    font-weight: 400;
    letter-spacing: 0.06em;
    opacity: 0.82;
    margin-top: 3px;
  }
}

.weather-card__caption {
  align-self: end;
  max-width: 62%;
  font-size: 0.844rem;
  line-height: 1.55;
  color: rgba(255, 255, 255, 0.94);
  text-shadow: 0 1px 8px rgba(0, 0, 0, 0.18);
}

/* Hourly strip is a neutral DATA surface, not part of the immersive sky — it
   flips with the theme (was a hard-coded near-black rgba band that read as
   "black blocks" over a light/day sky). Mirrors the mockup's `.whours`:
   `--surface` ground, `--border` rules, accent bars. */
.weather-card__rail {
  display: grid;
  grid-template-columns: repeat(8, 1fr);
  background: var(--bg-surface-2);
  border-top: 1px solid var(--border);
}

.weather-card__hour {
  padding: 12px 0 11px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  text-align: center;
  border-right: 1px solid var(--border);

  &:last-child {
    border-right: none;
  }

  &--cur {
    background: color-mix(in oklab, var(--accent-primary) 8%, transparent);

    .weather-card__hour-temp {
      color: var(--accent-primary);
    }
  }

  &--peak .weather-card__hour-temp {
    color: var(--accent-primary);
  }
}

.weather-card__hour-temp {
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.75rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--text-primary);
  letter-spacing: 0;
}

/* Vertical mini-bar: a fixed-height track with the bar bottom-aligned so all
   bars share one baseline; height scales with the hour's temperature (bound
   inline off barWidth). Peak/current hours brighten for emphasis. */
.weather-card__hour-track {
  height: 30px;
  display: flex;
  align-items: flex-end;
}

.weather-card__hour-bar {
  width: 5px;
  min-height: 3px;
  border-radius: 3px;
  background: var(--accent-primary);
  opacity: 0.28;
  transition: opacity 200ms ease;
}

.weather-card__hour--cur .weather-card__hour-bar {
  opacity: 0.55;
}

.weather-card__hour--peak .weather-card__hour-bar {
  opacity: 0.9;
}

.weather-card__hour-label {
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.656rem;
  color: var(--text-tertiary);
  font-variant-numeric: tabular-nums;
}
</style>

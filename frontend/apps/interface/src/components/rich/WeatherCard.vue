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

/** Time-of-day phase from local hour + sunrise/sunset; coarse day/night when astro times missing. */
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
  <div class="rich-card weather-card">
    <div class="weather-card__body">
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

.rich-card.weather-card {
  padding: 0;
  overflow: hidden;
  width: 100%;
  max-width: 100%;
}

.weather-card__body {
  min-height: 220px;
  display: grid;
  grid-template-rows: 1fr auto;
  padding: 18px 22px;
  color: var(--text);
}

.weather-card__readout {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.weather-card__temp {
  font-family: var(--font-display);
  font-size: 3.75rem;
  font-weight: 900;
  letter-spacing: -0.03em;
  line-height: 0.9;
  font-variant-numeric: tabular-nums;

  sup {
    font-size: 1.625rem;
    vertical-align: super;
  }
}

.weather-card__loc {
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-weight: 600;
  text-align: right;
  margin-top: 4px;

  div:last-child {
    font-size: var(--fs-mono);
    font-weight: 400;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin-top: 3px;
  }
}

.weather-card__caption {
  align-self: end;
  max-width: 62%;
  font-size: var(--fs-body);
  line-height: 1.55;
  color: var(--text);
}

.weather-card__rail {
  display: grid;
  grid-template-columns: repeat(8, 1fr);
  background: var(--surface-2);
}

.weather-card__hour {
  padding: 12px 0 11px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  text-align: center;

  &--cur {
    background: var(--cell);

    .weather-card__hour-temp {
      color: var(--pink-text);
    }
  }

  &--peak .weather-card__hour-temp {
    color: var(--pink-text);
  }
}

.weather-card__hour-temp {
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--text);
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
  background: var(--control);
}

.weather-card__hour--cur .weather-card__hour-bar {
  background: var(--muted);
}

.weather-card__hour--peak .weather-card__hour-bar {
  background: var(--pink);
}

.weather-card__hour-label {
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
</style>

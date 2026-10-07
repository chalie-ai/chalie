<script setup lang="ts">
import { computed } from 'vue';
import { renderMarkup } from '../../composables/useMarkup';
import { DAYS, parseDate, formatClock } from '../../utils/time';
import {
  Cloud,
  CloudDrizzle,
  CloudFog,
  CloudLightning,
  CloudMoon,
  CloudRain,
  CloudSnow,
  CloudSun,
  Droplets,
  Moon,
  Sun,
  Umbrella,
  Wind,
  type LucideIcon,
} from '@lucide/vue';

export interface WeatherPayload {
  location?: string;
  temperature_c?: number;
  feels_like_c?: number;
  condition?: string;
  humidity_pct?: number;
  wind_kmh?: number;
  wind_direction?: string;
  uv_index?: number | null;
  is_daylight?: boolean;
  forecast_tomorrow_precip_chance_pct?: number;
  sunrise?: string;
  sunset?: string;
  hourly?: Array<{
    hour: number;
    temp_c: number;
    code?: number;
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

/** The provider's is_daylight wins; when it is missing the local phase decides. */
const isNight = computed<boolean>(() => {
  const daylight = props.payload.is_daylight;
  return daylight != null ? !daylight : phase.value === 'night';
});

const roundedTemp = computed<number>(() => Math.round(props.payload.temperature_c ?? 0));

const dayLabel = computed<string>(() => DAYS[now.getDay()]);

const timeLabel = computed<string>(() => formatClock(now));

/** Mono line under the numeral: "<condition> · feels <n>°", the feels part
    dropping out when feels_like_c is missing. */
const subline = computed<string>(() => {
  const feels =
    props.payload.feels_like_c == null
      ? null
      : `feels ${Math.round(props.payload.feels_like_c)}°`;
  return [props.payload.condition, feels].filter(Boolean).join(' · ');
});

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

/**
 * Provider condition free-text → glyph, by keyword. Thunder wins over the rain
 * or snow it comes with, "Mainly clear" over plain "clear" and "Partly cloudy"
 * over "cloud"; snow is checked before rain so "snow showers" lands on the snow
 * glyph (matching WMO 85/86).
 */
const conditionIcon = computed<LucideIcon>(() => {
  const t = (props.payload.condition || '').toLowerCase();
  if (t.includes('thunder')) return CloudLightning;
  if (t.includes('mainly clear') || t.includes('partly')) return isNight.value ? CloudMoon : CloudSun;
  if (t.includes('clear') || t.includes('sunny')) return isNight.value ? Moon : Sun;
  if (t.includes('overcast') || t.includes('cloud')) return Cloud;
  if (t.includes('fog') || t.includes('mist')) return CloudFog;
  if (t.includes('snow') || t.includes('sleet')) return CloudSnow;
  if (t.includes('drizzle')) return CloudDrizzle;
  if (t.includes('rain') || t.includes('shower')) return CloudRain;
  return Cloud;
});

/** Minute of the day of a location-local "YYYY-MM-DDTHH:MM" sunrise/sunset — the
    same clock as the hourly rail's hours — or the fallback when the provider sent none. */
function astroMinute(s: string | undefined, fallbackHour: number): number {
  const m = /T(\d{2}):(\d{2})/.exec(s ?? '');
  return m ? Number(m[1]) * 60 + Number(m[2]) : fallbackHour * 60;
}

/** WMO code → glyph for the hourly rail, moon-lit outside daylight; null means no code, no glyph. */
function wmoGlyph(code: number | undefined, night: boolean): LucideIcon | null {
  if (code == null) return null;
  if (code === 0) return night ? Moon : Sun;
  if (code === 1 || code === 2) return night ? CloudMoon : CloudSun;
  if (code === 3) return Cloud;
  if (code === 45 || code === 48) return CloudFog;
  if (code >= 51 && code <= 57) return CloudDrizzle;
  if ((code >= 61 && code <= 67) || (code >= 80 && code <= 82)) return CloudRain;
  if ((code >= 71 && code <= 77) || code === 85 || code === 86) return CloudSnow;
  if (code >= 95 && code <= 99) return CloudLightning;
  return Cloud;
}

interface HourCell {
  hour: number;
  temp_c: number;
  isCurrent: boolean;
  isPeak: boolean;
  icon: LucideIcon | null;
}

const hourCells = computed<HourCell[]>(() => {
  const hourly = props.payload.hourly;
  if (!hourly || hourly.length === 0) return [];

  const max = Math.max(...hourly.map((h) => h.temp_c));
  const currentHour = now.getHours();
  // An hour reads as night when it starts before sunrise or at/after sunset.
  const rise = astroMinute(props.payload.sunrise, 6);
  const set = astroMinute(props.payload.sunset, 20);

  return hourly.map((h) => ({
    hour: h.hour,
    temp_c: h.temp_c,
    isCurrent: h.hour === currentHour,
    isPeak: h.temp_c === max,
    icon: wmoGlyph(h.code, h.hour * 60 < rise || h.hour * 60 >= set),
  }));
});

/** Stat chips, in display order — each only when its field is present. */
const chips = computed<Array<{ icon: LucideIcon; label: string }>>(() => {
  const p = props.payload;
  const out: Array<{ icon: LucideIcon; label: string }> = [];
  if (p.wind_kmh != null) {
    out.push({
      icon: Wind,
      label: `${p.wind_kmh} km/h${p.wind_direction ? ` ${p.wind_direction}` : ''}`,
    });
  }
  if (p.humidity_pct != null) out.push({ icon: Droplets, label: `${p.humidity_pct}%` });
  if (p.uv_index != null) out.push({ icon: Sun, label: `UV ${p.uv_index}` });
  if (p.forecast_tomorrow_precip_chance_pct != null) {
    out.push({ icon: Umbrella, label: `${p.forecast_tomorrow_precip_chance_pct}% tmrw` });
  }
  return out;
});

/** Art glyph motion: the Sun turns slowly, cloud-family glyphs drift, the Moon sits still. */
const artMotion = computed<'spin' | 'drift' | ''>(() => {
  if (conditionIcon.value === Sun) return 'spin';
  if (conditionIcon.value === Moon) return '';
  return 'drift';
});
</script>

<template>
  <div class="rich-card weather-card">
    <div class="weather-card__body" :class="{ 'weather-card__body--night': isNight }">
      <div class="weather-card__readout">
        <div class="weather-card__temp-col">
          <div class="weather-card__temp">{{ roundedTemp }}<sup>°</sup></div>
          <div v-if="subline" class="weather-card__subline">{{ subline }}</div>
        </div>

        <div class="weather-card__loc">
          <div>{{ payload.location || '—' }}</div>
          <div>{{ dayLabel }} · {{ timeLabel }}</div>
        </div>
      </div>

      <!-- Caption: synthesis (HTML) or fallback plain text -->
      <div v-if="synthesis" class="weather-card__caption" v-html="captionHtml" />
      <div v-else class="weather-card__caption">{{ fallbackCaption }}</div>

      <!-- Stat chips: each only when its field is present -->
      <div v-if="chips.length" class="weather-card__chips">
        <span v-for="chip in chips" :key="chip.label" class="weather-card__chip">
          <component :is="chip.icon" :size="14" aria-hidden="true" />
          {{ chip.label }}
        </span>
      </div>

      <!-- Oversized condition glyph as corner artwork, cropped off the body -->
      <component
        :is="conditionIcon"
        class="weather-card__art"
        :class="{
          'weather-card__art--spin': artMotion === 'spin',
          'weather-card__art--drift': artMotion === 'drift',
        }"
        :size="320"
        :stroke-width="1"
        aria-hidden="true"
      />
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
        <component
          :is="cell.icon"
          v-if="cell.icon"
          class="weather-card__hour-icon"
          :size="18"
          aria-hidden="true"
        />
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
  margin-bottom: var(--space-md);
}

/* Tinted ground: pink over --surface by day, deeper plum (pink over --bg) at night. */
.weather-card__body {
  position: relative;
  overflow: hidden;
  min-height: 220px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 18px 22px;
  color: var(--text);
  background: color-mix(in oklab, var(--pink) 9%, var(--surface));
}

.weather-card__body--night {
  background: color-mix(in oklab, var(--pink) 7%, var(--bg));
}

/* Readout, chips and caption sit above the corner artwork. */
.weather-card__readout,
.weather-card__chips,
.weather-card__caption {
  position: relative;
  z-index: 1;
}

.weather-card__readout {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.weather-card__temp-col {
  display: flex;
  flex-direction: column;
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

.weather-card__subline {
  margin-top: 10px;
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
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

.weather-card__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.weather-card__chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 9px;
  background: var(--cell);
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  color: var(--text);

  svg {
    color: var(--muted);
  }
}

.weather-card__caption {
  max-width: 62%;
  font-size: var(--fs-body);
  line-height: 1.55;
  color: var(--text);
}

/* Oversized condition glyph as artwork, cropped off the bottom-right corner. */
.weather-card__art {
  position: absolute;
  right: -90px;
  bottom: -120px;
  z-index: 0;
  color: color-mix(in oklab, var(--pink) 30%, transparent);
  pointer-events: none;
}

/* Only the Sun turns — with the shared `spin` keyframe from the global sheet. */
.weather-card__art--spin {
  animation: spin 60s linear infinite;
}

/* Cloud-family glyphs drift a few px instead. */
.weather-card__art--drift {
  animation: weather-card-drift 8s ease-in-out infinite alternate;
}

@keyframes weather-card-drift {
  from {
    translate: 0 0;
  }
  to {
    translate: 8px -4px;
  }
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

    .weather-card__hour-temp,
    .weather-card__hour-icon {
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

.weather-card__hour-icon {
  color: var(--muted);
}

.weather-card__hour-label {
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
</style>

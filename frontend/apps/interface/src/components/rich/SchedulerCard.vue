<script setup lang="ts">
import { computed } from 'vue';
import { formatCadence } from '../../utils/time';

export interface SchedulerRecord {
  id: number;
  message: string;
  /** Localized (user-timezone) ISO string — the schedule's activation floor. */
  start_at: string;
  /** Cron fields (`null` = "every" on that field). */
  day?: number | null;
  hour?: number | null;
  minute?: number | null;
}

export interface SchedulerPayload {
  record: SchedulerRecord;
}

const props = defineProps<{
  payload: SchedulerPayload;
  synthesis?: string;
}>();

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'] as const;
const MONTHS = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec',
] as const;

// The backend already localizes start_at to the user's timezone — parse and
// read the components directly, no client-side offset math.
function parseLocal(isoStr: string | null | undefined): Date | null {
  if (!isoStr) return null;
  const d = new Date(isoStr);
  return Number.isNaN(d.getTime()) ? null : d;
}

function formatTime(d: Date): string {
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
}

const record = computed(() => props.payload.record);

const startAt = computed(() => parseLocal(record.value.start_at));

const whenDay = computed(() => (startAt.value ? DAYS[startAt.value.getDay()] : '—'));
const whenDate = computed(() => (startAt.value ? String(startAt.value.getDate()) : '—'));
const whenMon = computed(() => (startAt.value ? MONTHS[startAt.value.getMonth()] : ''));

const title = computed(() => record.value.message || props.synthesis || '');

const metaTime = computed(() => (startAt.value ? formatTime(startAt.value) : null));

/** Cadence label, only when the record actually carries cron fields. */
const cadenceText = computed((): string | null => {
  const r = record.value;
  if (r.day === undefined && r.hour === undefined && r.minute === undefined) return null;
  return formatCadence(r.day ?? null, r.hour ?? null, r.minute ?? null);
});

const metaText = computed((): string => {
  if (!cadenceText.value) return '';
  return (metaTime.value ? ' · ' : '') + cadenceText.value;
});
</script>

<template>
  <div class="rich-card scheduler-card">
    <div class="scheduler-card__when">
      <div class="scheduler-card__when-day">{{ whenDay }}</div>
      <div class="scheduler-card__when-date">{{ whenDate }}</div>
      <div class="scheduler-card__when-mon">{{ whenMon }}</div>
    </div>

    <div class="scheduler-card__info">
      <h4 class="scheduler-card__title">{{ title }}</h4>
      <div class="scheduler-card__meta">
        <b v-if="metaTime">{{ metaTime }}</b>
        <template v-if="metaText">{{ metaText }}</template>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
// Card-specific overrides only; base .rich-card rules are global.

.rich-card.scheduler-card {
  max-width: 620px;
}

.scheduler-card {
  display: grid;
  grid-template-columns: 52px 1fr;
  gap: 16px;
  align-items: center;
}

.scheduler-card__when {
  border-radius: 10px;
  background: color-mix(in oklab, var(--violet) 12%, transparent);
  border: 1px solid color-mix(in oklab, var(--violet) 26%, transparent);
  padding: 6px 4px;
  text-align: center;
}

.scheduler-card__when-day {
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.58rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: color-mix(in oklab, var(--violet) 80%, var(--text-secondary));
}

.scheduler-card__when-date {
  font-size: 1.35rem;
  font-weight: 500;
  letter-spacing: -0.03em;
  line-height: 1;
  margin: 2px 0;
  color: var(--text-primary);
}

.scheduler-card__when-mon {
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.58rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-tertiary);
}

.scheduler-card__title {
  font-size: 0.96rem;
  font-weight: 500;
  letter-spacing: -0.005em;
  margin: 0 0 2px;
  color: var(--text-primary);
}

.scheduler-card__meta {
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.74rem;
  color: var(--text-tertiary);
  letter-spacing: 0.04em;

  b {
    color: var(--text-secondary);
    font-weight: 500;
  }
}

</style>

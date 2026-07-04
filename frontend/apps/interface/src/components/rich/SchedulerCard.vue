<script setup lang="ts">
import { computed } from 'vue';

export interface SchedulerRecord {
  id: string | number;
  message: string;
  /**
   * The create record emits `due_at_local` (+`due_at_utc`), NOT a bare `due_at`;
   * legacy read `record.due_at` so the date block always rendered "—". Reading
   * this field instead is an intentional bug-fix divergence.
   */
  due_at_local: string | null;
  due_at_utc?: string | null;
  item_type?: string | null;
  recurrence?: string | null;
  /** Present only on the already-existed dedup path (omits item_type/recurrence). */
  note?: string;
}

export interface SchedulerSameDayItem {
  id: string | number;
  message: string;
  due_at: string | null;
  recurrence: string | null;
}

export interface SchedulerPayload {
  record: SchedulerRecord;
  same_day_items?: SchedulerSameDayItem[];
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

function parseDueAt(dueAtStr: string | null | undefined): Date | null {
  if (!dueAtStr) return null;
  const d = new Date(dueAtStr);
  return Number.isNaN(d.getTime()) ? null : d;
}

function formatTime(d: Date): string {
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
}

const record = computed(() => props.payload.record);

// due_at_local is the field the create card actually carries (see its doc-comment).
const dueAt = computed(() => parseDueAt(record.value.due_at_local));

const whenDay = computed(() => (dueAt.value ? DAYS[dueAt.value.getDay()] : '—'));
const whenDate = computed(() => (dueAt.value ? String(dueAt.value.getDate()) : '—'));
const whenMon = computed(() => (dueAt.value ? MONTHS[dueAt.value.getMonth()] : ''));

const title = computed(() => record.value.message || props.synthesis || '');

const metaTime = computed(() => (dueAt.value ? formatTime(dueAt.value) : null));

const metaText = computed((): string => {
  const textParts: string[] = [];
  if (record.value.recurrence) textParts.push(record.value.recurrence);
  if (record.value.item_type === 'prompt') textParts.push('prompt');
  if (textParts.length === 0) return '';
  return (dueAt.value ? ' · ' : '') + textParts.join(' · ');
});

const sameDay = computed(() =>
  Array.isArray(props.payload.same_day_items) ? props.payload.same_day_items : [],
);
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

    <div v-if="sameDay.length > 0" class="scheduler-card__same-day">
      <div class="scheduler-card__same-day-label">Also on this day · {{ sameDay.length }}</div>
      <div v-for="item in sameDay" :key="item.id" class="scheduler-card__same-day-item">
        <span class="scheduler-card__same-day-text">{{ item.message || '' }}</span>
        <span v-if="parseDueAt(item.due_at)" class="scheduler-card__same-day-time">{{
          formatTime(parseDueAt(item.due_at)!)
        }}</span>
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

.scheduler-card__same-day {
  grid-column: 1 / -1;
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid color-mix(in oklab, var(--border) 60%, transparent);
}

.scheduler-card__same-day-label {
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.62rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-tertiary);
  margin-bottom: 8px;
}

.scheduler-card__same-day-item {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  padding: 5px 0;
  font-size: 0.88rem;
  color: var(--text-secondary);
  border-bottom: 1px solid color-mix(in oklab, var(--border) 35%, transparent);

  &:last-child {
    border-bottom: none;
  }
}

.scheduler-card__same-day-text {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.scheduler-card__same-day-time {
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.72rem;
  color: var(--text-tertiary);
  letter-spacing: 0.04em;
  flex-shrink: 0;
}
</style>

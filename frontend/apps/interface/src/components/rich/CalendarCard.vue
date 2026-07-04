<script setup lang="ts">
import { computed } from 'vue';

export interface CalendarEvent {
  dtstart?: string;
  dtend?: string;
  title?: string;
  all_day?: boolean;
  calendar_name?: string;
  location?: string;
  attendees?: string[];
}

export interface CalendarPayload {
  event?: CalendarEvent;
  events?: CalendarEvent[];
  count?: number;
  action_performed?: string;
}

const props = defineProps<{ payload: CalendarPayload; synthesis?: string }>();

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

function parseISO(s: string | undefined | null): Date | null {
  if (!s) return null;
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
}

function formatTime(d: Date): string {
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
}

function dateKey(d: Date): string {
  return `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
}

/** The single event to render, whether from payload.event or a 1-item list. */
const singleEvent = computed<CalendarEvent | null>(() => {
  if (props.payload.event) return props.payload.event;
  const evs = props.payload.events;
  return Array.isArray(evs) && evs.length === 1 ? evs[0] : null;
});

interface WhenBlock {
  day: string;
  date: number | string;
  mon: string;
}

const whenBlock = computed<WhenBlock>(() => {
  const ev = singleEvent.value;
  const d = ev ? parseISO(ev.dtstart) : null;
  return {
    day: d ? DAYS[d.getDay()] : '—',
    date: d ? d.getDate() : '—',
    mon: d ? MONTHS[d.getMonth()] : '',
  };
});

interface SingleMeta {
  timePart: string | null;
  calendarName: string | null;
}

const singleMeta = computed<SingleMeta>(() => {
  const ev = singleEvent.value;
  if (!ev) return { timePart: null, calendarName: null };

  const dtstart = ev.all_day ? null : parseISO(ev.dtstart);
  const dtend = dtstart && parseISO(ev.dtend);
  let timePart: string | null = ev.all_day ? 'All day' : null;
  if (dtstart) {
    timePart = dtend ? `${formatTime(dtstart)} – ${formatTime(dtend)}` : formatTime(dtstart);
  }

  return { timePart, calendarName: ev.calendar_name ?? null };
});

interface ListRow {
  timeText: string;
  title: string;
  location: string | null;
}

interface DayGroup {
  key: string;
  label: string | null;
  rows: ListRow[];
}

const dayGroups = computed<DayGroup[]>(() => {
  const evs = props.payload.events;
  if (!Array.isArray(evs) || evs.length <= 1) return [];

  const groupMap = new Map<string, { date: Date | null; rows: ListRow[] }>();

  for (const ev of evs) {
    const dt = parseISO(ev.dtstart);
    const key = dt ? dateKey(dt) : 'unknown';

    if (!groupMap.has(key)) {
      groupMap.set(key, { date: dt, rows: [] });
    }

    groupMap.get(key)!.rows.push({
      timeText: ev.all_day ? 'All day' : dt ? formatTime(dt) : '',
      title: ev.title ?? '',
      location: ev.location ?? null,
    });
  }

  return [...groupMap].map(([key, group]) => ({
    key,
    label: group.date
      ? `${DAYS[group.date.getDay()]} ${group.date.getDate()} ${MONTHS[group.date.getMonth()]}`
      : null,
    rows: group.rows,
  }));
});
</script>

<template>
  <div v-if="singleEvent" class="rich-card calendar-card">
    <div class="calendar-card__when">
      <div class="calendar-card__when-day">{{ whenBlock.day }}</div>
      <div class="calendar-card__when-date">{{ whenBlock.date }}</div>
      <div class="calendar-card__when-mon">{{ whenBlock.mon }}</div>
    </div>

    <div class="calendar-card__info">
      <h4 class="calendar-card__title">{{ singleEvent.title ?? '' }}</h4>

      <div class="calendar-card__meta">
        <b v-if="singleMeta.timePart">{{ singleMeta.timePart }}</b>
        <template v-if="singleMeta.calendarName">
          <template v-if="singleMeta.timePart"> · </template>{{ singleMeta.calendarName }}
        </template>
      </div>

      <div v-if="singleEvent.location" class="calendar-card__location">
        {{ singleEvent.location }}
      </div>

      <div v-if="singleEvent.attendees?.length" class="calendar-card__attendees">
        {{ singleEvent.attendees.join(', ') }}
      </div>
    </div>
  </div>

  <div v-else-if="dayGroups.length > 0" class="rich-card calendar-card calendar-card--list">
    <div v-for="group in dayGroups" :key="group.key" class="calendar-card__day-group">
      <div v-if="group.label" class="calendar-card__day-label">
        {{ group.label }}
      </div>

      <div v-for="(row, rowIdx) in group.rows" :key="rowIdx" class="calendar-card__row">
        <span class="calendar-card__row-time">{{ row.timeText }}</span>
        <span class="calendar-card__row-title">{{ row.title }}</span>
        <span v-if="row.location" class="calendar-card__row-loc">{{ row.location }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
/* Card-specific rules only; base .rich-card chrome lives globally in base_card.css. */

.rich-card.calendar-card {
  max-width: 620px;
}

.calendar-card:not(.calendar-card--list) {
  display: grid;
  grid-template-columns: 52px 1fr;
  gap: 16px;
  align-items: center;
}

.calendar-card__when {
  border-radius: 10px;
  background: color-mix(in oklab, var(--violet) 12%, transparent);
  border: 1px solid color-mix(in oklab, var(--violet) 26%, transparent);
  padding: 6px 4px;
  text-align: center;
}

.calendar-card__when-day {
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.58rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: color-mix(in oklab, var(--violet) 80%, var(--text-secondary));
}

.calendar-card__when-date {
  font-size: 1.35rem;
  font-weight: 500;
  letter-spacing: -0.03em;
  line-height: 1;
  margin: 2px 0;
  color: var(--text-primary);
}

.calendar-card__when-mon {
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.58rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-tertiary);
}

.calendar-card__title {
  font-size: 0.96rem;
  font-weight: 500;
  letter-spacing: -0.005em;
  margin: 0 0 2px;
  color: var(--text-primary);
}

.calendar-card__meta {
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.74rem;
  color: var(--text-tertiary);
  letter-spacing: 0.04em;
}

.calendar-card__meta b {
  color: var(--text-secondary);
  font-weight: 500;
}

.calendar-card__location {
  font-size: 0.82rem;
  color: var(--text-secondary);
  margin-top: 4px;
}

.calendar-card__attendees {
  font-size: 0.76rem;
  color: var(--text-tertiary);
  margin-top: 4px;
  line-height: 1.4;
}

.calendar-card--list {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.calendar-card__day-group {
  padding: 0;
}

.calendar-card__day-group + .calendar-card__day-group {
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid color-mix(in oklab, var(--border) 60%, transparent);
}

.calendar-card__day-label {
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.62rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: color-mix(in oklab, var(--violet) 80%, var(--text-secondary));
  margin-bottom: 8px;
}

.calendar-card__row {
  display: flex;
  align-items: baseline;
  gap: 12px;
  padding: 5px 0;
  font-size: 0.88rem;
  border-bottom: 1px solid color-mix(in oklab, var(--border) 35%, transparent);
}

.calendar-card__row:last-child {
  border-bottom: none;
}

.calendar-card__row-time {
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.74rem;
  color: var(--text-secondary);
  font-weight: 500;
  letter-spacing: 0.04em;
  flex-shrink: 0;
  min-width: 48px;
}

.calendar-card__row-title {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text-primary);
}

.calendar-card__row-loc {
  font-size: 0.76rem;
  color: var(--text-tertiary);
  flex-shrink: 0;
  max-width: 160px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>

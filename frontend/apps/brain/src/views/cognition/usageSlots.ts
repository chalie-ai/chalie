export type UsageWindow = 'hour' | 'day' | 'week' | 'month' | 'lifetime';

export type UsageType = 'all' | 'foreground' | 'background';

/** The only place filter values map to their display vocabulary; 'all' sends no wire filter. */
export const USAGE_TYPES: { value: UsageType; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'foreground', label: 'Foreground' },
  { value: 'background', label: 'Background' },
];

export interface BucketValue {
  input: number;
  output: number;
}

export interface SlotRow {
  label: string;
  input: number;
  output: number;
}

const MONTH_NAMES = [
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
];
const DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const TABLE_HEADERS: Record<string, string> = {
  hour: 'Hour',
  day: 'Hour',
  week: 'Day',
  month: 'Day',
  lifetime: 'Month',
};

export function fmtTokens(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
  return String(n);
}

function buildHourSlots(bm: Record<string, BucketValue>): SlotRow[] {
  return Object.entries(bm)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => ({ label: k.slice(11, 16), input: v.input, output: v.output }));
}

function buildDaySlots(bm: Record<string, BucketValue>): SlotRow[] {
  const todayPrefix = new Date().toISOString().slice(0, 10);
  const slots: SlotRow[] = [];
  for (let h = 0; h < 24; h++) {
    const hh = String(h).padStart(2, '0');
    const key = `${todayPrefix}T${hh}:00:00`;
    const d = bm[key] ?? { input: 0, output: 0 };
    slots.push({ label: `${hh}:00`, input: d.input, output: d.output });
  }
  return slots;
}

function buildWeekSlots(bm: Record<string, BucketValue>, now: Date): SlotRow[] {
  const slots: SlotRow[] = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date(now);
    d.setUTCDate(d.getUTCDate() - i);
    const key = d.toISOString().slice(0, 10);
    const dd = String(d.getUTCDate()).padStart(2, '0');
    const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
    const label = `${DAY_NAMES[d.getUTCDay()]} ${dd}/${mm}`;
    const data = bm[key] ?? { input: 0, output: 0 };
    slots.push({ label, input: data.input, output: data.output });
  }
  return slots;
}

function buildMonthSlots(bm: Record<string, BucketValue>, now: Date): SlotRow[] {
  const year = now.getUTCFullYear();
  const month = now.getUTCMonth();
  const daysInMonth = new Date(Date.UTC(year, month + 1, 0)).getUTCDate();
  const slots: SlotRow[] = [];
  for (let day = 1; day <= daysInMonth; day++) {
    const key = `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
    const label = `${String(day).padStart(2, '0')} ${MONTH_NAMES[month]}`;
    const d = bm[key] ?? { input: 0, output: 0 };
    slots.push({ label, input: d.input, output: d.output });
  }
  return slots;
}

function buildLifetimeSlots(bm: Record<string, BucketValue>, now: Date): SlotRow[] {
  const keys = Object.keys(bm).sort();
  if (keys.length === 0) return [];
  const monthAgg: Record<string, BucketValue> = {};
  for (const [k, v] of Object.entries(bm)) {
    const mk = k.slice(0, 7);
    if (!monthAgg[mk]) monthAgg[mk] = { input: 0, output: 0 };
    monthAgg[mk].input += v.input;
    monthAgg[mk].output += v.output;
  }
  const slots: SlotRow[] = [];
  let [y, m] = keys[0].slice(0, 7).split('-').map(Number) as [number, number];
  const endY = now.getUTCFullYear();
  const endM = now.getUTCMonth() + 1;
  while (y < endY || (y === endY && m <= endM)) {
    const mk = `${y}-${String(m).padStart(2, '0')}`;
    const label = `${MONTH_NAMES[m - 1]} ${String(y).slice(2)}`;
    const d = monthAgg[mk] ?? { input: 0, output: 0 };
    slots.push({ label, input: d.input, output: d.output });
    if (++m > 12) {
      m = 1;
      y++;
    }
  }
  return slots;
}

export function buildTableSlots(bm: Record<string, BucketValue>, window: UsageWindow): SlotRow[] {
  const now = new Date();
  if (window === 'hour') return buildHourSlots(bm);
  if (window === 'day') return buildDaySlots(bm);
  if (window === 'week') return buildWeekSlots(bm, now);
  if (window === 'month') return buildMonthSlots(bm, now);
  if (window === 'lifetime') return buildLifetimeSlots(bm, now);
  return [];
}

export function tableHeaderFor(window: UsageWindow): string {
  return TABLE_HEADERS[window] ?? 'Date';
}

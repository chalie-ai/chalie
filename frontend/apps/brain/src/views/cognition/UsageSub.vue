<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import type { UsageResponse } from '../../api/cognition';
import { cognition } from '../../api/cognition';
import EmptyState from '../../ui/EmptyState.vue';

const _MONTH_NAMES = [
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
const _DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const _TABLE_HEADERS: Record<string, string> = {
  hour: 'Hour',
  day: 'Hour',
  week: 'Day',
  month: 'Day',
  lifetime: 'Month',
};

type UsageWindow = 'hour' | 'day' | 'week' | 'month' | 'lifetime';

const usageWindow = ref<UsageWindow>('day');
const data = ref<UsageResponse | null>(null);
const loading = ref(true);
const loadFailed = ref(false);

function fmtTokens(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
  return String(n);
}

let fetchGen = 0;

async function load(): Promise<void> {
  const gen = ++fetchGen;
  loading.value = true;
  loadFailed.value = false;
  try {
    const result = await cognition.tokenUsage(usageWindow.value);
    if (gen !== fetchGen) return;
    data.value = result;
  } catch {
    if (gen !== fetchGen) return;
    loadFailed.value = true;
  } finally {
    if (gen === fetchGen) loading.value = false;
  }
}

function selectWindow(w: UsageWindow): void {
  usageWindow.value = w;
  load();
}

onMounted(load);

interface BucketValue {
  input: number;
  output: number;
}

const bucketMap = computed((): Record<string, BucketValue> => {
  const entries = data.value?.entries ?? [];
  const map: Record<string, BucketValue> = {};
  for (const e of entries) {
    const b = e.bucket || '';
    if (!map[b]) map[b] = { input: 0, output: 0 };
    map[b].input += (e.tokens_input ?? 0) + (e.tokens_cache_read ?? 0);
    map[b].output += (e.tokens_output ?? 0) + (e.tokens_thinking ?? 0);
  }
  return map;
});

const windowLabels: Record<UsageWindow, string> = {
  hour: 'Last Hour',
  day: 'Last 24h',
  week: 'Last 7 Days',
  month: 'Last 30 Days',
  lifetime: 'All Time',
};

const summaryCards = computed(() => {
  const rawSummary = data.value?.summary ?? {};
  return [
    {
      value: fmtTokens(rawSummary.total_tokens ?? 0),
      label: windowLabels[usageWindow.value] ?? 'Total Tokens',
    },
    {
      value: rawSummary.cache_hit_pct == null ? 'N/A' : `${rawSummary.cache_hit_pct}%`,
      label: 'Cache Hit Rate',
    },
    { value: fmtTokens(rawSummary.tokens_today ?? 0), label: 'Today (UTC)' },
    { value: rawSummary.most_active_model || '—', label: 'Top Model' },
  ];
});

interface ChartBar {
  label: string;
  input: number;
  output: number;
}

const chartData = computed((): ChartBar[] => {
  return Object.entries(bucketMap.value)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([bucket, v]) => ({
      label:
        usageWindow.value === 'hour' || usageWindow.value === 'day'
          ? bucket.slice(11, 16)
          : bucket.slice(5, 10),
      input: v.input,
      output: v.output,
    }));
});

interface SlotRow {
  label: string;
  input: number;
  output: number;
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
    const label = `${_DAY_NAMES[d.getUTCDay()]} ${dd}/${mm}`;
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
    const label = `${String(day).padStart(2, '0')} ${_MONTH_NAMES[month]}`;
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
    const label = `${_MONTH_NAMES[m - 1]} ${String(y).slice(2)}`;
    const d = monthAgg[mk] ?? { input: 0, output: 0 };
    slots.push({ label, input: d.input, output: d.output });
    if (++m > 12) {
      m = 1;
      y++;
    }
  }
  return slots;
}

function buildTableSlots(bm: Record<string, BucketValue>): SlotRow[] {
  const now = new Date();
  if (usageWindow.value === 'hour') return buildHourSlots(bm);
  if (usageWindow.value === 'day') return buildDaySlots(bm);
  if (usageWindow.value === 'week') return buildWeekSlots(bm, now);
  if (usageWindow.value === 'month') return buildMonthSlots(bm, now);
  if (usageWindow.value === 'lifetime') return buildLifetimeSlots(bm, now);
  return [];
}

const tableSlots = computed((): SlotRow[] => buildTableSlots(bucketMap.value));

const tableHeader = computed((): string => _TABLE_HEADERS[usageWindow.value] ?? 'Date');

interface BarGeom {
  x: number;
  outputY: number;
  outputH: number;
  inputY: number;
  inputH: number;
  barW: number;
  label: string;
  showLabel: boolean;
  labelX: number;
  labelY: number;
}

const chartGeom = computed(() => {
  const chart = chartData.value;
  if (chart.length === 0) return null;
  const maxVal = Math.max(...chart.map((d) => (d.input || 0) + (d.output || 0)), 1);
  const barW = Math.max(16, Math.floor(600 / chart.length) - 4);
  const h = 160;
  const labelH = 40;
  const showEvery = chart.length > 20 ? Math.ceil(chart.length / 15) : 1;
  const svgW = chart.length * (barW + 4);

  const bars: BarGeom[] = chart.map((d, i) => {
    const inputH = ((d.input || 0) / maxVal) * h;
    const outputH = ((d.output || 0) / maxVal) * h;
    const x = i * (barW + 4);
    const showLabel = i % showEvery === 0;
    return {
      x,
      outputY: h - inputH - outputH,
      outputH,
      inputY: h - inputH,
      inputH,
      barW,
      label: d.label ?? '',
      showLabel,
      labelX: x + barW / 2,
      labelY: h + 6,
    };
  });

  return { bars, svgW, h, labelH };
});
</script>

<template>
  <template v-if="loading"><div class="loading">Loading…</div></template>
  <template v-else-if="loadFailed"><EmptyState message="Failed to load data." /></template>
  <template v-else>
    <div id="usageWindowTabs" class="filter-tabs">
      <button
        v-for="w in ['hour', 'day', 'week', 'month', 'lifetime'] as const"
        :key="w"
        class="filter-tab"
        :class="{ active: w === usageWindow }"
        @click="selectWindow(w)"
      >
        {{ w.charAt(0).toUpperCase() + w.slice(1) }}
      </button>
    </div>

    <div class="stat-grid">
      <div v-for="card in summaryCards" :key="card.label" class="stat-card">
        <div class="stat-value">{{ card.value }}</div>
        <div class="stat-label">{{ card.label }}</div>
      </div>
    </div>

    <template v-if="chartGeom">
      <div class="chart-wrap">
        <!-- eslint-disable-next-line vue/html-self-closing -->
        <svg
          class="usage-chart"
          :viewBox="`0 0 ${chartGeom.svgW} ${chartGeom.h + chartGeom.labelH}`"
          preserveAspectRatio="none"
        >
          <g v-for="bar in chartGeom.bars" :key="bar.label">
            <rect
              :x="bar.x"
              :y="bar.outputY"
              :width="bar.barW"
              :height="bar.outputH"
              class="bar-cloud"
            />
            <rect
              :x="bar.x"
              :y="bar.inputY"
              :width="bar.barW"
              :height="bar.inputH"
              class="bar-local"
            />
            <text
              v-if="bar.showLabel"
              :x="bar.labelX"
              :y="bar.labelY"
              class="bar-label"
              :transform="`rotate(45 ${bar.labelX} ${bar.labelY})`"
            >
              {{ bar.label }}
            </text>
          </g>
        </svg>
      </div>
      <div class="chart-legend">
        <span class="legend-local">Chat</span>
        <span class="legend-cloud">Subconscious</span>
      </div>
    </template>

    <table v-if="tableSlots.length > 0" class="usage-table">
      <thead>
        <tr>
          <th>{{ tableHeader }}</th>
          <th>Input Tokens</th>
          <th>Output Tokens</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="slot in tableSlots" :key="slot.label">
          <td>{{ slot.label }}</td>
          <td class="num">{{ (slot.input || 0).toLocaleString() }}</td>
          <td class="num">{{ (slot.output || 0).toLocaleString() }}</td>
        </tr>
      </tbody>
    </table>
  </template>
</template>

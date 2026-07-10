<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import type { UsageResponse } from '../../api/cognition';
import { cognition } from '../../api/cognition';
import { capitalize } from '../../utils/format';
import EmptyState from '../../ui/EmptyState.vue';
import { type UsageWindow, type BucketValue, type SlotRow, fmtTokens, buildTableSlots, tableHeaderFor } from './usageSlots';

const usageWindow = ref<UsageWindow>('day');
const data = ref<UsageResponse | null>(null);
const loading = ref(true);
const loadFailed = ref(false);

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

const tableSlots = computed((): SlotRow[] => buildTableSlots(bucketMap.value, usageWindow.value));

const tableHeader = computed((): string => tableHeaderFor(usageWindow.value));

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
        {{ capitalize(w) }}
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

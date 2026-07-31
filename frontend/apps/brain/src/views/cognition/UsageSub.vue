<script setup lang="ts">
import { computed, ref } from 'vue';
import type { UsageResponse } from '../../api/cognition';
import { cognition } from '../../api/cognition';
import { capitalize } from '../../utils/format';
import { useAsyncResource } from '@chalie/shared';
import EmptyState from '../../ui/EmptyState.vue';
import { type UsageWindow, type UsageType, type BucketValue, type SlotRow, USAGE_TYPES, fmtTokens, buildTableSlots, tableHeaderFor } from './usageSlots';

const usageWindow = ref<UsageWindow>('day');
const usageType = ref<UsageType>('all');
const {
  data,
  loading,
  error: loadFailed,
  reload,
} = useAsyncResource<UsageResponse | null>(
  () => cognition.tokenUsage(usageWindow.value, usageType.value === 'all' ? undefined : usageType.value),
  {
    initial: null,
    guarded: true,
  },
);

function selectWindow(w: UsageWindow): void {
  usageWindow.value = w;
  reload();
}

function selectType(t: UsageType): void {
  usageType.value = t;
  reload();
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
    <div class="records-controls">
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
      <div id="usageTypeTabs" class="filter-tabs">
        <button
          v-for="t in USAGE_TYPES"
          :key="t.value"
          class="filter-tab"
          :class="{ active: t.value === usageType }"
          @click="selectType(t.value)"
        >
          {{ t.label }}
        </button>
      </div>
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
              class="bar-output"
            />
            <rect
              :x="bar.x"
              :y="bar.inputY"
              :width="bar.barW"
              :height="bar.inputH"
              class="bar-input"
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
        <span class="legend-input">Input</span>
        <span class="legend-output">Output</span>
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

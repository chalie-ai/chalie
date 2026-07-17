<script setup lang="ts">
import { computed, ref } from 'vue';
import type { WorldState, WorldTelemetry } from '../../api/cognition';
import { cognition } from '../../api/cognition';
import { mdToHtml as renderMd } from '../../utils/format';
import { useAsyncResource } from '@chalie/shared';
import EmptyState from '../../ui/EmptyState.vue';
import SegmentedControl from '../../ui/SegmentedControl.vue';

const {
  data: worldState,
  loading,
  error: loadFailed,
} = useAsyncResource(() => cognition.worldState(), { initial: {} as WorldState });

const viewMode = ref<'formatted' | 'raw'>('formatted');

const inputs = computed(() => worldState.value.inputs || {});
const telemetry = computed<WorldTelemetry>(() => inputs.value.telemetry || {});
const signals = computed<Record<string, { label?: string; [k: string]: unknown }>>(
  () => inputs.value.signals || {},
);
const bgProcs = computed<(string | Record<string, unknown>)[]>(
  () => inputs.value.bg_processes || [],
);

const deviceInfo = computed(() => telemetry.value.device || {});
const battery = computed(() => telemetry.value.battery || {});
const location = computed(() => telemetry.value.location_name || '');
const localTime = computed(() => telemetry.value.local_time || '');
const timezone = computed(() => telemetry.value.timezone || '');
const prefs = computed(() => telemetry.value.preferences || {});

const signalEntries = computed(() => Object.entries(signals.value));

const batteryText = computed((): string => {
  const pct = Math.round((battery.value.level || 0) * 100);
  return `${pct}%${battery.value.charging ? ' ⚡ charging' : ''}`;
});

const deviceText = computed((): string => {
  const d = deviceInfo.value;
  return `${d.class || '—'} · ${d.platform || ''} · ${d.screen_w || '?'}×${d.screen_h || '?'}`;
});

// World telemetry tags like `[signal:news]` are bare brackets (no link target),
// so the shared renderer leaves them as text — wrap them as world-tag chips here.
function mdToHtml(md: string): string {
  return renderMd(md).replace(/\[([^\]]+)\]/g, '<span class="world-tag">$1</span>');
}
</script>

<template>
  <template v-if="loading"><div class="loading">Loading…</div></template>
  <template v-else-if="loadFailed"><EmptyState message="Failed to load data." /></template>
  <template v-else>
    <div class="world-state-grid">
      <div class="world-section">
        <h4>Device &amp; Environment</h4>
        <table class="records-table">
          <thead>
            <tr>
              <th scope="col" class="sr-only">Property</th>
              <th scope="col" class="sr-only">Value</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="key-cell">Location</td>
              <td>{{ location || '—' }}</td>
            </tr>
            <tr>
              <td class="key-cell">Local Time</td>
              <td>{{ localTime }} ({{ timezone }})</td>
            </tr>
            <tr>
              <td class="key-cell">Device</td>
              <td>{{ deviceText }}</td>
            </tr>
            <tr>
              <td class="key-cell">Battery</td>
              <td>{{ batteryText }}</td>
            </tr>
            <tr>
              <td class="key-cell">Network</td>
              <td>{{ telemetry.connection || '—' }}</td>
            </tr>
            <tr>
              <td class="key-cell">Theme</td>
              <td>{{ prefs.color_scheme || '—' }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div v-if="signalEntries.length > 0" class="world-section">
        <h4>Active Signals</h4>
        <table class="records-table">
          <thead>
            <tr>
              <th>Signal</th>
              <th>Label</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="[k, v] in signalEntries" :key="k">
              <td class="key-cell">{{ k }}</td>
              <td>{{ v.label || JSON.stringify(v) }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="world-section">
        <h4>Background Processes</h4>
        <table v-if="bgProcs.length > 0" class="records-table">
          <thead>
            <tr>
              <th scope="col" class="sr-only">Process</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(p, i) in bgProcs" :key="i">
              <td>{{ typeof p === 'string' ? p : JSON.stringify(p) }}</td>
            </tr>
          </tbody>
        </table>
        <p v-else class="panel-desc">None running.</p>
      </div>

      <div v-if="worldState.rendered" class="world-section">
        <div class="world-sees-header">
          <h4>What Chalie Sees</h4>
          <SegmentedControl
            v-model="viewMode"
            :segments="[
              { value: 'formatted', label: 'Formatted' },
              { value: 'raw', label: 'Raw' },
            ]"
          />
        </div>
        <div v-if="viewMode === 'raw'" class="code-block">
          <pre><code>{{ worldState.rendered }}</code></pre>
        </div>
        <!-- eslint-disable-next-line vue/no-v-html -->
        <div
          v-else
          class="doc-content world-formatted"
          v-html="mdToHtml(worldState.rendered)"
        ></div>
      </div>
    </div>
  </template>
</template>

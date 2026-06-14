<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { cognition } from '../../api/cognition';
import type { WorldState, WorldTelemetry, WorldSchedule } from '../../api/cognition';
import { formatDate, escapeHtml } from '../../utils/format';
import EmptyState from '../../ui/EmptyState.vue';
import SegmentedControl from '../../ui/SegmentedControl.vue';

const worldState = ref<WorldState>({});
const viewMode = ref<'formatted' | 'raw'>('formatted');
const loading = ref(false);
const loadFailed = ref(false);

async function load(): Promise<void> {
  loading.value = true;
  loadFailed.value = false;
  try {
    worldState.value = await cognition.worldState();
  } catch {
    loadFailed.value = true;
  } finally {
    loading.value = false;
  }
}

onMounted(load);

const inputs = computed(() => worldState.value.inputs || {});
const telemetry = computed<WorldTelemetry>(() => inputs.value.telemetry || {});
const schedule = computed<WorldSchedule[]>(() => inputs.value.schedule || []);
const signals = computed<Record<string, { label?: string; [k: string]: unknown }>>(() => inputs.value.signals || {});
const bgProcs = computed<(string | Record<string, unknown>)[]>(() => inputs.value.bg_processes || []);

const deviceInfo = computed(() => telemetry.value.device || {});
const battery = computed(() => telemetry.value.battery || {});
const location = computed(() => telemetry.value.location_name || '');
const localTime = computed(() => telemetry.value.local_time || '');
const timezone = computed(() => telemetry.value.timezone || '');
const prefs = computed(() => telemetry.value.preferences || {});

const signalEntries = computed(() => Object.entries(signals.value));
const pendingSched = computed(() => schedule.value.filter((s) => s.status === 'pending'));

function batteryText(): string {
  const pct = Math.round((battery.value.level || 0) * 100);
  return `${pct}%${battery.value.charging ? ' ⚡ charging' : ''}`;
}

function deviceText(): string {
  const d = deviceInfo.value;
  return `${d.class || '—'} · ${d.platform || ''} · ${d.screen_w || '?'}×${d.screen_h || '?'}`;
}

function mdToHtml(md: string): string {
  return escapeHtml(md)
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h3>$1</h3>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/^\* (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`)
    .replace(/\[([^\]]+)\]/g, '<span class="world-tag">$1</span>')
    .replace(/\n{2,}/g, '<br>')
    .replace(/\n/g, '\n');
}
</script>

<template>
  <template v-if="loading"><div class="loading">Loading…</div></template>
  <template v-else-if="loadFailed"><EmptyState message="Failed to load data." /></template>
  <template v-else>
    <div class="world-state-grid">
      <!-- Device & Environment -->
      <div class="world-section">
        <h4>Device &amp; Environment</h4>
        <table class="records-table">
          <tbody>
            <tr><td class="key-cell">Location</td><td>{{ location || '—' }}</td></tr>
            <tr><td class="key-cell">Local Time</td><td>{{ localTime }} ({{ timezone }})</td></tr>
            <tr><td class="key-cell">Device</td><td>{{ deviceText() }}</td></tr>
            <tr><td class="key-cell">Battery</td><td>{{ batteryText() }}</td></tr>
            <tr><td class="key-cell">Network</td><td>{{ telemetry.connection || '—' }}</td></tr>
            <tr><td class="key-cell">Theme</td><td>{{ prefs.color_scheme || '—' }}</td></tr>
          </tbody>
        </table>
      </div>

      <!-- Pending Schedules (conditional) -->
      <div v-if="pendingSched.length > 0" class="world-section">
        <h4>Pending Schedules</h4>
        <table class="records-table">
          <thead>
            <tr><th>Message</th><th>Due</th><th>Recurrence</th></tr>
          </thead>
          <tbody>
            <tr v-for="(s, i) in pendingSched" :key="i">
              <td class="key-cell">{{ s.message || '' }}</td>
              <td>{{ formatDate(s.due_at) }}</td>
              <td>{{ s.recurrence || '—' }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Active Signals (conditional) -->
      <div v-if="signalEntries.length > 0" class="world-section">
        <h4>Active Signals</h4>
        <table class="records-table">
          <thead>
            <tr><th>Signal</th><th>Label</th></tr>
          </thead>
          <tbody>
            <tr v-for="([k, v], i) in signalEntries" :key="i">
              <td class="key-cell">{{ k }}</td>
              <td>{{ v.label || JSON.stringify(v) }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Background Processes -->
      <div class="world-section">
        <h4>Background Processes</h4>
        <table v-if="bgProcs.length > 0" class="records-table">
          <tbody>
            <tr v-for="(p, i) in bgProcs" :key="i">
              <td>{{ typeof p === 'string' ? p : JSON.stringify(p) }}</td>
            </tr>
          </tbody>
        </table>
        <p v-else class="panel-desc">None running.</p>
      </div>

      <!-- What Chalie Sees (conditional) -->
      <div v-if="worldState.rendered" class="world-section">
        <div class="world-sees-header">
          <h4>What Chalie Sees</h4>
          <SegmentedControl
            v-model="viewMode"
            :segments="[{ value: 'formatted', label: 'Formatted' }, { value: 'raw', label: 'Raw' }]"
          />
        </div>
        <div v-if="viewMode === 'raw'" class="code-block">
          <pre><code>{{ worldState.rendered }}</code></pre>
        </div>
        <!-- eslint-disable-next-line vue/no-v-html -->
        <div v-else class="doc-content world-formatted" v-html="mdToHtml(worldState.rendered)"></div>
      </div>
    </div>
  </template>
</template>

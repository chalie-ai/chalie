<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { policies } from '../../api/policies';
import type { PolicyRow, BlockedEntry } from '../../api/policies';
import { formatDate } from '../../utils/format';
import { HttpError } from '@chalie/shared';
import { useToast } from '../../composables/useToast';

const props = defineProps<{ channel: 'chat' | 'subconscious' | 'external_agent' }>();

const { show: showToast } = useToast();

const SETTINGS = ['allow', 'ask', 'deny'] as const;
const cap = (v: string): string => v.charAt(0).toUpperCase() + v.slice(1);

const rows = ref<PolicyRow[]>([]);
const blocked = ref<BlockedEntry[]>([]);
const loading = ref(true);
const blockedOpen = ref(false);

interface PolicyCategory {
  cat: string;
  isMcp: boolean;
  rows: { r: PolicyRow; label: string }[];
}

const categories = computed<PolicyCategory[]>(() => {
  const byCat: Record<string, PolicyCategory> = {};
  for (const r of rows.value) {
    if (r.channel !== props.channel) continue;
    const isMcp = !!r.group;
    const cat = r.group || r.permission.split('.')[0];
    const label = r.label || r.permission;
    (byCat[cat] ??= { cat, isMcp, rows: [] }).rows.push({ r, label });
  }
  return Object.values(byCat).sort((a, b) => a.cat.localeCompare(b.cat));
});

async function load(): Promise<void> {
  loading.value = true;
  try {
    const [polRes, blockRes] = await Promise.allSettled([policies.list(), policies.blocked()]);
    if (polRes.status === 'fulfilled') rows.value = polRes.value.policies ?? [];
    if (blockRes.status === 'fulfilled') blocked.value = blockRes.value.entries ?? [];
    const networkFailure = [polRes, blockRes].some(
      (r) => r.status === 'rejected' && !(r.reason instanceof HttpError),
    );
    if (networkFailure) showToast('Failed to load policies', 'error');
  } finally {
    loading.value = false;
  }
}

async function setPermission(r: PolicyRow, value: string): Promise<void> {
  const previous = r.setting;
  r.setting = value;
  try {
    await policies.update({ channel: props.channel, permission: r.permission, setting: value });
    showToast(`${r.permission} → ${value}`, 'success');
  } catch (e) {
    r.setting = previous;
    showToast(e instanceof HttpError ? 'Failed to update policy' : 'Network error', 'error');
  }
}

async function setAll(setting: string): Promise<void> {
  const targets = rows.value.filter((r) => r.channel === props.channel && r.setting !== setting);
  if (targets.length === 0) {
    showToast(`All already ${setting}`, 'success');
    return;
  }
  const results = await Promise.all(
    targets.map((r) =>
      policies
        .update({ channel: props.channel, permission: r.permission, setting })
        .then(() => true)
        .catch(() => false),
    ),
  );
  targets.forEach((r, i) => {
    if (results[i]) r.setting = setting;
  });
  const ok = results.filter(Boolean).length;
  if (ok === targets.length) showToast(`All ${props.channel} → ${setting} (${ok})`, 'success');
  else showToast(`Updated ${ok}/${targets.length} — some failed`, 'error');
}

onMounted(load);
</script>

<template>
  <div class="panel-header">
    <h2>Policies</h2>
    <div class="panel-header-actions">
      <div class="segmented policy-bulk" title="Set every permission in this channel">
        <button
          v-for="v in SETTINGS"
          :key="v"
          class="seg-btn"
          @click="setAll(v)"
        >{{ cap(v) }}</button>
      </div>
    </div>
  </div>

  <div v-if="loading" class="loading">Loading…</div>

  <div v-else class="policies-grid">
    <div
      v-for="c in categories"
      :key="c.cat"
      class="policy-category"
    >
      <h4 class="section-head">
        <span v-if="c.isMcp" class="badge badge-cyan">MCP</span>
        {{ c.cat }}
      </h4>
      <div
        v-for="{ r, label } in c.rows"
        :key="r.permission"
        class="policy-rule"
      >
        <span class="policy-label">{{ label }}</span>
        <div class="segmented" :data-permission="r.permission">
          <button
            v-for="v in SETTINGS"
            :key="v"
            class="seg-btn"
            :class="{ active: v === r.setting }"
            @click="setPermission(r, v)"
          >{{ cap(v) }}</button>
        </div>
      </div>
    </div>
  </div>

  <div v-if="blocked.length > 0" class="blocked-section">
    <button class="blocked-toggle" @click="blockedOpen = !blockedOpen">
      Blocked Actions Log {{ blockedOpen ? '▴' : '▾' }}
    </button>
    <div v-if="blockedOpen" class="blocked-list">
      <div
        v-for="(b, idx) in blocked"
        :key="idx"
        class="blocked-item"
      >
        <span class="badge badge-danger">{{ b.action_id || '' }}</span>
        <span>{{ b.context || '' }}</span>
        <span class="blocked-time">{{ formatDate(b.created_at) }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { ChevronLeft } from '@lucide/vue';
import { ConfigType } from '@chalie/shared';
import type { ThreadSummary, TurnBlock } from '../../api/threads';
import { threads } from '../../api/threads';
import { formatDate, mdToHtml } from '../../utils/format';
import EmptyState from '../../ui/EmptyState.vue';

const viewMode = ref<'list' | 'detail'>('list');
const rows = ref<ThreadSummary[]>([]);
const block = ref<TurnBlock | null>(null);
const loading = ref(false);
const loadFailed = ref(false);
const detailLoading = ref(false);
const detailFailed = ref(false);

async function load(): Promise<void> {
  loading.value = true;
  loadFailed.value = false;
  try {
    const feed = await threads.list(ConfigType.DISCOVERY);
    rows.value = (feed.threads ?? []).filter((t) => t.turn_id != null);
  } catch {
    loadFailed.value = true;
  } finally {
    loading.value = false;
  }
}

async function openDetail(turnId: number): Promise<void> {
  block.value = null;
  detailFailed.value = false;
  detailLoading.value = true;
  viewMode.value = 'detail';
  try {
    block.value = await threads.item(turnId, ConfigType.DISCOVERY);
  } catch {
    detailFailed.value = true;
  } finally {
    detailLoading.value = false;
  }
}

function backToList(): void {
  viewMode.value = 'list';
  block.value = null;
}

onMounted(load);
</script>

<template>
  <template v-if="viewMode === 'detail'">
    <div class="provider-form-page provider-form-page--full">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm back-btn" @click="backToList">
          <ChevronLeft :size="14" /> Back
        </button>
        <h3>{{ block ? formatDate(block.last_activity_at) : 'Research Run' }}</h3>
      </div>
      <template v-if="detailLoading"><div class="loading">Loading…</div></template>
      <template v-else-if="detailFailed"
        ><EmptyState message="Failed to load research run."
      /></template>
      <template v-else-if="block">
        <div class="research-sections">
          <div class="research-section">
            <div class="research-section-heading">Transcript</div>
            <!-- eslint-disable-next-line vue/no-v-html -->
            <div
              v-for="m in block.messages.filter((msg) => msg.role === 'assistant')"
              :key="m.id"
              class="research-section-body md"
              v-html="mdToHtml(m.content)"
            ></div>
          </div>
        </div>
      </template>
    </div>
  </template>

  <template v-else>
    <template v-if="loading"><div class="loading">Loading…</div></template>
    <template v-else-if="loadFailed"><EmptyState message="Failed to load data." /></template>
    <template v-else>
      <EmptyState v-if="rows.length === 0" message="No research runs found." />
      <table v-else class="records-table">
        <thead>
          <tr>
            <th>Ran At</th>
            <th>Researched</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="r in rows"
            :key="r.turn_id ?? 0"
            class="clickable-row"
            @click="r.turn_id != null && openDetail(r.turn_id)"
          >
            <td>{{ formatDate(r.last_activity_at) }}</td>
            <td class="val-cell">{{ r.preview }}</td>
          </tr>
        </tbody>
      </table>
    </template>
  </template>
</template>

<style scoped>
.clickable-row {
  cursor: pointer;
}
.clickable-row:hover td {
  background: var(--bg-hover);
}
.research-sections {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  padding: 1rem 0;
}
.research-section-heading {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  margin-bottom: 0.5rem;
}
.research-section-body {
  font-size: 0.875rem;
  color: var(--text-primary);
  line-height: 1.6;
}
.research-section-body + .research-section-body {
  margin-top: 1rem;
}
.md :deep(h3) {
  font-size: 0.95rem;
  font-weight: 600;
  margin: 0.75rem 0 0.35rem;
}
.md :deep(ul) {
  margin: 0.35rem 0;
  padding-left: 1.25rem;
}
.md :deep(li) {
  margin: 0.15rem 0;
}
.md :deep(code) {
  font-family: var(--font-mono, monospace);
  font-size: 0.85em;
  background: var(--bg-subtle, var(--bg-hover));
  padding: 0.1em 0.3em;
  border-radius: 3px;
}
.md :deep(a) {
  color: var(--accent, var(--text-link));
  text-decoration: underline;
}
</style>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { ChevronLeft } from '@lucide/vue';
import { cognition } from '../../api/cognition';
import type { AutoResearchRun, AutoResearchDetail } from '../../api/cognition';
import { formatDate, mdToHtml } from '../../utils/format';
import EmptyState from '../../ui/EmptyState.vue';

const viewMode = ref<'list' | 'detail'>('list');
const runs = ref<AutoResearchRun[]>([]);
const detail = ref<AutoResearchDetail | null>(null);
const loading = ref(false);
const loadFailed = ref(false);
const detailLoading = ref(false);
const detailFailed = ref(false);

async function load(): Promise<void> {
  loading.value = true;
  loadFailed.value = false;
  try {
    runs.value = (await cognition.research()).runs ?? [];
  } catch {
    loadFailed.value = true;
  } finally {
    loading.value = false;
  }
}

async function openDetail(run: AutoResearchRun): Promise<void> {
  detail.value = null;
  detailFailed.value = false;
  detailLoading.value = true;
  viewMode.value = 'detail';
  try {
    detail.value = (await cognition.researchDetail(run.id)).run;
  } catch {
    detailFailed.value = true;
  } finally {
    detailLoading.value = false;
  }
}

function backToList(): void {
  viewMode.value = 'list';
  detail.value = null;
}

onMounted(load);
</script>

<template>
  <template v-if="viewMode === 'detail'">
    <div class="provider-form-page" style="max-width:none">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm back-btn" @click="backToList">
          <ChevronLeft :size="14" /> Back
        </button>
        <h3>{{ detail ? formatDate(detail.ran_at) : 'Research Run' }}</h3>
      </div>
      <template v-if="detailLoading"><div class="loading">Loading…</div></template>
      <template v-else-if="detailFailed"><EmptyState message="Failed to load research run." /></template>
      <template v-else-if="detail">
        <div class="research-sections">
          <div class="research-section">
            <div class="research-section-heading">User Summary</div>
            <!-- eslint-disable-next-line vue/no-v-html -->
            <div class="research-section-body md" v-html="mdToHtml(detail.user_summary)"></div>
          </div>
          <div class="research-section">
            <div class="research-section-heading">Compacted Summary</div>
            <!-- eslint-disable-next-line vue/no-v-html -->
            <div class="research-section-body md" v-html="mdToHtml(detail.compacted_summary)"></div>
          </div>
          <div class="research-section">
            <div class="research-section-heading">Transcript</div>
            <!-- eslint-disable-next-line vue/no-v-html -->
            <div class="research-section-body md" v-html="mdToHtml(detail.transcript)"></div>
          </div>
        </div>
      </template>
    </div>
  </template>

  <template v-else>
    <template v-if="loading"><div class="loading">Loading…</div></template>
    <template v-else-if="loadFailed"><EmptyState message="Failed to load data." /></template>
    <template v-else>
      <EmptyState v-if="runs.length === 0" message="No research runs found." />
      <table v-else class="records-table">
        <thead>
          <tr>
            <th>Ran At</th>
            <th>Researched</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="r in runs"
            :key="r.id"
            class="clickable-row"
            @click="openDetail(r)"
          >
            <td>{{ formatDate(r.ran_at) }}</td>
            <td class="val-cell">{{ r.researched }}</td>
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

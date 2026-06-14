<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { cognition } from '../../api/cognition';
import type { MemoryRecord } from '../../api/cognition';
import { formatDate } from '../../utils/format';
import EmptyState from '../../ui/EmptyState.vue';

type MemorySource = 'episodes' | 'user' | 'system';

const SOURCES: MemorySource[] = ['episodes', 'user', 'system'];

const source = ref<MemorySource>('episodes');
const search = ref('');
const records = ref<MemoryRecord[]>([]);
const hasMore = ref(false);
const offset = ref(0);
const loading = ref(false);
const loadFailed = ref(false);

let fetchGen = 0;

async function load(): Promise<void> {
  const gen = ++fetchGen;
  loading.value = true;
  loadFailed.value = false;
  try {
    const data = await cognition.memory({
      source: source.value,
      limit: 50,
      offset: offset.value,
      q: search.value || undefined,
    });
    if (gen !== fetchGen) return;
    records.value = data.rows || [];
    hasMore.value = data.has_more || false;
  } catch {
    if (gen !== fetchGen) return;
    loadFailed.value = true;
  } finally {
    if (gen === fetchGen) loading.value = false;
  }
}

function selectSource(s: MemorySource): void {
  source.value = s;
  offset.value = 0;
  load();
}

function submitSearch(): void {
  offset.value = 0;
  load();
}

function loadMore(): void {
  offset.value += 50;
  load();
}

onMounted(load);
</script>

<template>
  <template v-if="loading"><div class="loading">Loading…</div></template>
  <template v-else-if="loadFailed"><EmptyState message="Failed to load data." /></template>
  <template v-else>
    <div class="records-controls">
      <div class="filter-tabs">
        <button
          v-for="s in SOURCES"
          :key="s"
          class="filter-tab"
          :class="{ active: source === s }"
          @click="selectSource(s)"
        >{{ s.charAt(0).toUpperCase() + s.slice(1) }}</button>
      </div>
      <input
        v-model="search"
        type="text"
        class="search-input"
        placeholder="Search…"
        @keydown.enter="submitSearch"
      >
    </div>
    <EmptyState v-if="records.length === 0" message="No records found." />
    <template v-else>
      <table class="records-table">
        <thead>
          <tr>
            <th>Created</th>
            <th>Last Accessed</th>
            <th>{{ source === 'episodes' ? 'Location' : 'Key' }}</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in records" :key="`${r.created}-${r.key ?? r.location ?? ''}`">
            <td>{{ formatDate(r.created) }}</td>
            <td>{{ formatDate(r.last_accessed) }}</td>
            <td class="key-cell">{{ source === 'episodes' ? (r.location || '') : (r.key || '') }}</td>
            <td class="val-cell">{{ r.value || '' }}</td>
          </tr>
        </tbody>
      </table>
      <div v-if="hasMore" class="records-footer">
        <button class="btn btn-secondary" @click="loadMore">Load more</button>
      </div>
    </template>
  </template>
</template>

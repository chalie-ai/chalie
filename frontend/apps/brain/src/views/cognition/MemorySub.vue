<script setup lang="ts">
import { computed, ref } from 'vue';
import { cognition, type MemoryResponse } from '../../api/cognition';
import { capitalize, formatDate } from '../../utils/format';
import { useAsyncResource } from '@chalie/shared';
import EmptyState from '../../ui/EmptyState.vue';

type MemorySource = 'episodes' | 'user' | 'system';

const SOURCES: MemorySource[] = ['episodes', 'user', 'system'];

const source = ref<MemorySource>('episodes');
const search = ref('');
const offset = ref(0);

const {
  data,
  loading,
  error: loadFailed,
  reload,
} = useAsyncResource<MemoryResponse>(
  () =>
    cognition.memory({
      source: source.value,
      limit: 50,
      offset: offset.value,
      q: search.value || undefined,
    }),
  { initial: { rows: [], has_more: false, generated_at: null }, guarded: true },
);

const records = computed(() => data.value.rows ?? []);
const hasMore = computed(() => data.value.has_more ?? false);

function selectSource(s: MemorySource): void {
  source.value = s;
  offset.value = 0;
  reload();
}

function submitSearch(): void {
  offset.value = 0;
  reload();
}

function loadMore(): void {
  offset.value += 50;
  reload();
}
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
        >
          {{ capitalize(s) }}
        </button>
      </div>
      <input
        v-model="search"
        type="text"
        class="search-input"
        placeholder="Search…"
        aria-label="Search…"
        @keydown.enter="submitSearch"
      />
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
            <td class="key-cell">{{ source === 'episodes' ? r.location || '' : r.key || '' }}</td>
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

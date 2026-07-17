<script setup lang="ts">
import type { CompactionEntry } from '../../api/cognition';
import { cognition } from '../../api/cognition';
import { useAsyncResource } from '@chalie/shared';
import EmptyState from '../../ui/EmptyState.vue';

const {
  data: compaction,
  loading,
  error: loadFailed,
} = useAsyncResource(
  async () => {
    const data = await cognition.compaction();
    return data.compaction || null;
  },
  { initial: null as CompactionEntry | null },
);
</script>

<template>
  <template v-if="loading"><div class="loading">Loading…</div></template>
  <template v-else-if="loadFailed"><EmptyState message="Failed to load data." /></template>
  <template v-else>
    <template v-if="!compaction?.summary">
      <p class="panel-desc">
        The continuity summary the chat carries forward after its context overflows and is
        compacted.
      </p>
      <EmptyState message="No compaction has run yet — the chat context has not overflowed." />
    </template>
    <template v-else>
      <p class="panel-desc">
        The continuity summary injected into the chat context after the conversation overflowed.
        This is what Chalie carries forward in place of the older turns.
      </p>
      <div class="stat-grid">
        <div class="stat-card">
          <div class="stat-value">{{ compaction.compacted_at || '—' }}</div>
          <div class="stat-label">Last Compacted</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">
            {{ compaction.compacted_up_to_id ? '#' + compaction.compacted_up_to_id : '—' }}
          </div>
          <div class="stat-label">Compacted Up To (transcript id)</div>
        </div>
      </div>
      <div class="code-block">
        <pre><code>{{ compaction.summary }}</code></pre>
      </div>
    </template>
  </template>
</template>

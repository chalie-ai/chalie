<script setup lang="ts">
import { cognition } from '../../api/cognition';
import type { ErrorEntry } from '../../api/cognition';
import { formatDate } from '../../utils/format';
import { useAsyncResource } from '@chalie/shared';
import EmptyState from '../../ui/EmptyState.vue';

const { data: errors, loading, error: loadFailed } = useAsyncResource(
  async () => (await cognition.errors()).errors ?? [],
  { initial: [] as ErrorEntry[] },
);
</script>

<template>
  <template v-if="loading"><div class="loading">Loading…</div></template>
  <template v-else-if="loadFailed"><EmptyState message="Failed to load data." /></template>
  <template v-else>
    <EmptyState v-if="errors.length === 0" message="No recent errors." />
    <div v-else class="error-list">
      <div v-for="(e, i) in errors" :key="i" class="error-item">
        <span class="error-time">{{ formatDate(e.time || e.timestamp) }}</span>
        <span class="error-msg">{{ e.message || '' }}</span>
      </div>
    </div>
  </template>
</template>

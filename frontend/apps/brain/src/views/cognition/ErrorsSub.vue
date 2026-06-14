<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { cognition } from '../../api/cognition';
import type { ErrorEntry } from '../../api/cognition';
import { formatDate } from '../../utils/format';
import EmptyState from '../../ui/EmptyState.vue';

const loading = ref(true);
const loadFailed = ref(false);
const errors = ref<ErrorEntry[]>([]);

onMounted(async () => {
  try {
    const data = await cognition.errors();
    errors.value = data.errors || [];
  } catch {
    loadFailed.value = true;
  } finally {
    loading.value = false;
  }
});
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

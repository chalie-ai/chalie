<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { cognition } from '../../api/cognition';
import type { Tool } from '../../api/cognition';
import { formatDate } from '../../utils/format';
import EmptyState from '../../ui/EmptyState.vue';

const loading = ref(true);
const loadFailed = ref(false);
const tools = ref<Tool[]>([]);

onMounted(async () => {
  try {
    const data = await cognition.tools();
    tools.value = data.tools || [];
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
    <EmptyState v-if="tools.length === 0" message="No tools loaded." />
    <table v-else class="records-table">
      <thead>
        <tr>
          <th>Name</th>
          <th>Calls</th>
          <th>Last Used</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(t, i) in tools" :key="i">
          <td class="key-cell">{{ t.tool_name || '' }}</td>
          <td>{{ t.count ?? '—' }}</td>
          <td>{{ formatDate(t.last_used_at) || '—' }}</td>
        </tr>
      </tbody>
    </table>
  </template>
</template>

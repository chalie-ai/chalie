<script setup lang="ts">
import EmptyState from './EmptyState.vue';

interface Column {
  key: string;
  label: string;
  class?: string;
}

const props = defineProps<{
  columns: Column[];
  rows: Record<string, unknown>[];
  emptyMessage?: string;
}>();
</script>

<template>
  <div>
    <table v-if="props.rows.length > 0" class="records-table">
      <thead>
        <tr>
          <th v-for="col in props.columns" :key="col.key">{{ col.label }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(row, i) in props.rows" :key="i">
          <td v-for="col in props.columns" :key="col.key" :class="col.class">
            <slot :name="`cell-${col.key}`" :row="row" :value="row[col.key]">
              {{ row[col.key] ?? '' }}
            </slot>
          </td>
        </tr>
      </tbody>
    </table>
    <EmptyState v-else :message="props.emptyMessage ?? 'No records found.'" />
  </div>
</template>

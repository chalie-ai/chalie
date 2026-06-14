<script setup lang="ts">
interface Segment {
  value: string;
  label: string;
}

const props = defineProps<{
  modelValue: string;
  segments: Segment[];
}>();

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void;
}>();
</script>

<template>
  <div class="segmented">
    <button
      v-for="seg in props.segments"
      :key="seg.value"
      :class="['seg-btn', { active: modelValue === seg.value }]"
      :data-val="seg.value"
      type="button"
      @click="emit('update:modelValue', seg.value)"
    >
      {{ seg.label }}
    </button>
  </div>
</template>

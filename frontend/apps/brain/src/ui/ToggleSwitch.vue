<script setup lang="ts">
const props = defineProps<{
  modelValue: boolean;
  label?: string;
}>();

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void;
}>();
</script>

<template>
  <label class="switch">
    <input
      type="checkbox"
      :checked="props.modelValue"
      @change="emit('update:modelValue', ($event.target as HTMLInputElement).checked)"
    />
    <span class="switch-track"></span>
    <span v-if="props.label" class="switch-label">{{ props.label }}</span>
  </label>
</template>

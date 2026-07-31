<script setup lang="ts">
const props = defineProps<{
  modelValue: boolean;
  label?: string;
  id?: string;
}>();

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void;
}>();
</script>

<template>
  <label class="switch">
    <input
      :id="props.id"
      type="checkbox"
      :checked="props.modelValue"
      @change="emit('update:modelValue', ($event.target as HTMLInputElement).checked)"
    />
    <span class="switch-track"></span>
    <span v-if="props.label" class="switch-label">{{ props.label }}</span>
  </label>
</template>

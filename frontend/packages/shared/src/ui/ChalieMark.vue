<script setup lang="ts">
// The Chalie mark: a lowercase c drawn as a clock arc, with a dot where the hand stops.
// At 32px and under the heavier small cut keeps the arc and the dot legible.
withDefaults(defineProps<{ size?: number; working?: boolean }>(), { size: 24, working: false });
</script>

<template>
  <svg
    class="chalie-mark"
    :class="{ working }"
    viewBox="0 0 100 100"
    :width="size"
    :height="size"
    aria-hidden="true"
  >
    <template v-if="size > 32">
      <path d="M75.39 67.78A31 31 0 1 1 65.5 23.15" stroke-width="18" />
      <circle cx="77.62" cy="35.93" r="11" />
    </template>
    <template v-else>
      <path d="M72.98 69.28A30 30 0 1 1 60.26 21.81" stroke-width="20" />
      <circle cx="77.19" cy="37.32" r="12" />
    </template>
  </svg>
</template>

<style scoped>
.chalie-mark { display: block; flex: none; color: var(--pink); }
path { fill: none; stroke: currentColor; }
circle { fill: currentColor; }
/* The dot pulses while Chalie is working. */
.working circle { animation: working 1.4s ease-in-out infinite; }
</style>

<script setup lang="ts">
/**
 * Loading overlay — shown on every load until backend health passes (Task A5).
 * For S5, fades out automatically after first paint via a short delay.
 * The skip button hides it immediately.
 */
import { ref, onMounted, onBeforeUnmount, nextTick } from 'vue';

const visible = ref(true);
const fading = ref(false);

let autoHideTimer: ReturnType<typeof setTimeout> | null = null;
let removeTimer: ReturnType<typeof setTimeout> | null = null;

function hide(): void {
  if (fading.value) return; // idempotent — skip + skip-during-fade are no-ops
  fading.value = true;
  // Wait for the CSS opacity transition (220ms) then remove from layout.
  removeTimer = setTimeout(() => {
    visible.value = false;
  }, 260);
}

onMounted(async () => {
  // Wait for first paint, then fade out. Task A5 replaces this with the
  // real voice-health / backend-ready gate.
  await nextTick();
  autoHideTimer = setTimeout(hide, 800);
});

onBeforeUnmount(() => {
  if (autoHideTimer !== null) clearTimeout(autoHideTimer);
  if (removeTimer !== null) clearTimeout(removeTimer);
});
</script>

<template>
  <output
    v-if="visible"
    id="loadingOverlay"
    class="loading-overlay"
    :class="{ 'loading-overlay--fading': fading }"
    aria-live="polite"
  >
    <div class="loading-overlay__content">
      <div class="loading-overlay__dot" aria-hidden="true"></div>
      <p class="loading-overlay__text">Waking up Chalie...</p>
      <button class="loading-overlay__skip" aria-label="Skip loading" @click="hide">skip</button>
    </div>
  </output>
</template>

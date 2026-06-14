<script setup lang="ts">
/**
 * Loading overlay — shown on every load until backend health passes (Task A5).
 *
 * Real gate: fade out after first paint AND once voiceStore.loading === false.
 * The skip button hides it immediately.
 *
 * Port of app.js _showLoadingOverlay / _dismissLoadingOverlay.
 * Transition: 220ms opacity fade, then remove from DOM after 260ms.
 */
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue';
import { useVoiceStore } from '../../stores/voice';

const voiceStore = useVoiceStore();

const visible = ref(true);
const fading = ref(false);

/** Whether the component has completed first paint. */
let firstPaintDone = false;

let removeTimer: ReturnType<typeof setTimeout> | null = null;
let unwatchVoice: (() => void) | null = null;

/**
 * Kick off the 220ms CSS fade then remove from the DOM after 260ms.
 * Idempotent — subsequent calls while fading are no-ops.
 */
function hide(): void {
  if (fading.value) return;
  fading.value = true;
  removeTimer = setTimeout(() => {
    visible.value = false;
  }, 260);
}

/**
 * Attempt to hide: both conditions must be met:
 *   1. First paint has happened (nextTick in onMounted).
 *   2. voiceStore.loading is false (health check resolved).
 */
function maybeHide(): void {
  if (firstPaintDone && !voiceStore.loading) {
    hide();
  }
}

onMounted(async () => {
  // Wait for first paint.
  await nextTick();
  firstPaintDone = true;

  // Guard: if voice already resolved before we mounted (e.g. fast cache hit).
  if (!voiceStore.loading) {
    hide();
    return;
  }

  // Watch for voiceStore.loading to become false.
  unwatchVoice = watch(
    () => voiceStore.loading,
    (isLoading) => {
      if (!isLoading) {
        maybeHide();
      }
    },
  );
});

onBeforeUnmount(() => {
  unwatchVoice?.();
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

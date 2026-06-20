<script setup lang="ts">
/**
 * Loading overlay — the "Waking up Chalie..." splash shown until the backend
 * signals readiness.
 *
 * Poll GET /ready every 2s, up to 120s, until { ready: true } → fade out. Skip
 * stops the poll and dismisses; on timeout it dismisses anyway so the UI is
 * never permanently blocked.
 *
 * Voice availability is a SEPARATE concern (voiceStore) and must NOT gate this
 * overlay — a slow/unavailable voice service can never block first paint.
 */
import { ref, onMounted, onBeforeUnmount } from 'vue';
import { system } from '../../api/system';

const POLL_INTERVAL_MS = 2000;
const MAX_WAIT_MS = 120_000;
const FADE_MS = 220;

const visible = ref(true);
const fading = ref(false);

/** Cleared to stop the poll loop (skip click or unmount). */
let pollActive = true;
let removeTimer: ReturnType<typeof setTimeout> | null = null;

/** Begin the CSS fade, then remove from the DOM. Idempotent while fading. */
function dismiss(): void {
  if (fading.value) return;
  fading.value = true;
  removeTimer = setTimeout(() => {
    visible.value = false;
  }, FADE_MS);
}

function skip(): void {
  pollActive = false;
  dismiss();
}

/** readyCheck never rejects, so no try/catch is needed. */
async function pollUntilReady(): Promise<void> {
  const deadline = Date.now() + MAX_WAIT_MS;
  while (pollActive && Date.now() < deadline) {
    const result = await system.readyCheck();
    if (!pollActive) return; // skip clicked during the await
    if (result.ready) {
      dismiss();
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
  // Timed out (or skipped mid-sleep) — dismiss so the UI is never blocked.
  if (pollActive) dismiss();
}

onMounted(() => {
  void pollUntilReady();
});

onBeforeUnmount(() => {
  pollActive = false;
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
      <button class="loading-overlay__skip" aria-label="Skip loading" @click="skip">skip</button>
    </div>
  </output>
</template>

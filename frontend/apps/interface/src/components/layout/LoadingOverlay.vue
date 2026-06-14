<script setup lang="ts">
/**
 * Loading overlay — the "Waking up Chalie..." splash shown on every load until
 * the backend signals readiness (Task A5).
 *
 * Gate (port of app.js _pollUntilReady, lines 204-216, + _dismissLoadingOverlay,
 * lines 733-748):
 *   poll GET /ready every 2s, up to 120s, until { ready: true } → fade out.
 *   The skip button stops the poll and dismisses immediately.
 *   On timeout the overlay dismisses anyway so the UI is never permanently
 *   blocked.
 *
 * Voice availability is a SEPARATE concern (voiceStore) governing only mic /
 * speaker visibility — it must NOT gate this overlay. Legacy runs
 * _initVoiceAvailability AFTER the overlay is already dismissed (app.js:238 vs
 * :246), so a slow or unavailable voice service can never block first paint.
 *
 * The overlay is an opaque full-screen layer (interface.scss: position:fixed;
 * inset:0; z-index:100; background:#06080e), so it covers the spine + dock on
 * its own — no need to toggle their display as the legacy DOM code did.
 *
 * Transition: 220ms opacity fade (CSS), then removal from the DOM.
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

/**
 * Begin the 220ms CSS fade, then remove from the DOM.
 * Idempotent — subsequent calls while already fading are no-ops.
 */
function dismiss(): void {
  if (fading.value) return;
  fading.value = true;
  removeTimer = setTimeout(() => {
    visible.value = false;
  }, FADE_MS);
}

/** Skip — stop polling and dismiss now (legacy: _readyPollActive = false). */
function skip(): void {
  pollActive = false;
  dismiss();
}

/**
 * Poll the backend ready endpoint until it reports ready, the deadline passes,
 * or skip stops the loop. readyCheck never rejects, so no try/catch is needed.
 */
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

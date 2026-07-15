<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted } from 'vue';
import { getWebSocket, isTauri, platform, useTheme } from '@chalie/shared';
import { useSessionStore } from './stores/session';
import { useVoiceStore } from './stores/voice';
import { useHeartbeat } from './composables/useHeartbeat';
import { useAmbientSensor } from './composables/useAmbientSensor';
import PresenceBar from './components/layout/PresenceBar.vue';
import ConversationFeed from './components/conversation/ConversationFeed.vue';
import ThreadPanel from './components/conversation/ThreadPanel.vue';
import SearchOverlay from './components/overlays/SearchOverlay.vue';
import InputDock from './components/layout/InputDock.vue';
import LoadingOverlay from './components/layout/LoadingOverlay.vue';
import PermissionStack from './components/overlays/PermissionStack.vue';
import TaskDrawer from './components/overlays/TaskDrawer.vue';
import SchedulerDock from './components/overlays/SchedulerDock.vue';
import UpdatePrompt from './components/overlays/UpdatePrompt.vue';
import VoicePlayerDialog from './components/voice/VoicePlayerDialog.vue';
import UnlockVault from './components/layout/UnlockVault.vue';
import ActionCardHost from './components/rich/ActionCardHost.vue';

const { init: initTheme } = useTheme();
const session = useSessionStore();
const voiceStore = useVoiceStore();

// When a thread panel is open the base layer (feed + footer dock) dims and
// blurs behind it — the mockup's baseStyle. The panel and top bar stay crisp.
const baseDimmed = computed(() => session.panelThreadId != null);

// Cmd/Ctrl-K toggles the thread-search overlay; the overlay owns Esc-to-close.
function onSearchHotkey(e: KeyboardEvent): void {
  if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    session.openSearch();
  }
}

// Single auth-failure redirect — wired to the session store (turn-level
// auth_failed), the heartbeat (periodic /auth/status), and the WebSocket send
// paths (POST /api/thread & /api/action returning 401). Mirrors the router gate.
let _authRedirected = false;
function handleAuthFailure(): void {
  if (_authRedirected) return;
  _authRedirected = true;
  window.location.replace(
    '/login/?next=' + encodeURIComponent(window.location.pathname + window.location.search),
  );
}

onMounted(() => {
  // Theme first, then session (WS connect). Voice availability runs independently —
  // it only governs mic/speaker visibility and must never gate the loading overlay.
  initTheme();

  session.onAuthFailure(handleAuthFailure);
  // A 401 on the WS send paths (POST /api/thread, /api/action) used to surface
  // as a generic failure with no auth-expiry signal — messages sent after
  // session expiry looked accepted while the spinner hung. Route them through
  // the same redirect as the REST path (ApiClient.fail401).
  getWebSocket().onAuthError(handleAuthFailure);
  session.init();
  voiceStore.checkAvailability();

  // Native shell only: request OS notification permission once so background
  // message notifications can fire. On web the browser drives its own prompt.
  if (isTauri) {
    void platform.requestNotificationPermission();
  }

  // Heartbeat also surfaces auth expiry via /auth/status.
  const heartbeat = useHeartbeat();
  heartbeat.onAuthFailure(handleAuthFailure);
  heartbeat.start();
  // Prime geolocation permission once after start.
  void heartbeat.requestLocationPermission();

  globalThis.addEventListener('keydown', onSearchHotkey);
});

onBeforeUnmount(() => {
  globalThis.removeEventListener('keydown', onSearchHotkey);
  useHeartbeat().stop();
  useAmbientSensor().destroy();
});
</script>

<template>
  <PresenceBar />

  <LoadingOverlay />

  <ConversationFeed :class="{ 'weave-dimmed': baseDimmed }" />

  <!-- Footer composer for starting a new thread — always present. It is the
       same component as a thread's inline reply; only the turn_id differs (null
       here scaffolds a new thread). Multiple docks share global stores, so the
       focus-routed handlers (voice/interrupt/attach strip) target the active
       dock only — see InputDock. -->
  <InputDock dock-id="main_spine_dock" :class="{ 'weave-dimmed': baseDimmed }" />

  <!-- Slide-over thread panel — opens over the feed when a pill or Reply action
       sets session.panelThreadId; carries its own reply dock. -->
  <ThreadPanel />

  <!-- Thread search overlay — Cmd/Ctrl-K or the top-bar search button. -->
  <SearchOverlay />

  <!-- ACT-cycle rich-card responses, outside the feed so they never pollute the turn buffer. -->
  <ActionCardHost />

  <!-- Teleport targets for dialogs / permission cards -->
  <div id="permStack" class="permission-stack"></div>
  <div id="overlayRoot"></div>

  <!-- PermissionStack teleports into #permStack; the rest self-render and
       self-subscribe to bus events. -->
  <PermissionStack />
  <TaskDrawer />
  <SchedulerDock />
  <UpdatePrompt />
  <VoicePlayerDialog />
  <UnlockVault />
</template>

<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref } from 'vue';
import { platform, isTauri, useTheme } from '@chalie/shared';
import { useSessionStore } from './stores/session';
import { useVoiceStore } from './stores/voice';
import { on } from './composables/useEventBus';
import { useHeartbeat } from './composables/useHeartbeat';
import { useAmbientSensor } from './composables/useAmbientSensor';
import AmbientCanvas from './components/layout/AmbientCanvas.vue';
import PresenceBar from './components/layout/PresenceBar.vue';
import ConversationFeed from './components/conversation/ConversationFeed.vue';
import InputDock from './components/layout/InputDock.vue';
import LoadingOverlay from './components/layout/LoadingOverlay.vue';
import MomentSearchDialog from './components/overlays/MomentSearchDialog.vue';
import PermissionStack from './components/overlays/PermissionStack.vue';
import TaskDrawer from './components/overlays/TaskDrawer.vue';
import QuickTipCard from './components/overlays/QuickTipCard.vue';
import UpdatePrompt from './components/overlays/UpdatePrompt.vue';
import VoicePlayerDialog from './components/voice/VoicePlayerDialog.vue';
import UnlockVault from './components/layout/UnlockVault.vue';

const { init: initTheme } = useTheme();
const session = useSessionStore();
const voiceStore = useVoiceStore();

const recallRef = ref<InstanceType<typeof MomentSearchDialog> | null>(null);

let _unbindRecall: (() => void) | null = null;

// Single auth-failure redirect — wired to BOTH the session store (turn-level
// auth_failed) and the heartbeat (periodic /auth/status). Mirrors the router gate.
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
  session.init();
  voiceStore.checkAvailability();

  // Native shell only: request OS notification permission once so background
  // message notifications can fire. On web the browser drives its own prompt.
  if (isTauri) {
    void platform.requestNotificationPermission();
  }

  _unbindRecall = on('chalie:open-recall', () => {
    recallRef.value?.open();
  });

  // Heartbeat also surfaces auth expiry via /auth/status.
  const heartbeat = useHeartbeat();
  heartbeat.onAuthFailure(handleAuthFailure);
  heartbeat.start();
  // Prime geolocation permission once after start.
  void heartbeat.requestLocationPermission();
});

onBeforeUnmount(() => {
  _unbindRecall?.();
  useHeartbeat().stop();
  useAmbientSensor().destroy();
});
</script>

<template>
  <AmbientCanvas />
  <div id="ambientBloom"></div>
  <div id="grainOverlay"></div>

  <PresenceBar />

  <LoadingOverlay />

  <ConversationFeed />

  <InputDock />

  <!-- Teleport targets for dialogs / permission cards -->
  <div id="permStack" class="permission-stack"></div>
  <div id="overlayRoot"></div>

  <!-- PermissionStack teleports into #permStack; the rest self-render and
       self-subscribe to bus events. MomentSearchDialog also exposes open() for recall. -->
  <PermissionStack />
  <TaskDrawer />
  <QuickTipCard />
  <UpdatePrompt />
  <VoicePlayerDialog />
  <MomentSearchDialog ref="recallRef" />
  <UnlockVault />
</template>

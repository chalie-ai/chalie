<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref } from 'vue';
import { useTheme } from '@chalie/shared';
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

const { init: initTheme } = useTheme();
const session = useSessionStore();
const voiceStore = useVoiceStore();

/** Imperative handle on the moment-search dialog (PresenceBar's recall button). */
const recallRef = ref<InstanceType<typeof MomentSearchDialog> | null>(null);

/** Unbind for the chalie:open-recall bus subscription. */
let _unbindRecall: (() => void) | null = null;

/**
 * Single auth-failure redirect — wired to BOTH the session store (turn-level
 * auth_failed / AuthError on history) and the heartbeat (periodic /auth/status
 * check). Mirrors the router gate's login redirect (router.ts:64-66).
 */
let _authRedirected = false;
function handleAuthFailure(): void {
  if (_authRedirected) return;
  _authRedirected = true;
  window.location.replace(
    '/login/?next=' + encodeURIComponent(window.location.pathname + window.location.search),
  );
}

onMounted(() => {
  // Theme init first, then session (WS connect). Voice availability runs
  // independently — it only governs mic/speaker visibility and must never gate
  // the loading overlay (which polls /ready on its own; see LoadingOverlay.vue).
  initTheme();

  session.onAuthFailure(handleAuthFailure);
  session.init();
  voiceStore.checkAvailability();

  // Recall button (PresenceBar) → open the moment-search dialog.
  _unbindRecall = on('chalie:open-recall', () => {
    recallRef.value?.open();
  });

  // Client context heartbeat: also surfaces auth expiry via /auth/status.
  const heartbeat = useHeartbeat();
  heartbeat.onAuthFailure(handleAuthFailure);
  heartbeat.start();
  // Port of app.js line 280: prime geolocation permission once after start.
  void heartbeat.requestLocationPermission();
});

onBeforeUnmount(() => {
  _unbindRecall?.();
  _unbindRecall = null;
  useHeartbeat().stop();
  useAmbientSensor().destroy();
});
</script>

<template>
  <!-- Ambient background layers -->
  <AmbientCanvas />
  <div id="ambientBloom"></div>
  <div id="grainOverlay"></div>

  <!-- Fixed presence bar -->
  <PresenceBar />

  <!-- Loading overlay — polls /ready, fades once the backend is up -->
  <LoadingOverlay />

  <!-- Scrollable conversation spine -->
  <ConversationFeed />

  <!-- Fixed input dock -->
  <InputDock />

  <!-- Teleport targets for dialogs / permission cards -->
  <div id="permStack" class="permission-stack"></div>
  <div id="overlayRoot"></div>

  <!-- Overlays / dialogs.
       PermissionStack teleports into #permStack; the rest self-render
       (TaskDrawer renders its own trigger; QuickTipCard / UpdatePrompt stay
       dormant until the backend emits their events; VoicePlayerDialog and
       MomentSearchDialog self-subscribe to their bus events — MomentSearchDialog
       also exposes open() for the recall button). -->
  <PermissionStack />
  <TaskDrawer />
  <QuickTipCard />
  <UpdatePrompt />
  <VoicePlayerDialog />
  <MomentSearchDialog ref="recallRef" />
</template>

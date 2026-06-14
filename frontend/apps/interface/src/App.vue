<script setup lang="ts">
import { onMounted } from 'vue';
import { useTheme } from '@chalie/shared';
import { useSessionStore } from './stores/session';
import { useVoiceStore } from './stores/voice';
import AmbientCanvas from './components/layout/AmbientCanvas.vue';
import PresenceBar from './components/layout/PresenceBar.vue';
import ConversationFeed from './components/conversation/ConversationFeed.vue';
import InputDock from './components/layout/InputDock.vue';
import LoadingOverlay from './components/layout/LoadingOverlay.vue';

const { init: initTheme } = useTheme();
const session = useSessionStore();
const voiceStore = useVoiceStore();

onMounted(() => {
  // Theme init first, then session (WS connect). Voice availability runs
  // independently — it only governs mic/speaker visibility and must never gate
  // the loading overlay (which polls /ready on its own; see LoadingOverlay.vue).
  initTheme();
  session.init();
  voiceStore.checkAvailability();
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

  <!-- Teleport targets for dialogs / permission cards (later tasks) -->
  <div id="permStack" class="permission-stack"></div>
  <div id="overlayRoot"></div>
</template>

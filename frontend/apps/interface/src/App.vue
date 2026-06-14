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
  // Theme init first, then session (WS connect), then voice availability.
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

  <!-- Loading overlay (fades after first paint; full gate wired in Task A5) -->
  <LoadingOverlay />

  <!-- Scrollable conversation spine -->
  <ConversationFeed />

  <!-- Fixed input dock -->
  <InputDock />

  <!-- Teleport targets for dialogs / permission cards (later tasks) -->
  <div id="permStack" class="permission-stack"></div>
  <div id="overlayRoot"></div>
</template>

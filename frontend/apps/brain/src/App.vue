<script setup lang="ts">
import { onBeforeUnmount, onMounted } from 'vue';
import { RouterView } from 'vue-router';
import { AskRow, getWebSocket, QuestionCard, useAsksStore, useTheme, useWaitClock } from '@chalie/shared';
import { useShellStore } from './stores/shell';
import { useHeartbeat } from './composables/useHeartbeat';
import BrainSidebar from './ui/BrainSidebar.vue';
import BrainTopbar from './ui/BrainTopbar.vue';
import CommandPalette from './ui/CommandPalette.vue';
import ConfirmDialog from './ui/ConfirmDialog.vue';
import ToastHost from './ui/ToastHost.vue';

const { init: initTheme } = useTheme();
const shell = useShellStore();
const heartbeat = useHeartbeat();
const asks = useAsksStore();
const now = useWaitClock(() => asks.queue.length);

/** Open shows an ask's turn, which lives in the chat. */
function openChat(): void {
  globalThis.location.assign('/');
}

// Registered after the palette's and the confirm dialog's own Escape listeners
// (children mount first), so an Escape one of them already handled is skipped.
function onKeydown(e: KeyboardEvent): void {
  if (e.key !== 'Escape' || e.defaultPrevented || !shell.mobileOpen) return;
  shell.closeMobileSidebar();
}

onMounted(() => {
  initTheme();
  heartbeat.start();
  document.addEventListener('keydown', onKeydown);
  // The open asks, kept current while any Brain page is open, so one is
  // answered where the user is.
  const ws = getWebSocket();
  ws.onConnect(() => void asks.refreshOpen());
  ws.onDrift((frame) => asks.route(frame));
  ws.connect();
});

onBeforeUnmount(() => {
  heartbeat.stop();
  document.removeEventListener('keydown', onKeydown);
});
</script>

<template>
  <div
    id="appShell"
    class="app-shell"
    :data-collapsed="shell.sidebarCollapsed || undefined"
    :data-mobile-open="shell.mobileOpen || undefined"
    :data-providers-only="shell.providersOnly || undefined"
  >
    <div id="mobileScrim" class="scrim" @click="shell.mobileOpen = false"></div>

    <BrainSidebar id="sidebar" />
    <BrainTopbar id="topbar" />

    <main class="main">
      <div id="panelRoot" class="main-inner">
        <RouterView />
      </div>
    </main>

    <div class="ask-dock" aria-live="polite">
      <AskRow v-for="ask in asks.rows" :key="ask.ask_id" :ask="ask" :now="now" @open="openChat" />
      <QuestionCard :questions="asks.questions" :now="now" @open="openChat" />
    </div>
  </div>

  <ToastHost id="toastHost" />

  <CommandPalette id="cpOverlay" />

  <!-- Singleton, always mounted so useConfirm() resolves -->
  <ConfirmDialog />
</template>

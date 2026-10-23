<script setup lang="ts">
import { onBeforeUnmount, onMounted } from 'vue';
import { RouterView } from 'vue-router';
import { useTheme } from '@chalie/shared';
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

onMounted(() => {
  initTheme();
  heartbeat.start();
});

onBeforeUnmount(() => {
  heartbeat.stop();
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
  </div>

  <ToastHost id="toastHost" />

  <CommandPalette id="cpOverlay" />

  <!-- Singleton, always mounted so useConfirm() resolves -->
  <ConfirmDialog />
</template>

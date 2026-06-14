<script setup lang="ts">
import { onMounted, onBeforeUnmount } from 'vue';
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

function closeMobileScrim(): void {
  shell.mobileOpen = false;
}
</script>

<template>
  <!-- Grain overlay (decorative, matches legacy .grain) -->
  <div class="grain"></div>

  <!-- Main app shell — grid: sidebar | topbar / main -->
  <div
    id="appShell"
    class="app-shell"
    :data-collapsed="shell.sidebarCollapsed || undefined"
    :data-mobile-open="shell.mobileOpen || undefined"
    :data-providers-only="shell.providersOnly || undefined"
  >
    <!-- Mobile scrim -->
    <div id="mobileScrim" class="scrim" @click="closeMobileScrim"></div>

    <!-- Sidebar -->
    <BrainSidebar id="sidebar" />

    <!-- Topbar -->
    <BrainTopbar id="topbar" />

    <!-- Main content with panel root -->
    <main class="main">
      <div id="panelRoot" class="main-inner">
        <RouterView />
      </div>
    </main>
  </div>

  <!-- Toast host (outside the grid, fixed-position) -->
  <ToastHost id="toastHost" />

  <!-- Command palette overlay -->
  <CommandPalette id="cpOverlay" />

  <!-- Confirm dialog (singleton, always mounted so useConfirm() resolves) -->
  <ConfirmDialog />
</template>

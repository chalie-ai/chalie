<!--
  BrainTopbar — port of legacy _renderTopbar() in app.js.
  Shows hamburger (mobile), collapse toggle (desktop), breadcrumb, search button.
-->
<script setup lang="ts">
import { computed } from 'vue';
import { useRoute } from 'vue-router';
import { useShellStore } from '../stores/shell';
import { Menu, PanelLeft, Search } from '@lucide/vue';

const shell = useShellStore();
const route = useRoute();

const LABELS: Record<string, string> = {
  providers: 'Providers', cognition: 'Cognition', scheduler: 'Scheduler',
  lists: 'Lists', documents: 'Documents', capabilities: 'Capabilities',
  vision: 'Vision', delegate: 'Delegate', policies: 'Policies', skills: 'Skills', mcp: 'MCP', brain: 'Brain',
  memory: 'Memory', tools: 'Tools', world: 'World state',
  personality: 'Personality', errors: 'Errors', usage: 'Usage',
  compaction: 'Compacted Summary',
  all: 'All', pending: 'Pending', fired: 'Fired', failed: 'Failed', cancelled: 'Cancelled',
  active: 'Active', processing: 'Processing', uploads: 'Uploads', deleted: 'Deleted',
  chat: 'Chat', background: 'Background', external: 'External agent',
};

const segments = computed(() => route.path.split('/').filter(Boolean));
const topLabel = computed(() => LABELS[segments.value[0]] ?? segments.value[0] ?? '');
const subLabel = computed(() => LABELS[segments.value[1]] ?? segments.value[1] ?? '');

function handleSearchKeydown(e: KeyboardEvent): void {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    shell.openCommandPalette();
  }
}
</script>

<template>
  <header class="topbar">
    <!-- Mobile hamburger -->
    <button class="icon-btn hamburger" aria-label="Open menu" @click="shell.openMobileSidebar()">
      <Menu :size="18" />
    </button>

    <!-- Desktop sidebar collapser -->
    <button
      class="icon-btn desktop-collapser"
      aria-label="Toggle sidebar"
      title="Toggle sidebar"
      @click="shell.toggleSidebar()"
    >
      <PanelLeft :size="18" />
    </button>

    <!-- Breadcrumb -->
    <div class="crumb">
      <span>Brain</span>
      <span class="sep">/</span>
      <span class="now">{{ topLabel }}</span>
      <template v-if="subLabel">
        <span class="sep">/</span>
        <span class="now">{{ subLabel }}</span>
      </template>
    </div>

    <!-- Search / command palette trigger -->
    <div class="topbar-center">
      <div
        class="topbar-search"
        role="button"
        tabindex="0"
        @click="shell.openCommandPalette()"
        @keydown="handleSearchKeydown"
      >
        <Search :size="14" />
        <span class="topbar-search-text">Search…</span>
        <kbd>⌘K</kbd>
      </div>
    </div>

    <div class="topbar-actions">
      <slot />
    </div>
  </header>
</template>

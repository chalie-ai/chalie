<!-- Hamburger (mobile), collapse toggle (desktop), breadcrumb, search button. -->
<script setup lang="ts">
import { computed } from 'vue';
import { useRoute } from 'vue-router';
import { useShellStore } from '../stores/shell';
import { Menu, PanelLeft, Search } from '@lucide/vue';
import { NAV } from './nav';

const shell = useShellStore();
const route = useRoute();

const segments = computed(() => route.path.split('/').filter(Boolean));
const item = computed(() => NAV.find((n) => n.id === segments.value[0]));
const topLabel = computed(() => item.value?.label ?? '');
const subLabel = computed(() => item.value?.sub?.find((s) => s.id === segments.value[1])?.label ?? '');

function handleSearchKeydown(e: KeyboardEvent): void {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    shell.openCommandPalette();
  }
}
</script>

<template>
  <header class="topbar">
    <button class="icon-btn hamburger" aria-label="Open menu" @click="shell.openMobileSidebar()">
      <Menu :size="18" />
    </button>

    <button
      class="icon-btn desktop-collapser"
      aria-label="Toggle sidebar"
      title="Toggle sidebar"
      @click="shell.toggleSidebar()"
    >
      <PanelLeft :size="18" />
    </button>

    <div class="crumb">
      <span>Brain</span>
      <span class="sep">/</span>
      <span class="now">{{ topLabel }}</span>
      <template v-if="subLabel">
        <span class="sep">/</span>
        <span class="now">{{ subLabel }}</span>
      </template>
    </div>

    <div class="topbar-center">
      <button
        type="button"
        class="topbar-search"
        @click="shell.openCommandPalette()"
        @keydown="handleSearchKeydown"
      >
        <Search :size="14" />
        <span class="topbar-search-text">Search…</span>
        <kbd>⌘K</kbd>
      </button>
    </div>

    <div class="topbar-actions">
      <slot />
    </div>
  </header>
</template>

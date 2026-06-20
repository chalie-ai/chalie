<!--
  BrainSidebar — port of frontend/brain/sidebar.js.
  Renders the two NAV groups (Cognition + System) with collapsible sub-lists,
  active-route highlighting via Vue Router, lock-banner when providersOnly,
  and a theme toggle button at the bottom.
-->
<script setup lang="ts">
import { computed } from 'vue';
import type { FunctionalComponent } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import {
  LayoutGrid,
  Brain,
  Calendar,
  List,
  FileText,
  Settings,
  ShieldCheck,
  BookOpen,
  Server,
  DatabaseBackup,
  ChevronRight,
  Sun,
  Moon,
} from '@lucide/vue';
import { useTheme } from '@chalie/shared';
import { useShellStore } from '../stores/shell';

const shell = useShellStore();
const route = useRoute();
const router = useRouter();
const { theme, toggle: toggleTheme } = useTheme();

// Absolute runtime URL — Flask serves /icons/icon.png regardless of Vue base path.
// Using :src (dynamic binding) prevents Vite/Rollup from attempting to resolve
// this as a local module import during build.
const brandIconUrl = '/icons/icon.png';

interface SubItem {
  id: string;
  label: string;
}

interface NavItem {
  id: string;
  label: string;
  icon: FunctionalComponent;
  group: 'cognition' | 'system';
  sub?: SubItem[];
}

// Port of BrainSidebar.NAV (sidebar.js:3-42).
const NAV: NavItem[] = [
  { id: 'providers', label: 'Providers', icon: LayoutGrid, group: 'cognition' },
  {
    id: 'cognition', label: 'Cognition', icon: Brain, group: 'cognition',
    sub: [
      { id: 'memory', label: 'Memory' },
      { id: 'tools', label: 'Tools' },
      { id: 'world', label: 'World state' },
      { id: 'personality', label: 'Personality' },
      { id: 'errors', label: 'Errors' },
      { id: 'usage', label: 'Usage' },
      { id: 'compaction', label: 'Compacted Summary' },
    ],
  },
  {
    id: 'scheduler', label: 'Scheduler', icon: Calendar, group: 'cognition',
    sub: [
      { id: 'all', label: 'All' },
      { id: 'pending', label: 'Pending' },
      { id: 'fired', label: 'Fired' },
      { id: 'failed', label: 'Failed' },
      { id: 'cancelled', label: 'Cancelled' },
    ],
  },
  { id: 'lists', label: 'Lists', icon: List, group: 'cognition' },
  {
    id: 'documents', label: 'Documents', icon: FileText, group: 'cognition',
    sub: [
      { id: 'active', label: 'Active' },
      { id: 'processing', label: 'Processing' },
      { id: 'uploads', label: 'Uploads' },
      { id: 'deleted', label: 'Deleted' },
    ],
  },
  { id: 'capabilities', label: 'Capabilities', icon: Settings, group: 'system' },
  {
    id: 'policies', label: 'Policies', icon: ShieldCheck, group: 'system',
    sub: [
      { id: 'chat', label: 'Chat' },
      { id: 'background', label: 'Background' },
      { id: 'external', label: 'External agent' },
    ],
  },
  { id: 'skills', label: 'Skills', icon: BookOpen, group: 'system' },
  { id: 'mcp', label: 'MCP', icon: Server, group: 'system' },
  { id: 'import-export', label: 'Import / Export', icon: DatabaseBackup, group: 'system' },
];

const cognitionItems = computed(() => NAV.filter((n) => n.group === 'cognition'));
const systemItems = computed(() => NAV.filter((n) => n.group === 'system'));

/** The active top-level section (first path segment after /). */
const activeSection = computed(() => route.path.split('/')[1] || 'providers');
/** The active sub-section (second path segment). */
const activeSub = computed(() => route.path.split('/')[2] || '');

function navigate(section: string, sub: string | null = null): void {
  if (shell.providersOnly && section !== 'providers') return;
  if (sub) {
    void router.push({ path: `/${section}/${sub}` });
  } else {
    void router.push({ path: `/${section}` });
  }
  if (window.innerWidth <= 900) shell.closeMobileSidebar();
}

function isActive(item: NavItem): boolean {
  return activeSection.value === item.id;
}

function isSubActive(item: NavItem, sub: SubItem): boolean {
  return isActive(item) && activeSub.value === sub.id;
}

function isExpanded(item: NavItem): boolean {
  return isActive(item) && !!item.sub && !shell.sidebarCollapsed;
}
</script>

<template>
  <aside class="sidebar">
    <!-- Brand -->
    <a class="sidebar-brand" href="/">
      <!-- eslint-disable-next-line vue/html-self-closing -->
      <img :src="brandIconUrl" alt="Chalie" width="28" height="28" />
      <div class="sidebar-brand-text">
        <div class="wordmark">Chalie</div>
        <div class="wordmark-sub">Brain</div>
      </div>
    </a>

    <!-- Navigation -->
    <nav class="sidebar-scroll">
      <!-- Providers-only lock banner -->
      <div v-if="shell.providersOnly && !shell.sidebarCollapsed" class="sidebar-lock-banner">
        <span class="sidebar-lock-icon"><LayoutGrid :size="16" /></span>
        <span>Add a provider to unlock the full dashboard.</span>
      </div>

      <!-- Cognition group -->
      <div class="nav-group">
        <div v-if="!shell.sidebarCollapsed" class="nav-group-title">Cognition</div>
        <template v-for="item in cognitionItems" :key="item.id">
          <div :data-section="item.id">
            <button
              :class="['nav-item', { active: isActive(item) }]"
              :data-nav="item.id"
              :data-expanded="isExpanded(item)"
              @click="navigate(item.id, item.sub ? item.sub[0].id : null)"
            >
              <span class="nav-icon"><component :is="item.icon" :size="18" /></span>
              <span class="nav-label">{{ item.label }}</span>
              <span v-if="item.sub" class="nav-chev"><ChevronRight :size="14" /></span>
            </button>
            <div v-if="item.sub" class="nav-sublist" :data-open="isExpanded(item)">
              <div>
                <button
                  v-for="sub in item.sub"
                  :key="sub.id"
                  :class="['nav-sub', { active: isSubActive(item, sub) }]"
                  :data-sub="sub.id"
                  @click="navigate(item.id, sub.id)"
                >
                  <span>{{ sub.label }}</span>
                </button>
              </div>
            </div>
          </div>
        </template>
      </div>

      <!-- System group -->
      <div class="nav-group">
        <div v-if="!shell.sidebarCollapsed" class="nav-group-title">System</div>
        <template v-for="item in systemItems" :key="item.id">
          <div :data-section="item.id">
            <button
              :class="['nav-item', { active: isActive(item) }]"
              :data-nav="item.id"
              :data-expanded="isExpanded(item)"
              @click="navigate(item.id, item.sub ? item.sub[0].id : null)"
            >
              <span class="nav-icon"><component :is="item.icon" :size="18" /></span>
              <span class="nav-label">{{ item.label }}</span>
              <span v-if="item.sub" class="nav-chev"><ChevronRight :size="14" /></span>
            </button>
          </div>
        </template>
      </div>
    </nav>

    <!-- Footer: theme toggle -->
    <div class="sidebar-footer">
      <button
        type="button"
        class="icon-btn theme-toggle"
        aria-label="Toggle theme"
        @click="toggleTheme"
      >
        <component :is="theme === 'dark' ? Sun : Moon" :size="18" />
      </button>
    </div>
  </aside>
</template>

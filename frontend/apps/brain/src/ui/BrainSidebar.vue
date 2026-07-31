<!-- Two NAV groups with collapsible sub-lists, active-route highlight, providersOnly lock banner, theme toggle. -->
<script setup lang="ts">
import type { FunctionalComponent } from 'vue';
import { computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import {
  BookOpen,
  Brain,
  Calendar,
  ChevronRight,
  DatabaseBackup,
  LayoutGrid,
  List,
  Moon,
  Network,
  Server,
  Settings,
  ShieldCheck,
  Smartphone,
  Sun,
} from '@lucide/vue';
import { useTheme } from '@chalie/shared';
import { useShellStore } from '../stores/shell';

const shell = useShellStore();
const route = useRoute();
const router = useRouter();
const { theme, toggle: toggleTheme } = useTheme();

// Absolute runtime URL; :src (dynamic binding) stops Vite/Rollup resolving it as a build-time module import.
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

const NAV: NavItem[] = [
  { id: 'providers', label: 'Providers', icon: LayoutGrid, group: 'cognition' },
  {
    id: 'cognition',
    label: 'Cognition',
    icon: Brain,
    group: 'cognition',
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
  { id: 'scheduler', label: 'Scheduler', icon: Calendar, group: 'cognition' },
  { id: 'lists', label: 'Lists', icon: List, group: 'cognition' },
  { id: 'capabilities', label: 'Capabilities', icon: Settings, group: 'system' },
  {
    id: 'policies',
    label: 'Policies',
    icon: ShieldCheck,
    group: 'system',
    sub: [
      { id: 'chat', label: 'Chat' },
      { id: 'background', label: 'Background' },
      { id: 'external', label: 'External agent' },
    ],
  },
  { id: 'skills', label: 'Skills', icon: BookOpen, group: 'system' },
  { id: 'mcp', label: 'MCP', icon: Server, group: 'system' },
  { id: 'import-export', label: 'Import / Export', icon: DatabaseBackup, group: 'system' },
  { id: 'system', label: 'System', icon: Network, group: 'system' },
  { id: 'link-device', label: 'Link device', icon: Smartphone, group: 'system' },
];

// Both nav groups render from one template; link-device is an in-development feature hidden unless the backend reports it on.
const NAV_GROUPS: { title: string; group: NavItem['group'] }[] = [
  { title: 'Cognition', group: 'cognition' },
  { title: 'System', group: 'system' },
];
const navGroups = computed(() =>
  NAV_GROUPS.map((g) => ({
    ...g,
    items: NAV.filter((n) => n.group === g.group && (n.id !== 'link-device' || shell.internalDev)),
  })),
);

const activeSection = computed(() => route.path.split('/')[1] || 'providers');
const activeSub = computed(() => route.path.split('/')[2] || '');

function navigate(section: string, sub: string | null = null): void {
  if (shell.providersOnly && section !== 'providers') return;
  router.push({ path: sub ? `/${section}/${sub}` : `/${section}` });
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
    <a class="sidebar-brand" href="/">
      <!-- eslint-disable-next-line vue/html-self-closing -->
      <img :src="brandIconUrl" alt="Chalie" width="28" height="28" />
      <div class="sidebar-brand-text">
        <div class="wordmark">Chalie</div>
        <div class="wordmark-sub">Brain</div>
      </div>
    </a>

    <nav class="sidebar-scroll">
      <div v-if="shell.providersOnly && !shell.sidebarCollapsed" class="sidebar-lock-banner">
        <span class="sidebar-lock-icon"><LayoutGrid :size="16" /></span>
        <span>Add a provider to unlock the full dashboard.</span>
      </div>

      <div v-for="g in navGroups" :key="g.group" class="nav-group">
        <div v-if="!shell.sidebarCollapsed" class="nav-group-title">{{ g.title }}</div>
        <template v-for="item in g.items" :key="item.id">
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
    </nav>

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

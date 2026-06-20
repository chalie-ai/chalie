<!-- Opened by ⌘K/Ctrl-K or the topbar search button; visibility driven by shell.commandPaletteOpen. -->
<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted, onBeforeUnmount } from 'vue';
import type { FunctionalComponent } from 'vue';
import { useRouter } from 'vue-router';
import {
  LayoutGrid, History, Wrench, Globe, Sparkles, TriangleAlert, ChartLine,
  FileText, Calendar, List, Upload, Trash2, Settings, ShieldCheck, BookOpen,
  Server, Brain,
} from '@lucide/vue';
import { useShellStore } from '../stores/shell';

const shell = useShellStore();
const router = useRouter();

const query = ref('');
const selectedIdx = ref(0);
const inputRef = ref<HTMLInputElement | null>(null);

interface CPItem {
  kind: string;
  label: string;
  icon: FunctionalComponent;
  path: string;
}

const ALL_ITEMS: CPItem[] = [
  { kind: 'Jump', label: 'Providers', icon: LayoutGrid, path: '/providers' },
  { kind: 'Jump', label: 'Cognition · Memory', icon: History, path: '/cognition/memory' },
  { kind: 'Jump', label: 'Cognition · Tools', icon: Wrench, path: '/cognition/tools' },
  { kind: 'Jump', label: 'Cognition · World state', icon: Globe, path: '/cognition/world' },
  { kind: 'Jump', label: 'Cognition · Personality', icon: Sparkles, path: '/cognition/personality' },
  { kind: 'Jump', label: 'Cognition · Errors', icon: TriangleAlert, path: '/cognition/errors' },
  { kind: 'Jump', label: 'Cognition · Usage', icon: ChartLine, path: '/cognition/usage' },
  { kind: 'Jump', label: 'Cognition · Compacted Summary', icon: FileText, path: '/cognition/compaction' },
  { kind: 'Jump', label: 'Scheduler · All', icon: Calendar, path: '/scheduler/all' },
  { kind: 'Jump', label: 'Scheduler · Pending', icon: Calendar, path: '/scheduler/pending' },
  { kind: 'Jump', label: 'Scheduler · Fired', icon: Calendar, path: '/scheduler/fired' },
  { kind: 'Jump', label: 'Scheduler · Failed', icon: Calendar, path: '/scheduler/failed' },
  { kind: 'Jump', label: 'Scheduler · Cancelled', icon: Calendar, path: '/scheduler/cancelled' },
  { kind: 'Jump', label: 'Lists', icon: List, path: '/lists' },
  { kind: 'Jump', label: 'Documents · Active', icon: FileText, path: '/documents/active' },
  { kind: 'Jump', label: 'Documents · Processing', icon: FileText, path: '/documents/processing' },
  { kind: 'Jump', label: 'Documents · Uploads', icon: Upload, path: '/documents/uploads' },
  { kind: 'Jump', label: 'Documents · Deleted', icon: Trash2, path: '/documents/deleted' },
  { kind: 'Jump', label: 'Capabilities', icon: Settings, path: '/capabilities' },
  { kind: 'Jump', label: 'Policies · Chat', icon: ShieldCheck, path: '/policies/chat' },
  { kind: 'Jump', label: 'Policies · Background', icon: ShieldCheck, path: '/policies/background' },
  { kind: 'Jump', label: 'Policies · External agent', icon: ShieldCheck, path: '/policies/external' },
  { kind: 'Jump', label: 'Skills', icon: BookOpen, path: '/skills' },
  { kind: 'Jump', label: 'MCP', icon: Server, path: '/mcp' },
  { kind: 'Jump', label: 'Import / Export', icon: Brain, path: '/import-export' },
];

const filtered = computed(() => {
  const q = query.value.toLowerCase();
  return q ? ALL_ITEMS.filter((i) => i.label.toLowerCase().includes(q)) : ALL_ITEMS;
});

watch(query, () => { selectedIdx.value = 0; });
watch(() => shell.commandPaletteOpen, (v) => {
  if (v) {
    query.value = '';
    selectedIdx.value = 0;
    void nextTick(() => inputRef.value?.focus());
  }
});

function selectItem(item: CPItem): void {
  if (shell.providersOnly && item.path !== '/providers') return;
  void router.push(item.path);
  shell.closeCommandPalette();
}

function onKeydown(e: KeyboardEvent): void {
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    selectedIdx.value = Math.min(filtered.value.length - 1, selectedIdx.value + 1);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    selectedIdx.value = Math.max(0, selectedIdx.value - 1);
  } else if (e.key === 'Enter') {
    e.preventDefault();
    const item = filtered.value[selectedIdx.value];
    if (item) selectItem(item);
  } else if (e.key === 'Escape') {
    shell.closeCommandPalette();
  }
}

function onGlobalKeydown(e: KeyboardEvent): void {
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
    e.preventDefault();
    shell.toggleCommandPalette();
  } else if (e.key === 'Escape' && shell.commandPaletteOpen) {
    shell.closeCommandPalette();
  }
}

onMounted(() => document.addEventListener('keydown', onGlobalKeydown));
onBeforeUnmount(() => document.removeEventListener('keydown', onGlobalKeydown));
</script>

<template>
  <div :class="['cp-overlay', { hidden: !shell.commandPaletteOpen }]" @click.self="shell.closeCommandPalette()">
    <div v-if="shell.commandPaletteOpen" class="cp">
      <input
        ref="inputRef"
        v-model="query"
        class="cp-input"
        placeholder="Search the brain… (try 'memory', 'policy')"
        @keydown="onKeydown"
      />
      <div class="cp-results">
        <template v-if="filtered.length > 0">
          <div class="cp-section">Jump</div>
          <div
            v-for="(item, idx) in filtered"
            :key="item.path"
            :class="['cp-row', { selected: idx === selectedIdx }]"
            :data-section="item.path.split('/')[1]"
            :data-sub="item.path.split('/')[2] || ''"
            @click="selectItem(item)"
          >
            <span class="cp-icon"><component :is="item.icon" :size="14" /></span>
            <span>{{ item.label }}</span>
            <span v-if="idx === selectedIdx" class="cp-hint">↵</span>
          </div>
        </template>
        <div v-else class="cp-empty">No matches. Try fewer words.</div>
      </div>
      <div class="cp-footer">
        <span><kbd>↑</kbd><kbd>↓</kbd> navigate · <kbd>↵</kbd> open · <kbd>esc</kbd> close</span>
        <span>{{ filtered.length }} result{{ filtered.length === 1 ? '' : 's' }}</span>
      </div>
    </div>
  </div>
</template>

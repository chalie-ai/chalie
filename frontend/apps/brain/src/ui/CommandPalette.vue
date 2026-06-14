<!--
  CommandPalette — port of legacy _renderCommandPalette() in app.js.
  Opened by ⌘K/Ctrl-K or the topbar search button.
  Provides the openCommandPalette injection consumed by BrainTopbar.
-->
<script setup lang="ts">
import { ref, computed, watch, nextTick, provide, onMounted, onBeforeUnmount } from 'vue';
import { useRouter } from 'vue-router';
import { useShellStore } from '../stores/shell';
import BrainIcon from './BrainIcon.vue';

const shell = useShellStore();
const router = useRouter();

const visible = ref(false);
const query = ref('');
const selectedIdx = ref(0);
const inputRef = ref<HTMLInputElement | null>(null);

interface CPItem {
  kind: string;
  label: string;
  icon: string;
  path: string;
}

// Port of allItems in _renderCommandPalette (app.js:162-178).
const ALL_ITEMS: CPItem[] = [
  { kind: 'Jump', label: 'Providers', icon: 'Providers', path: '/providers' },
  { kind: 'Jump', label: 'Vision', icon: 'Eye', path: '/vision' },
  { kind: 'Jump', label: 'Cognition · Memory', icon: 'Memory', path: '/cognition/memory' },
  { kind: 'Jump', label: 'Cognition · Tools', icon: 'Tool', path: '/cognition/tools' },
  { kind: 'Jump', label: 'Cognition · World state', icon: 'Globe', path: '/cognition/world' },
  { kind: 'Jump', label: 'Cognition · Personality', icon: 'Sparkles', path: '/cognition/personality' },
  { kind: 'Jump', label: 'Cognition · Errors', icon: 'Alert', path: '/cognition/errors' },
  { kind: 'Jump', label: 'Cognition · Usage', icon: 'Chart', path: '/cognition/usage' },
  { kind: 'Jump', label: 'Cognition · Compacted Summary', icon: 'Document', path: '/cognition/compaction' },
  { kind: 'Jump', label: 'Scheduler · All', icon: 'Calendar', path: '/scheduler/all' },
  { kind: 'Jump', label: 'Scheduler · Pending', icon: 'Calendar', path: '/scheduler/pending' },
  { kind: 'Jump', label: 'Scheduler · Fired', icon: 'Calendar', path: '/scheduler/fired' },
  { kind: 'Jump', label: 'Scheduler · Failed', icon: 'Calendar', path: '/scheduler/failed' },
  { kind: 'Jump', label: 'Scheduler · Cancelled', icon: 'Calendar', path: '/scheduler/cancelled' },
  { kind: 'Jump', label: 'Lists', icon: 'List', path: '/lists' },
  { kind: 'Jump', label: 'Documents · Active', icon: 'Document', path: '/documents/active' },
  { kind: 'Jump', label: 'Documents · Processing', icon: 'Document', path: '/documents/processing' },
  { kind: 'Jump', label: 'Documents · Uploads', icon: 'Upload', path: '/documents/uploads' },
  { kind: 'Jump', label: 'Documents · Deleted', icon: 'Trash', path: '/documents/deleted' },
  { kind: 'Jump', label: 'Capabilities', icon: 'Capability', path: '/capabilities' },
  { kind: 'Jump', label: 'Policies · Chat', icon: 'Policy', path: '/policies/chat' },
  { kind: 'Jump', label: 'Policies · Background', icon: 'Policy', path: '/policies/background' },
  { kind: 'Jump', label: 'Policies · External agent', icon: 'Policy', path: '/policies/external' },
  { kind: 'Jump', label: 'Skills', icon: 'Skill', path: '/skills' },
  { kind: 'Jump', label: 'MCP', icon: 'Server', path: '/mcp' },
  { kind: 'Jump', label: 'Brain', icon: 'Brain', path: '/brain' },
];

const filtered = computed(() => {
  const q = query.value.toLowerCase();
  return q ? ALL_ITEMS.filter((i) => i.label.toLowerCase().includes(q)) : ALL_ITEMS;
});

watch(query, () => { selectedIdx.value = 0; });
watch(visible, (v) => {
  if (v) {
    selectedIdx.value = 0;
    void nextTick(() => inputRef.value?.focus());
  }
});

function open(): void {
  query.value = '';
  visible.value = true;
}

function close(): void {
  visible.value = false;
}

function selectItem(item: CPItem): void {
  if (shell.providersOnly && item.path !== '/providers') return;
  void router.push(item.path);
  close();
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
    close();
  }
}

function onGlobalKeydown(e: KeyboardEvent): void {
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
    e.preventDefault();
    if (visible.value) close(); else open();
  } else if (e.key === 'Escape' && visible.value) {
    close();
  }
}

onMounted(() => document.addEventListener('keydown', onGlobalKeydown));
onBeforeUnmount(() => document.removeEventListener('keydown', onGlobalKeydown));

// Provide the open function so BrainTopbar can call it via inject.
provide('openCommandPalette', open);
</script>

<template>
  <div :class="['cp-overlay', { hidden: !visible }]" @click.self="close">
    <div class="cp" v-if="visible">
      <input
        ref="inputRef"
        class="cp-input"
        v-model="query"
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
            <span class="cp-icon"><BrainIcon :name="item.icon" :size="14" /></span>
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

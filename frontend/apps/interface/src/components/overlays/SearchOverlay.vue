<script setup lang="ts">
import { nextTick, ref, watch } from 'vue';
import { Search } from '@lucide/vue';
import { ConfigType } from '@chalie/shared';
import { useSessionStore } from '../../stores/session';
import { conversation as convoApi } from '../../api/conversation';
import type { ConversationThread } from '../../api/conversation';

const session = useSessionStore();

/** Thread search — a direct GET /api/threads/all?q=, no client-side cache. */
async function searchThreads(q: string): Promise<ConversationThread[]> {
  if (!q.trim()) return [];
  try {
    const { threads } = await convoApi.threads(5, undefined, q, ConfigType.USER);
    return threads;
  } catch (e) {
    console.warn('Thread search failed', e);
    return [];
  }
}

const query = ref('');
const results = ref<ConversationThread[]>([]);
const inputEl = ref<HTMLInputElement | null>(null);

let debounceTimer: ReturnType<typeof setTimeout> | null = null;

watch(
  () => session.searchOpen,
  (open) => {
    if (open) {
      query.value = '';
      results.value = [];
      document.body.classList.add('no-scroll');
      nextTick(() => inputEl.value?.focus());
    } else {
      document.body.classList.remove('no-scroll');
    }
  },
);

function onInput(e: Event): void {
  query.value = (e.target as HTMLInputElement).value;
  if (debounceTimer !== null) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(async () => {
    results.value = await searchThreads(query.value);
  }, 120);
}

function onKey(e: KeyboardEvent): void {
  if (e.key === 'Escape') session.closeSearch();
}

function pick(item: ConversationThread): void {
  if (item.turn_id === null) return;
  session.openThreadPanel(item.turn_id, item.type);
  session.closeSearch();
}
</script>

<template>
  <Teleport to="body">
    <div
      v-if="session.searchOpen"
      class="search-scrim"
      @click.self="session.closeSearch()"
      @keydown="onKey"
    >
      <div class="search-modal" @click.stop>
        <div class="search-input-row">
          <Search :size="16" stroke="var(--pink-text)" :stroke-width="2" />
          <input
            ref="inputEl"
            class="search-input"
            placeholder="Search threads…"
            aria-label="Search threads"
            autocomplete="off"
            @input="onInput"
            @keydown="onKey"
          />
          <span class="esc-chip">esc</span>
        </div>

        <div v-if="query.trim()" class="search-results">
          <div v-if="results.length === 0" class="empty-state">No matches.</div>

          <button
            v-for="item in results"
            :key="item.turn_id ?? item.preview"
            class="result-row"
            :disabled="item.turn_id === null"
            @click="pick(item)"
          >
            <span class="result-body">
              <span class="result-name">{{ item.gist ?? item.preview }}</span>
              <span class="result-snippet">{{ item.preview }}</span>
            </span>
            <span class="result-tag">thread</span>
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.search-scrim {
  position: fixed;
  inset: 0;
  z-index: 1200;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 78px;
  animation: fade var(--dur-2) var(--ease-out);
  font-family: var(--font-ui);
}

.search-modal {
  width: 580px;
  max-width: 92vw;
  background: var(--surface);
  overflow: hidden;
}

.search-input-row {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 14px 16px;
}

.search-input {
  flex: 1;
  font:
    400 var(--fs-body) var(--font-ui);
  color: var(--text);
  background: transparent;
  border: 0;
  outline: 0;
}

.search-input::placeholder {
  color: var(--muted);
}

.esc-chip {
  font:
    600 var(--fs-mono) var(--font-ui);
  color: var(--muted);
  padding: 2px 6px;
  flex-shrink: 0;
}

.search-results {
  max-height: 48vh;
  overflow: auto;
  padding: 8px;
}

.empty-state {
  padding: 22px;
  text-align: center;
  font:
    400 var(--fs-body) var(--font-ui);
  color: var(--muted);
}

.result-row {
  display: flex;
  align-items: center;
  gap: 11px;
  width: 100%;
  text-align: left;
  background: transparent;
  padding: 10px 11px;
  cursor: pointer;
  color: var(--text);
  transition:
    background-color var(--dur-1) var(--ease-out),
    translate var(--dur-1) var(--ease-out);
}

.result-row:disabled {
  cursor: default;
  color: var(--muted);
}

.result-row:not(:disabled):hover {
  background: var(--cell);
}

.result-row:not(:disabled):active {
  translate: 0 1px;
}

.result-body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.result-name {
  display: block;
  font:
    600 var(--fs-body) var(--font-ui);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.result-snippet {
  display: block;
  font:
    400 var(--fs-body) var(--font-ui);
  color: var(--muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.result-tag {
  font:
    600 var(--fs-mono) var(--font-ui);
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--muted);
  padding: 2px 6px;
  flex-shrink: 0;
}
</style>

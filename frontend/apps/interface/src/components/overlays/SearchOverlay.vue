<script setup lang="ts">
import { ref, watch, nextTick } from 'vue';
import { Search } from '@lucide/vue';
import { useSessionStore } from '../../stores/session';
import { useConversationFeed } from '../../composables/useConversationFeed';
import type { ConversationThread } from '../../api/conversation';

const session = useSessionStore();
const feed = useConversationFeed();

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
    results.value = await feed.searchThreads(query.value);
  }, 120);
}

function onKey(e: KeyboardEvent): void {
  if (e.key === 'Escape') session.closeSearch();
}

function pick(item: ConversationThread): void {
  if (item.turn_id === null) return;
  session.openThreadPanel(item.turn_id);
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
          <Search :size="16" stroke="var(--violet)" :stroke-width="2" />
          <input
            ref="inputEl"
            class="search-input"
            placeholder="Search threads…"
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
  background: var(--scrim-overlay);
  backdrop-filter: blur(3px);
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 78px;
  animation: overlayIn 0.15s ease;
  font-family: Inter, sans-serif;
}

.search-modal {
  width: 580px;
  max-width: 92vw;
  background: var(--bg-2);
  border: 1px solid color-mix(in oklab, var(--violet) 18%, transparent);
  border-radius: 16px;
  box-shadow:
    0 24px 64px rgba(0, 0, 0, 0.6),
    0 0 0 1px color-mix(in oklab, var(--violet) 8%, transparent);
  overflow: hidden;
}

.search-input-row {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 14px 16px;
  border-bottom: 1px solid var(--border);
}

.search-input {
  flex: 1;
  font: 400 14px Inter, sans-serif;
  color: var(--text-primary);
  background: transparent;
  border: 0;
  outline: 0;
}

.search-input::placeholder {
  color: var(--text-muted);
}

.esc-chip {
  font: 600 10px Inter, sans-serif;
  color: var(--text-muted);
  border: 1px solid var(--border-strong);
  border-radius: 5px;
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
  font: 400 13px Inter, sans-serif;
  color: var(--text-muted);
}

.result-row {
  display: flex;
  align-items: center;
  gap: 11px;
  width: 100%;
  text-align: left;
  background: transparent;
  border: none;
  border-radius: 10px;
  padding: 10px 11px;
  cursor: pointer;
  color: var(--text-primary);
}

.result-row:disabled {
  cursor: default;
  opacity: 0.5;
}

.result-row:not(:disabled):hover {
  background: color-mix(in oklab, var(--violet) 8%, transparent);
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
  font: 600 12.5px Inter, sans-serif;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.result-snippet {
  display: block;
  font: 400 11.5px Inter, sans-serif;
  color: var(--text-tertiary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.result-tag {
  font: 600 9px Inter, sans-serif;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--text-muted);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 2px 6px;
  flex-shrink: 0;
}
</style>

<script setup lang="ts">
/**
 * ThreadSearchDialog — semantic thread search overlay (replaces the deleted
 * recall dialog). Searches the thread_gist index via GET /threads/search.
 */
import { ref, computed, watch, onMounted, onBeforeUnmount, nextTick } from 'vue';
import { Search, X, Loader2 } from '@lucide/vue';
import { useSessionStore } from '../../stores/session';
import { useConversationStore } from '../../stores/conversation';
import { on } from '../../composables/useEventBus';
import type { ThreadSearchResult } from '../../api/conversation';

const session = useSessionStore();
const conversationStore = useConversationStore();

const visible = ref(false);
const query = ref('');
const results = ref<ThreadSearchResult[]>([]);
const loading = ref(false);
const error = ref<string | null>(null);
const inputRef = ref<HTMLInputElement | null>(null);

let _searchTimer: ReturnType<typeof setTimeout> | null = null;
let _unbindBus: (() => void) | null = null;

const hasQuery = computed(() => query.value.trim().length > 0);

// Debounced search — fires 350ms after the last keystroke.
watch(query, (val) => {
  if (_searchTimer) clearTimeout(_searchTimer);
  const trimmed = val.trim();
  if (!trimmed) {
    results.value = [];
    error.value = null;
    return;
  }
  _searchTimer = setTimeout(runSearch, 350);
});

async function runSearch(): Promise<void> {
  const q = query.value.trim();
  if (!q) return;
  loading.value = true;
  error.value = null;
  try {
    const data = await session.searchThreads(q);
    results.value = data.results ?? [];
  } catch {
    error.value = 'Search failed. Please try again.';
    results.value = [];
  } finally {
    loading.value = false;
  }
}

function open(): void {
  visible.value = true;
  nextTick(() => inputRef.value?.focus());
}

function close(): void {
  visible.value = false;
  query.value = '';
  results.value = [];
  error.value = null;
}

function onKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape' && visible.value) close();
}

/** Click a result: open the thread in the slide-over panel and close the dialog. */
async function onResultClick(result: ThreadSearchResult): Promise<void> {
  close();
  // Pull the thread into the list if it sits beyond the loaded pages, so the
  // panel header can show its gist; the panel loads the rows by turn_id itself.
  while (
    !conversationStore.threadsExhausted &&
    !conversationStore.threads.some((t) => t.turn_id === result.turn_id)
  ) {
    await session.loadRecentConversation();
  }
  await session.openThreadPanel(result.turn_id);
}

onMounted(() => {
  document.addEventListener('keydown', onKeydown);
  _unbindBus = on('chalie:open-thread-search', () => open());
});

onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKeydown);
  if (_searchTimer) clearTimeout(_searchTimer);
  _unbindBus?.();
});
</script>

<template>
  <Transition name="search-fade">
    <div v-if="visible" class="search-overlay" @click.self="close">
      <div class="search-dialog">
        <div class="search-dialog__header">
          <Search class="search-dialog__icon" :size="18" aria-hidden="true" />
          <input
            ref="inputRef"
            v-model="query"
            class="search-dialog__input"
            type="text"
            placeholder="Search threads..."
            aria-label="Search threads"
          />
          <button
            class="search-dialog__close"
            type="button"
            aria-label="Close search"
            @click="close"
          >
            <X :size="16" />
          </button>
        </div>

        <div class="search-dialog__body">
          <div v-if="loading" class="search-dialog__loading">
            <Loader2 class="search-dialog__spinner" :size="20" />
            <span>Searching...</span>
          </div>

          <div v-else-if="error" class="search-dialog__error">{{ error }}</div>

          <div v-else-if="hasQuery && !results.length" class="search-dialog__empty">
            No matching threads found.
          </div>

          <ul v-else-if="results.length" class="search-dialog__results">
            <li
              v-for="r in results"
              :key="r.turn_id"
              class="search-result"
              :data-turn-id="r.turn_id"
              @click="onResultClick(r)"
            >
              <div class="search-result__gist">{{ r.summary }}</div>
              <div class="search-result__meta">
                <span class="search-result__turn">Thread #{{ r.turn_id }}</span>
                <span v-if="r.updated_at" class="search-result__date">{{ r.updated_at }}</span>
              </div>
            </li>
          </ul>

          <div v-else class="search-dialog__hint">
            Search across all thread summaries. Type to find conversations by topic.
          </div>
        </div>
      </div>
    </div>
  </Transition>
</template>

<style scoped lang="scss">
.search-overlay {
  position: fixed;
  inset: 0;
  z-index: 200;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding-top: 12vh;
  background: color-mix(in oklab, var(--bg) 60%, transparent);
  backdrop-filter: blur(8px) saturate(120%);
  -webkit-backdrop-filter: blur(8px) saturate(120%);
}

.search-dialog {
  width: min(640px, calc(100vw - 32px));
  max-height: 70vh;
  display: flex;
  flex-direction: column;
  background: var(--bg-2);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-card);
  overflow: hidden;
}

.search-dialog__header {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-md) var(--space-lg);
  border-bottom: 1px solid var(--border);
}

.search-dialog__icon {
  color: var(--text-tertiary);
  flex-shrink: 0;
}

.search-dialog__input {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  color: var(--text-primary);
  font-size: 1rem;
  padding: var(--space-xs) 0;
}

.search-dialog__input::placeholder {
  color: var(--text-tertiary);
}

.search-dialog__close {
  flex-shrink: 0;
  background: transparent;
  border: none;
  color: var(--text-tertiary);
  cursor: pointer;
  padding: var(--space-xs);
  border-radius: var(--radius-sm);
  transition: color var(--duration-fast), background var(--duration-fast);
}

.search-dialog__close:hover {
  color: var(--text-primary);
  background: var(--bg-surface);
}

.search-dialog__body {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-sm) 0;
}

.search-dialog__loading,
.search-dialog__empty,
.search-dialog__error,
.search-dialog__hint {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-lg);
  color: var(--text-secondary);
  font-size: var(--font-size-sm);
  justify-content: center;
}

.search-dialog__error {
  color: var(--error);
}

.search-dialog__spinner {
  animation: history-spin 0.7s linear infinite;
}

@keyframes history-spin {
  to { transform: rotate(360deg); }
}

.search-dialog__results {
  list-style: none;
  margin: 0;
  padding: 0;
}

.search-result {
  padding: var(--space-md) var(--space-lg);
  cursor: pointer;
  border-bottom: 1px solid var(--border-subtle);
  transition: background var(--duration-fast);
}

.search-result:hover {
  background: color-mix(in oklab, var(--violet) 5%, transparent);
}

.search-result:last-child {
  border-bottom: none;
}

.search-result__gist {
  font-size: 0.875rem;
  color: var(--text-primary);
  line-height: 1.4;
  margin-bottom: var(--space-xs);
}

.search-result__meta {
  display: flex;
  align-items: center;
  gap: var(--space-md);
  font-size: 0.72rem;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
}

.search-result__turn {
  letter-spacing: 0.04em;
}

/* Transition */
.search-fade-enter-active,
.search-fade-leave-active {
  transition: opacity 200ms ease;
}

.search-fade-enter-from,
.search-fade-leave-to {
  opacity: 0;
}
</style>
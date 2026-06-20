<script setup lang="ts">
/**
 * MomentSearchDialog — full-screen recall overlay (recall-only; the Remember/pin
 * flow lives in the session store). Opened via the exposed open().
 *
 * Rendering: the backend returns { value, message_text } per item. We render
 * value as the primary label and message_text as the body, avoiding the
 * non-existent title/topic/summary keys.
 */
import { ref, onBeforeUnmount } from 'vue';
import { X } from '@lucide/vue';
import { moments } from '../../api/moments';
import type { Moment } from '../../api/moments';

const DEBOUNCE_MS = 500;
const FOCUS_DEFER_MS = 100;

const dialogRef = ref<HTMLDialogElement | null>(null);
const inputRef = ref<HTMLInputElement | null>(null);

type ViewState = 'empty' | 'loading' | 'results' | 'error';

const query = ref('');
const viewState = ref<ViewState>('empty');
const results = ref<Moment[]>([]);
let debounceTimer: ReturnType<typeof setTimeout> | null = null;

function openRecall(): void {
  query.value = '';
  results.value = [];
  viewState.value = 'empty';
  dialogRef.value?.showModal();
  setTimeout(() => inputRef.value?.focus(), FOCUS_DEFER_MS);
}

function close(): void {
  if (debounceTimer !== null) {
    clearTimeout(debounceTimer);
    debounceTimer = null;
  }
  dialogRef.value?.close();
}

function handleInput(): void {
  if (debounceTimer !== null) clearTimeout(debounceTimer);

  const q = query.value.trim();
  if (!q) {
    viewState.value = 'empty';
    results.value = [];
    return;
  }

  viewState.value = 'loading';
  debounceTimer = setTimeout(() => { void runSearch(q); }, DEBOUNCE_MS);
}

async function runSearch(q: string): Promise<void> {
  try {
    const data = await moments.search(q);
    results.value = data.items ?? [];
    viewState.value = 'results';
  } catch (err) {
    console.warn('[MomentSearchDialog] search failed:', err);
    results.value = [];
    viewState.value = 'error';
  }
}

// Native Escape key via <dialog>.
function handleCancel(e: Event): void {
  e.preventDefault();
  close();
}

onBeforeUnmount(() => {
  if (debounceTimer !== null) clearTimeout(debounceTimer);
});

defineExpose({ open: openRecall });
</script>

<template>
  <dialog
    ref="dialogRef"
    class="moment-search-dialog"
    aria-label="Recall"
    @cancel="handleCancel"
  >
    <div class="moment-search-dialog__content">
      <div class="moment-search-dialog__header">
        <h2 class="moment-search-dialog__title">Recall</h2>
        <button
          class="moment-search-dialog__close btn-icon"
          aria-label="Close"
          @click="close"
        >
          <X :size="16" aria-hidden="true" />
        </button>
      </div>

      <input
        ref="inputRef"
        v-model="query"
        type="text"
        class="moment-search-dialog__input"
        placeholder="Recall something..."
        autocomplete="off"
        @input="handleInput"
      />

      <div class="moment-search-dialog__results">
        <div v-if="viewState === 'loading'" class="moment-search-dialog__shimmer">
          <div></div>
          <div></div>
          <div></div>
        </div>

        <template v-else-if="viewState === 'results'">
          <template v-if="results.length > 0">
            <div
              v-for="item in results"
              :key="String(item.id ?? item.transcript_id ?? item.key)"
              class="moment-search-dialog__item"
              role="button"
              tabindex="0"
              @click="close"
              @keydown.enter="close"
            >
              <div class="moment-search-dialog__item-title">{{ item.value || item.key }}</div>
              <div
                v-if="item.message_text && item.message_text !== item.value"
                class="moment-search-dialog__item-text"
              >{{ item.message_text }}</div>
            </div>
          </template>
          <div v-else class="moment-search-dialog__empty">
            I couldn't recall anything like that yet.
          </div>
        </template>

        <div v-else-if="viewState === 'error'" class="moment-search-dialog__empty">
          Something went wrong. Try again.
        </div>

        <div v-else class="moment-search-dialog__empty">
          Your remembered answers will appear here.
        </div>
      </div>
    </div>
  </dialog>
</template>

<style scoped lang="scss">
.moment-search-dialog {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  margin: 0;
  padding: 0;
  border: none;
  border-radius: 12px;
  background: var(--bg-2);
  color: var(--text-primary);
  box-shadow: 0 8px 48px rgba(0, 0, 0, 0.28);
  width: min(560px, calc(100vw - 32px));
  max-height: calc(100vh - 64px);
  overflow: hidden;
  z-index: 300;

  &::backdrop {
    background: rgba(0, 0, 0, 0.45);
  }
}

.moment-search-dialog__content {
  display: flex;
  flex-direction: column;
  max-height: calc(100vh - 64px);
}

.moment-search-dialog__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 16px 0;
  flex-shrink: 0;
}

.moment-search-dialog__title {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
}

.moment-search-dialog__close {
  color: var(--text-secondary);

  &:hover {
    color: var(--text-primary);
  }
}

.moment-search-dialog__input {
  display: block;
  width: 100%;
  box-sizing: border-box;
  margin: 12px 0 0;
  padding: 10px 16px;
  border: none;
  border-bottom: 1px solid var(--border);
  border-top: 1px solid var(--border);
  background: transparent;
  color: var(--text-primary);
  font-size: 15px;
  outline: none;
  flex-shrink: 0;

  &::placeholder {
    color: var(--text-tertiary, var(--text-secondary));
  }
}

.moment-search-dialog__results {
  flex: 1;
  overflow-y: auto;
  padding: 8px 0;
  min-height: 80px;
}

.moment-search-dialog__shimmer {
  padding: 8px 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;

  > div {
    height: 40px;
    border-radius: 6px;
    background: linear-gradient(
      90deg,
      var(--bg-surface) 25%,
      var(--bg-surface-2) 50%,
      var(--bg-surface) 75%
    );
    background-size: 200% 100%;
    animation: shimmer-sweep 1.4s infinite;
  }
}

@keyframes shimmer-sweep {
  0%   { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

.moment-search-dialog__item {
  padding: 10px 16px;
  cursor: pointer;
  border-radius: 6px;
  margin: 2px 8px;

  &:hover,
  &:focus-visible {
    background: color-mix(in oklab, var(--text) 6%, transparent);
    outline: none;
  }
}

.moment-search-dialog__item-title {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
  line-height: 1.4;
}

.moment-search-dialog__item-text {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.4;
  margin-top: 2px;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.moment-search-dialog__empty {
  padding: 20px 16px;
  font-size: 13px;
  color: var(--text-secondary);
  text-align: center;
}
</style>

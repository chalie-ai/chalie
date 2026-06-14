<script setup lang="ts">
import { ref, computed } from 'vue';
import { emit } from '../../composables/useEventBus';

export interface ListData {
  skill?: 'list';
  id: string | number;
  name: string;
  items: Array<{ content: string; checked: boolean }>;
}

export interface ListPayload extends ListData {
  /**
   * Some payloads nest the list under `list`. Legacy list.js:34 unwrapped
   * `payload.list || payload`; we replicate that so both shapes render.
   */
  list?: ListData;
}

interface ListItem {
  content: string;
  checked: boolean;
  /** In-flight guard — blocks re-toggling a row while its action is pending. */
  busy?: boolean;
}

const props = defineProps<{ payload: ListPayload; synthesis?: string }>();

// Unwrap the nested-or-flat shape (legacy list.js:34: `payload.list || payload`).
const listData = computed<ListData>(() => props.payload.list ?? props.payload);

// Local reactive copy for optimistic UI — toggles update the DOM immediately.
// `items ?? []` guards a malformed/historic payload (legacy list.js:34).
const items = ref<ListItem[]>((listData.value.items ?? []).map((i) => ({ ...i })));

const progressPercent = computed<number>(() => {
  const total = items.value.length;
  if (total === 0) return 0;
  const done = items.value.filter((i) => i.checked).length;
  return (done / total) * 100;
});

const doneCount = computed<number>(() => items.value.filter((i) => i.checked).length);

function onToggle(item: ListItem): void {
  // Block re-toggling while a previous flip is still in flight (legacy list.js:87).
  if (item.busy) return;
  const newState = !item.checked;

  // Optimistic flip — revert on backend error (legacy list.js:91-114).
  item.checked = newState;
  item.busy = true;

  const revert = (): void => {
    item.checked = !newState;
    item.busy = false;
  };

  // The session store's chalie:silent-action listener reads detail.payload and
  // posts it via ws.sendAction; onMessage/onError/onDone drive the optimistic UI.
  emit('chalie:silent-action', {
    payload: {
      skill: 'list',
      action: 'check',
      id: listData.value.id,
      items: [{ content: item.content, checked: newState }],
    },
    onMessage: () => {
      item.busy = false;
    },
    onError: revert,
    onDone: () => {
      item.busy = false;
    },
  });
}
</script>

<template>
  <div class="rich-card list-card">
    <!-- Header: title + progress count -->
    <div class="list-card__head">
      <h4 class="list-card__title">{{ listData.name }}</h4>
      <div class="list-card__progress">
        <b>{{ doneCount }}</b> / {{ items.length }} done
      </div>
    </div>

    <!-- Progress bar -->
    <div class="list-card__bar">
      <div class="list-card__bar-fill" :style="{ width: progressPercent + '%' }" />
    </div>

    <!-- Checklist items -->
    <div class="list-card__items">
      <div
        v-for="(item, index) in items"
        :key="index"
        class="list-card__item"
        :class="{ 'list-card__item--done': item.checked }"
        @click="onToggle(item)"
      >
        <span class="list-card__check" aria-hidden="true">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="3.5"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <polyline points="20 6 9 17 4 12" />
          </svg>
        </span>
        <div class="list-card__text">{{ item.content }}</div>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
.list-card__head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 10px;
}

.list-card__title {
  font-size: 1.05rem;
  font-weight: 500;
  letter-spacing: -0.005em;
  margin: 0;
  color: var(--text-primary);
}

.list-card__progress {
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.72rem;
  color: var(--text-tertiary);
  letter-spacing: 0.06em;
}

.list-card__progress b {
  color: var(--success);
  font-weight: 500;
}

.list-card__bar {
  height: 2px;
  background: var(--border);
  border-radius: 1px;
  overflow: hidden;
  margin-bottom: 6px;
}

.list-card__bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--violet), var(--violet-hover, #B07CFF));
  box-shadow: 0 0 6px color-mix(in oklab, var(--violet) 50%, transparent);
  border-radius: 1px;
  transition: width 300ms ease;
}

.list-card__items {
  display: flex;
  flex-direction: column;
}

.list-card__item {
  display: grid;
  grid-template-columns: 22px 1fr;
  gap: 12px;
  align-items: flex-start;
  padding: 10px 0;
  border-bottom: 1px solid color-mix(in oklab, var(--border) 60%, transparent);
  cursor: pointer;
  transition: background 160ms ease;

  &:last-child {
    border-bottom: none;
  }

  &:hover {
    background: color-mix(in oklab, var(--text) 1.5%, transparent);
  }
}

.list-card__check {
  width: 18px;
  height: 18px;
  border-radius: 6px;
  border: 1.5px solid var(--border-strong);
  background: transparent;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-top: 1px;
  flex-shrink: 0;
  transition: all 160ms ease;

  svg {
    width: 11px;
    height: 11px;
    opacity: 0;
    color: white;
  }
}

.list-card__item--done {
  .list-card__check {
    background: var(--violet);
    border-color: var(--violet);
    box-shadow: 0 0 8px color-mix(in oklab, var(--violet) 50%, transparent);

    svg {
      opacity: 1;
    }
  }

  .list-card__text {
    color: var(--text-tertiary);
    text-decoration: line-through;
    text-decoration-color: color-mix(in oklab, var(--text-tertiary) 60%, transparent);
  }
}

.list-card__text {
  font-size: 0.94rem;
  color: var(--text-primary);
  line-height: 1.4;
}
</style>

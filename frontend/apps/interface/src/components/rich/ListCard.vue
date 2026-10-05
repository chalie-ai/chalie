<script setup lang="ts">
import { computed, ref } from 'vue';
import { Check } from '@lucide/vue';
import { toggleItem } from '../../api/lists';

export interface ListData {
  skill?: 'list';
  id: string | number;
  name: string;
  items: Array<{ id: string | number; content: string; checked: boolean }>;
}

export interface ListPayload extends ListData {
  /** Some payloads nest the list under `list`; both shapes must render. */
  list?: ListData;
}

interface ListItem {
  id: string | number;
  content: string;
  checked: boolean;
  /** In-flight guard — blocks re-toggling a row while its action is pending. */
  busy?: boolean;
}

const props = defineProps<{ payload: ListPayload; synthesis?: string }>();

const listData = computed<ListData>(() => props.payload.list ?? props.payload);

// Local reactive copy for optimistic UI; `?? []` guards a malformed payload.
const items = ref<ListItem[]>((listData.value.items ?? []).map((i) => ({ ...i })));

const doneCount = computed<number>(() => items.value.filter((i) => i.checked).length);

const progressPercent = computed<number>(() =>
  items.value.length ? (doneCount.value / items.value.length) * 100 : 0,
);

async function onToggle(item: ListItem): Promise<void> {
  if (item.busy) return;
  const newState = !item.checked;

  // Optimistic flip — revert on backend error.
  item.checked = newState;
  item.busy = true;

  try {
    await toggleItem(listData.value.id, item.id, newState);
    item.busy = false;
  } catch {
    item.checked = !newState;
    item.busy = false;
  }
}
</script>

<template>
  <div class="rich-card list-card">
    <div class="list-card__head">
      <h4 class="list-card__title">{{ listData.name }}</h4>
      <div class="list-card__progress">
        <b>{{ doneCount }}</b> / {{ items.length }} done
      </div>
    </div>

    <div class="list-card__bar">
      <div class="list-card__bar-fill" :style="{ width: progressPercent + '%' }" />
    </div>

    <div class="list-card__items">
      <div
        v-for="(item, index) in items"
        :key="index"
        class="list-card__item"
        :class="{ 'list-card__item--done': item.checked }"
        @click="onToggle(item)"
      >
        <span class="list-card__check" aria-hidden="true">
          <Check :stroke-width="3.5" />
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
  color: var(--text);
}

.list-card__progress {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--muted);
  letter-spacing: 0.06em;
}

.list-card__progress b {
  color: var(--allow-text);
  font-weight: 500;
}

.list-card__bar {
  height: 2px;
  background: var(--cell);
  overflow: hidden;
  margin-bottom: 6px;
}

.list-card__bar-fill {
  height: 100%;
  background: var(--pink);
  transition: width var(--dur-3) var(--ease-out);
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
  border-bottom: 1px solid var(--line);
  cursor: pointer;
  transition: background-color var(--dur-1) var(--ease-out);

  &:last-child {
    border-bottom: none;
  }

  &:hover {
    background: var(--surface-2);
  }
}

.list-card__check {
  width: 18px;
  height: 18px;
  border: 1.5px solid var(--control);
  background: transparent;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-top: 1px;
  flex-shrink: 0;
  transition:
    background-color var(--dur-1) var(--ease-out),
    border-color var(--dur-1) var(--ease-out);

  svg {
    width: 11px;
    height: 11px;
    opacity: 0;
    color: var(--on-pink);
  }
}

.list-card__item--done {
  .list-card__check {
    background: var(--pink);
    border-color: var(--pink);

    svg {
      opacity: 1;
    }
  }

  .list-card__text {
    color: var(--muted);
    text-decoration: line-through;
  }
}

.list-card__text {
  font-size: 0.94rem;
  color: var(--text);
  line-height: 1.4;
}
</style>

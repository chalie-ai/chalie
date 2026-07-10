<!-- D15 — spine surface wrapper: TurnView through settle0 plus the fork-pill
     row (moved out of ConversationFeed.vue, which no longer knows anything
     about turn/pill state). Mounted imperatively by turnDom.upsertTurnToSurfaces
     for every turn of type `type` — one instance per rendered turn, tracking
     its OWN working/done state off the 'turn-state-changed' DOM signal
     (utils/turnDom.ts) rather than any shared store. -->
<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { ConfigType } from '@chalie/shared';
import type { ConversationTurnBlock } from '../../api/conversation';
import { useSessionStore } from '../../stores/session';
import { isThreadActive } from '../../utils/threadActivity';
import TurnView from './TurnView.vue';

const props = withDefaults(
  defineProps<{ block: ConversationTurnBlock; type?: string }>(),
  { type: ConfigType.USER },
);

const session = useSessionStore();

// A forked thread carries reply rows past settle0 (`thread_message` flag).
const isForked = computed(() => props.block.messages.some((m) => m.thread_message));

// Initialised from the block's own server-derived `working` field; every
// SUBSEQUENT transition comes only from the DOM signal turnDom.setTurnWorking
// / setTurnDone dispatch — never re-read from a new `block` prop, since a
// content-only re-render (no state change) must not clobber it.
const working = ref(props.block.working);
const done = ref(false);

function onStateChanged(e: Event): void {
  const detail = (e as CustomEvent<{ turnId: number; type: string; working?: boolean; done?: boolean }>).detail;
  if (detail.turnId !== props.block.turn_id || detail.type !== props.type) return;
  if ('working' in detail) working.value = !!detail.working;
  if ('done' in detail) done.value = !!detail.done;
}

onMounted(() => document.addEventListener('turn-state-changed', onStateChanged));
onBeforeUnmount(() => document.removeEventListener('turn-state-changed', onStateChanged));

/** Pill status drives its border and dot colour. */
const pillStatus = computed<'working' | 'done' | 'thread' | 'idle'>(() => {
  if (working.value) return 'working';
  if (done.value) return 'done';
  if (isThreadActive(props.block.last_activity_at)) return 'thread';
  return 'idle';
});

function onPillClick(): void {
  session.openThreadPanel(props.block.turn_id, props.type);
}

function onReply(): void {
  session.openThreadPanel(props.block.turn_id, props.type);
}
</script>

<template>
  <TurnView :block="block" :type="type" @reply="onReply" />

  <!-- Thread opener: a forked turn gets a Weave pill. -->
  <div v-if="isForked" class="feed-pill-row">
    <button
      class="thread-pill"
      :class="`thread-pill--${pillStatus}`"
      type="button"
      @click="onPillClick"
    >
      <span class="thread-pill__dot" aria-hidden="true" />
      <span class="thread-pill__summary">{{ block.gist || block.preview || 'Conversation' }}</span>
      <span class="thread-pill__chevron" aria-hidden="true">›</span>
    </button>
  </div>
</template>

<style scoped lang="scss">
/* Thread pill — a collapsed fork off the conversation. */
.feed-pill-row {
  width: 100%;
  max-width: var(--dock-width);
  margin: 14px auto 0;
  padding-left: calc(var(--avatar-size) + 18px);
}

.thread-pill {
  display: inline-flex;
  align-items: center;
  gap: 9px;
  max-width: 100%;
  padding: 6px 11px;
  border-radius: 11px;
  background: var(--bg-surface-2);
  border: 1px solid var(--border-strong);
  cursor: pointer;
  transition:
    background var(--duration-fast) ease,
    border-color var(--duration-fast) ease;
}

.thread-pill:hover {
  background: color-mix(in oklab, var(--violet) 7%, var(--bg-surface-2));
}

.thread-pill:disabled {
  cursor: default;
  opacity: 0.6;
}

.thread-pill--working {
  border-color: color-mix(in oklab, var(--status-main) 45%, transparent);
}
.thread-pill--done {
  border-color: color-mix(in oklab, var(--cyan) 45%, transparent);
}
.thread-pill--thread {
  border-color: color-mix(in oklab, var(--violet) 35%, transparent);
}

.thread-pill__summary {
  flex: 1 1 auto;
  min-width: 0;
  font-size: 13px;
  color: var(--text-tertiary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.thread-pill__chevron {
  flex-shrink: 0;
  font-size: 15px;
  line-height: 1;
  color: var(--text-primary);
  opacity: 0.35;
}

.thread-pill__dot {
  flex-shrink: 0;
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.thread-pill--working .thread-pill__dot {
  background: var(--status-main);
  box-shadow: 0 0 8px color-mix(in oklab, var(--status-main) 45%, transparent);
  animation: pulseV 1.4s ease-in-out infinite;
}

.thread-pill--done .thread-pill__dot {
  background: var(--cyan);
  box-shadow: 0 0 8px color-mix(in oklab, var(--cyan) 45%, transparent);
}

.thread-pill--thread .thread-pill__dot {
  background: var(--violet);
  box-shadow: 0 0 8px color-mix(in oklab, var(--violet) 40%, transparent);
}

.thread-pill--idle .thread-pill__dot {
  background: transparent;
  border: 1.5px solid color-mix(in oklab, var(--text-primary) 30%, transparent);
}
</style>

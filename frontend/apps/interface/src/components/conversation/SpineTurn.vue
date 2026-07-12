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

/** Pill status drives its dot colour. */
const pillStatus = computed<'working' | 'done' | 'thread' | 'idle'>(() => {
  if (working.value) return 'working';
  if (done.value) return 'done';
  if (isThreadActive(props.block.last_activity_at)) return 'thread';
  return 'idle';
});

// A forked turn's thread opener now rides INLINE on its settle0 footer meta line
// (rendered by BubbleFooter), not as a separate pill row. We hand TurnView the
// status + gist label; null for a non-forked turn leaves the footer pill-free.
const threadPill = computed<{ status: 'working' | 'done' | 'thread' | 'idle'; label: string } | null>(
  () =>
    isForked.value
      ? { status: pillStatus.value, label: props.block.gist || props.block.preview || 'Conversation' }
      : null,
);

function onOpenThread(): void {
  session.openThreadPanel(props.block.turn_id, props.type);
}
</script>

<template>
  <TurnView
    :block="block"
    :type="type"
    :thread-pill="threadPill"
    @reply="onOpenThread"
    @open-thread="onOpenThread"
  />
</template>

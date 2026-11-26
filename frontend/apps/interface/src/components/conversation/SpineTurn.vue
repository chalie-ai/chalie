<!-- D15 — spine surface wrapper: TurnView through settle0 plus the fork-pill
     row (moved out of ConversationFeed.vue, which no longer knows anything
     about turn/pill state). Mounted imperatively by turnDom.upsertTurnToSurfaces
     for every turn of type `type` — one instance per rendered turn. It listens
     for nothing: whether the turn is working or done lives on the turn's own
     root as `data-working` / `data-done`, which turnDom stamps on mount and
     toggles in place, and the thread pill's dot reads them from the
     stylesheet. -->
<script setup lang="ts">
import { computed } from 'vue';
import { ConfigType } from '@chalie/shared';
import type { ConversationTurnBlock } from '../../api/conversation';
import { useSessionStore } from '../../stores/session';
import { isThreadActive } from '../../utils/threadActivity';
import { quotePreview } from '../../utils/quoteBlocks';
import TurnView from './TurnView.vue';

const props = withDefaults(
  defineProps<{ block: ConversationTurnBlock; type?: string }>(),
  { type: ConfigType.USER },
);

const session = useSessionStore();

// A forked thread carries reply rows past settle0 (`thread_message` flag).
const isForked = computed(() => props.block.messages.some((m) => m.thread_message));

// A forked turn's thread opener rides INLINE on its closing footer meta line
// (rendered by BubbleFooter), not as a separate pill row. We hand TurnView the
// status the block itself says + the gist label; null for a non-forked turn
// leaves the footer pill-free. Working and done override the dot from the
// turn root's attributes.
const threadPill = computed<{ status: 'thread' | 'idle'; label: string } | null>(() =>
  isForked.value
    ? {
        status: isThreadActive(props.block.last_activity_at) ? 'thread' : 'idle',
        label: props.block.gist || quotePreview(props.block.preview) || 'Conversation',
      }
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

<!-- Renders the conversation spine: EVERY turn inline as Weave avatar rows (its
     rows through settle0), plus thread opener pills for forked turns. -->
<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted, onBeforeUnmount } from 'vue';
import { useConversationFeed } from '../../composables/useConversationFeed';
import type { ConversationTurnBlock } from '../../api/conversation';
import { useSessionStore } from '../../stores/session';
import { useAutoscroll } from '../../composables/useAutoscroll';
import TurnView from './TurnView.vue';

const feed = useConversationFeed();
const session = useSessionStore();

const feedRef = ref<HTMLElement | null>(null);
const { scrollToBottom, forceScrollToBottom } = useAutoscroll(feedRef);

type FeedEntry = {
  block: ConversationTurnBlock;
  isForked: boolean;
};

const feedEntries = computed<FeedEntry[]>(() =>
  feed.sortedBlocks.value.map((block) => ({
    block,
    isForked: feed.isForkedThread(block.turn_id),
  })),
);

/** Pill status drives its border and dot colour. */
function pillStatus(block: ConversationTurnBlock): 'working' | 'done' | 'thread' | 'idle' {
  const phase = feed.threadPhase(block.turn_id);
  if (phase) return phase;
  if (feed.isThreadActive(block.last_activity_at)) return 'thread';
  return 'idle';
}

function onPillClick(turnId: number): void {
  void session.openThreadPanel(turnId);
}

function onReply(turnId: number): void {
  void session.openThreadPanel(turnId);
}

// History pagination: on scroll within 150px of the top (and not already
// loading/exhausted), anchor-preserve then paginate.
let _paginating = false;

async function _onScrollPaginate(): Promise<void> {
  if (_paginating) return;
  if (session.historyLoading || !feed.hasMore) return;

  const scrollable = document.documentElement.scrollHeight > window.innerHeight + 100;
  if (!scrollable) return;
  if (window.scrollY >= 150) return;

  _paginating = true;
  try {
    const prevHeight = document.body.scrollHeight;
    const prevScrollY = window.scrollY;
    await session.loadRecentConversation();
    await nextTick();
    const added = document.body.scrollHeight - prevHeight;
    if (added > 0) {
      window.scrollTo({ top: prevScrollY + added, behavior: 'instant' });
    }
  } finally {
    _paginating = false;
  }
}

// Deep watch on sortedBlocks: narration/pill growth inside an in-flight turn
// leaves the array length unchanged, so a shallow watch would stop following
// the trail mid-cycle. flush:'post' lets scrollToBottom measure settled height.
watch(() => feed.sortedBlocks.value, scrollToBottom, { deep: true, flush: 'post' });

onMounted(async () => {
  document.addEventListener('session:turn-done', forceScrollToBottom);
  document.addEventListener('session:history-initial-loaded', forceScrollToBottom);

  await session.loadRecentConversation();

  window.addEventListener('scroll', _onScrollPaginate, { passive: true });
});

onBeforeUnmount(() => {
  window.removeEventListener('scroll', _onScrollPaginate);
  document.removeEventListener('session:turn-done', forceScrollToBottom);
  document.removeEventListener('session:history-initial-loaded', forceScrollToBottom);
});
</script>

<template>
  <main id="conversationFeed" ref="feedRef" class="conversation-spine">
    <div v-if="session.historyLoading" class="history-loader">
      <output class="history-loader__spinner" aria-label="Loading history" />
    </div>

    <div v-if="!feed.hasMore" class="history-end-pill">
      <span class="history-end-pill__label">End of thread history</span>
    </div>

    <template v-for="entry in feedEntries" :key="`i-${entry.block.turn_id}`">
      <!-- The turn's rows through settle0 — always inline, never collapsed. -->
      <TurnView :block="entry.block" @reply="onReply" />

      <!-- Thread opener: a forked turn gets a Weave pill. -->
      <div v-if="entry.isForked" class="feed-pill-row">
        <button
          class="thread-pill"
          :class="`thread-pill--${pillStatus(entry.block)}`"
          type="button"
          @click="onPillClick(entry.block.turn_id)"
        >
          <span class="thread-pill__dot" aria-hidden="true" />
          <span class="thread-pill__summary">{{ entry.block.gist || entry.block.preview || 'Conversation' }}</span>
          <span class="thread-pill__chevron" aria-hidden="true">›</span>
        </button>
      </div>
    </template>
  </main>
</template>

<style scoped lang="scss">
.history-loader {
  display: flex;
  justify-content: center;
  padding: 12px 0;
}

.history-loader__spinner {
  width: 18px;
  height: 18px;
  border: 2px solid color-mix(in oklab, var(--violet) 20%, transparent);
  border-top-color: var(--violet);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

.history-end-pill {
  display: flex;
  justify-content: center;
  padding: 16px 0 8px;
}

.history-end-pill__label {
  font-size: 11px;
  color: var(--text-muted);
  background: color-mix(in oklab, var(--text) 3%, transparent);
  border: 1px solid color-mix(in oklab, var(--text) 7%, transparent);
  border-radius: 20px;
  padding: 4px 14px;
  letter-spacing: 0.04em;
}

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
  transition: background var(--duration-fast) ease, border-color var(--duration-fast) ease;
}

.thread-pill:hover {
  background: color-mix(in oklab, var(--violet) 7%, var(--bg-surface-2));
}

.thread-pill:disabled {
  cursor: default;
  opacity: 0.6;
}

.thread-pill--working { border-color: color-mix(in oklab, var(--status-main) 45%, transparent); }
.thread-pill--done { border-color: color-mix(in oklab, var(--cyan) 45%, transparent); }
.thread-pill--thread { border-color: color-mix(in oklab, var(--violet) 35%, transparent); }

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

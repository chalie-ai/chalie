<!-- Conversation spine shell: history loader, end-of-history pill, scroll
     pagination, and autoscroll. Rendering itself is delegated entirely to the
     DOM contract — this component registers the spine as a surface (D14) and
     upserts fetched turns through it; it no longer holds turn data itself. -->
<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue';
import { ConfigType } from '@chalie/shared';
import { conversation as convoApi } from '../../api/conversation';
import { useSessionStore } from '../../stores/session';
import { useAutoscroll } from '../../composables/useAutoscroll';
import { registerSurface, unregisterSurface, upsertTurnToSurfaces, SPINE_SURFACE_ID } from '../../utils/turnDom';
import SpineTurn from './SpineTurn.vue';

const PAGE_SIZE = 20;

const session = useSessionStore();

const feedRef = ref<HTMLElement | null>(null);
const turnsRef = ref<HTMLElement | null>(null);
const { scrollToBottom, forceScrollToBottom } = useAutoscroll(feedRef);

// D17 — the pagination cursor (offset/hasMore) is UI-local, owned here.
const offset = ref(0);
const hasMoreRef = ref(true);

async function _fetchPage(pageOffset: number): Promise<number> {
  const { threads, has_more } = await convoApi.threads(PAGE_SIZE, pageOffset, undefined, ConfigType.USER);
  hasMoreRef.value = has_more;
  if (!threads.length) return 0;

  const ids = threads.map((t) => t.turn_id).filter((id): id is number => id != null);
  if (!ids.length) return threads.length;

  const { blocks } = await convoApi.batch(ids, ConfigType.USER);
  for (const block of blocks) {
    upsertTurnToSurfaces(block, ConfigType.USER);
  }
  return threads.length;
}

/** Registered with the session store (D17) — the store keeps `historyLoading`
 *  + the AuthError/initial-load event semantics, this owns the cursor. */
async function _loadRecent(): Promise<{ isInitialLoad: boolean; loadedAny: boolean }> {
  const isInitialLoad = offset.value === 0;
  offset.value = 0;
  hasMoreRef.value = true;
  const loaded = await _fetchPage(0);
  offset.value = loaded;
  return { isInitialLoad, loadedAny: loaded > 0 };
}

async function _loadMore(): Promise<void> {
  if (!hasMoreRef.value) return;
  const loaded = await _fetchPage(offset.value);
  offset.value += loaded;
}

// History pagination: on scroll within 150px of the top (and not already
// loading/exhausted), anchor-preserve then paginate.
let _paginating = false;

async function _onScrollPaginate(): Promise<void> {
  if (_paginating) return;
  if (session.historyLoading || !hasMoreRef.value) return;

  const scrollable = document.documentElement.scrollHeight > window.innerHeight + 100;
  if (!scrollable) return;
  if (window.scrollY >= 150) return;

  _paginating = true;
  try {
    const prevHeight = document.body.scrollHeight;
    const prevScrollY = window.scrollY;
    await _loadMore();
    await nextTick();
    const added = document.body.scrollHeight - prevHeight;
    if (added > 0) {
      window.scrollTo({ top: prevScrollY + added, behavior: 'instant' });
    }
  } finally {
    _paginating = false;
  }
}

// Autoscroll: turnDom dispatches 'turn-upserted' after every DOM render
// (initial load, WS-driven refetch, live growth mid-turn) — replaces the old
// deep watch on the buffer's sortedBlocks.
function onTurnUpserted(): void {
  scrollToBottom();
}

onMounted(async () => {
  if (turnsRef.value) {
    registerSurface({
      id: SPINE_SURFACE_ID,
      type: ConfigType.USER,
      container: turnsRef.value,
      component: SpineTurn,
    });
  }

  document.addEventListener('session:history-initial-loaded', forceScrollToBottom);
  document.addEventListener('turn-upserted', onTurnUpserted);

  session.registerHistoryLoader(_loadRecent);
  await session.loadRecentConversation();

  window.addEventListener('scroll', _onScrollPaginate, { passive: true });
});

onBeforeUnmount(() => {
  window.removeEventListener('scroll', _onScrollPaginate);
  document.removeEventListener('session:history-initial-loaded', forceScrollToBottom);
  document.removeEventListener('turn-upserted', onTurnUpserted);
  unregisterSurface(SPINE_SURFACE_ID);
});
</script>

<template>
  <main id="conversationFeed" ref="feedRef" class="conversation-spine">
    <div v-if="session.historyLoading" class="history-loader">
      <output class="history-loader__spinner" aria-label="Loading history" />
    </div>

    <div v-if="!hasMoreRef" class="history-end-pill">
      <span class="history-end-pill__label">End of thread history</span>
    </div>

    <div ref="turnsRef" class="conversation-spine__turns" />
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
</style>

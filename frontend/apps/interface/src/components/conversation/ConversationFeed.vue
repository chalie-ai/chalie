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
import { registerSurface, unregisterSurface, upsertTurnToSurfaces, evictOldestAbove, nextTicket, SPINE_SURFACE_ID } from '../../utils/turnDom';
import { syncDaymarks } from '../../utils/daymarks';
import { preloadRichCards } from '../rich/richRegistry';
import SpineTurn from './SpineTurn.vue';

const PAGE_SIZE = 10;
// Retained-turn window: while pinned near the bottom, older off-screen turns
// past this many are evicted and lazy-reloaded on scroll-up (bounds the DOM).
const KEEP_RENDERED = 30;

const session = useSessionStore();

const feedRef = ref<HTMLElement | null>(null);
const turnsRef = ref<HTMLElement | null>(null);
const { scrollToBottom, forceScrollToBottom, userScrolledUp } = useAutoscroll(feedRef);

// D17 — the pagination cursor (offset/hasMore) is UI-local, owned here.
const offset = ref(0);
const hasMoreRef = ref(true);

async function _fetchPage(pageOffset: number): Promise<number> {
  const { threads, has_more } = await convoApi.threads(PAGE_SIZE, pageOffset, undefined, ConfigType.USER);
  hasMoreRef.value = has_more;
  if (!threads.length) return 0;

  const ids = threads.map((t) => t.turn_id).filter((id): id is number => id != null);
  if (!ids.length) return threads.length;

  // Ticketed: a running turn in this page can settle while the read is out,
  // and its settle must not be repainted by this older snapshot (see turnDom's
  // `UpsertOptions.ticket`).
  const ticket = nextTicket();
  const { blocks } = await convoApi.batch(ids, ConfigType.USER);
  // The scroll that follows a page load measures final heights only if the
  // page's rich cards are already in the DOM, not still fetching their chunks.
  await preloadRichCards(blocks.flatMap((b) => b.messages.flatMap((m) => m.segments ?? [])));
  for (const block of blocks) {
    upsertTurnToSurfaces(block, ConfigType.USER, { ticket });
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

// The top of the feed is in view: within 150px of it, or the page is too
// short to scroll at all (no scroll event will ever come to reach it).
function _topInView(): boolean {
  const scrollable = document.documentElement.scrollHeight > window.innerHeight + 100;
  return !scrollable || window.scrollY < 150;
}

// History pagination: while the top is in view and history remains, load the
// next older page and keep the reader's place, so a page too short to push the
// top out of view is followed by the next one.
let _paginating = false;

async function _onScrollPaginate(): Promise<void> {
  if (_paginating) return;
  _paginating = true;
  try {
    while (hasMoreRef.value && !session.historyLoading && _topInView()) {
      const prevHeight = document.body.scrollHeight;
      const prevScrollY = window.scrollY;
      const loaded = await _fetchPage(offset.value);
      if (!loaded) break;
      offset.value += loaded;
      await nextTick();
      const added = document.body.scrollHeight - prevHeight;
      if (added > 0) {
        window.scrollTo({ top: prevScrollY + added, behavior: 'instant' });
      }
    }
  } finally {
    _paginating = false;
  }
}

// Autoscroll: turnDom dispatches 'turn-upserted' after every DOM render
// (initial load, WS-driven refetch, live growth mid-turn) — replaces the old
// deep watch on the buffer's sortedBlocks. Reconcile the day dividers first so
// their height is already in the layout when autoscroll measures the bottom.
function onTurnUpserted(): void {
  const c = turnsRef.value;
  // Bound the DOM: while pinned near the bottom, drop the oldest off-screen
  // turns past the retained window and resync the pagination cursor so scroll-up
  // re-fetches exactly what was evicted. Gated on !userScrolledUp so reading
  // history is never trimmed underneath the user.
  if (c && !userScrolledUp.value) {
    const before = c.childElementCount;
    evictOldestAbove(c, KEEP_RENDERED);
    if (c.childElementCount < before) {
      offset.value = c.querySelectorAll(':scope > [data-version]').length;
      hasMoreRef.value = true;
    }
  }
  if (c) syncDaymarks(c);
  scrollToBottom();
}

// The first load's jump to the bottom, settled once the jump has been made.
let landed: Promise<void> = Promise.resolve();
function onInitialLoaded(): void {
  landed = forceScrollToBottom();
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

  document.addEventListener('session:history-initial-loaded', onInitialLoaded);
  document.addEventListener('turn-upserted', onTurnUpserted);

  session.registerHistoryLoader(_loadRecent);
  await session.loadRecentConversation();

  // Pagination starts only once the first load has landed at the bottom: the
  // autoscroll that follows each rendered turn sets off from the top, and its
  // first step would otherwise read as the reader reaching the top.
  await landed;
  window.addEventListener('scroll', _onScrollPaginate, { passive: true });
  void _onScrollPaginate();
});

onBeforeUnmount(() => {
  window.removeEventListener('scroll', _onScrollPaginate);
  document.removeEventListener('session:history-initial-loaded', onInitialLoaded);
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
      <span class="tag">End of thread history</span>
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
  border: 2px solid var(--line);
  border-top-color: var(--pink-text);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

.history-end-pill {
  display: flex;
  justify-content: center;
  padding: 16px 0 8px;
}
</style>

<!-- Renders the conversation spine: EVERY turn inline as Weave avatar rows (its
     rows through settle0), plus the live in-flight turn. A turn with replies past
     settle0 also gets a Thread opener beneath it; the opener and the reply action
     open the thread in the slide-over panel. Threads never collapse the spine. -->
<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted, onBeforeUnmount } from 'vue';
import { useConversationStore } from '../../stores/conversation';
import type { ConversationForm, ThreadListItem } from '../../stores/conversation';
import { useSessionStore } from '../../stores/session';
import { useAutoscroll } from '../../composables/useAutoscroll';
import TurnView from './TurnView.vue';

const conversationStore = useConversationStore();
const session = useSessionStore();

const feedRef = ref<HTMLElement | null>(null);
const { scrollToBottom, forceScrollToBottom } = useAutoscroll(feedRef);

// Live turn: forms not belonging to any known thread (in-flight).
const liveTurnForms = computed<ConversationForm[]>(() => {
  const known = new Set(conversationStore.threads.map((t) => t.turn_id).filter((t): t is number => t != null));
  return conversationStore.forms.filter((f) => f.turnId == null || !known.has(f.turnId));
});

const liveTurn = computed(() => {
  if (!liveTurnForms.value.length) return null;
  const forms = liveTurnForms.value;
  return { id: forms[0].id, forms };
});

// Every turn renders inline. Its spine forms are the rows through settle0 (the
// reply continuation — thread_message forms — is dropped); a turn that HAS any
// such reply also carries a Thread opener (`thread` non-null). The live in-flight
// turn is inline with no opener.
type FeedEntry = {
  id: number;
  forms: ConversationForm[];
  working: boolean;
  thread: ThreadListItem | null;
};

const feedEntries = computed<FeedEntry[]>(() => {
  const entries: FeedEntry[] = [];
  for (const t of conversationStore.threads) {
    const all = conversationStore.forms.filter((f) => f.turnId === t.turn_id);
    const spine = all.filter((f) => !f.threadMessage);
    if (!spine.length) continue; // not yet hydrated — fills in on batch load
    // A bound turn's spinner is driven by its own `working` signal (isTurnWorking
    // inside TurnView); only the unbound live turn below needs the prop.
    entries.push({ id: spine[0].id, forms: spine, working: false, thread: all.length > spine.length ? t : null });
  }
  // The live turn has no turn_id yet, so its spinner can't come from isTurnWorking
  // — drive it from this surface's send guard until `working` binds the thread.
  if (liveTurn.value) {
    entries.push({ id: liveTurn.value.id, forms: liveTurn.value.forms, working: session.isSending, thread: null });
  }
  return entries;
});

/** Pill status drives its border, dot and badge. `new` once the backend tracks
 *  unread replies; `thread` while inside the 1-hour active window; else idle. */
function pillStatus(t: ThreadListItem): 'new' | 'thread' | 'idle' {
  if (t.unread) return 'new';
  if (conversationStore.isThreadActive(t.last_activity_at)) return 'thread';
  return 'idle';
}

function onPillClick(thread: ThreadListItem): void {
  if (thread.turn_id == null) return;
  void session.openThreadPanel(thread.turn_id);
}

function onReply(turnId: number): void {
  void session.openThreadPanel(turnId);
}

// History pagination: on scroll within 150px of the top (and not already
// loading/exhausted), anchor-preserve then paginate.
let _paginating = false;

async function _onScrollPaginate(): Promise<void> {
  if (_paginating) return;
  if (session.historyLoading || conversationStore.threadsExhausted) return;

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

// Deep watch (not count-only): narration/pill growth inside an in-flight ACT
// form leaves forms.length unchanged, so a shallow watch would stop following
// the trail mid-cycle. flush:'post' lets scrollToBottom measure settled height;
// it's the GUARDED smooth variant that self-skips when the user has scrolled up.
watch(() => conversationStore.forms, scrollToBottom, { deep: true, flush: 'post' });

onMounted(async () => {
  document.addEventListener('session:turn-done', forceScrollToBottom);
  document.addEventListener('session:history-initial-loaded', forceScrollToBottom);

  // Initial thread-list load — this is the ONLY trigger (App.vue does NOT call it).
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

    <div v-if="conversationStore.threadsExhausted" class="history-end-pill">
      <span class="history-end-pill__label">End of thread history</span>
    </div>

    <template v-for="entry in feedEntries" :key="`i-${entry.id}`">
      <!-- The turn's rows through settle0 — always inline, never collapsed. -->
      <TurnView :forms="entry.forms" :working="entry.working" @reply="onReply" />

      <!-- Thread opener: a turn with replies past settle0 gets a Weave pill that
           opens the full thread in the slide-over panel. -->
      <div v-if="entry.thread" class="feed-pill-row">
        <button
          class="thread-pill"
          :class="`thread-pill--${pillStatus(entry.thread)}`"
          type="button"
          @click="onPillClick(entry.thread)"
        >
          <span class="thread-pill__icon" aria-hidden="true">
            <svg
              width="13"
              height="13"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2.2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <circle cx="6" cy="6" r="3" />
              <circle cx="18" cy="18" r="3" />
              <path d="M6 9c0 6 6 3 6 9" />
            </svg>
          </span>
          <span class="thread-pill__label">Thread</span>
          <span class="thread-pill__summary">{{ entry.thread.gist || entry.thread.preview || 'Conversation' }}</span>
          <span class="thread-pill__dot" aria-hidden="true" />
          <span v-if="entry.thread.unread" class="thread-pill__badge">{{ entry.thread.unread }} new</span>
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
  animation: history-spin 0.7s linear infinite;
}

@keyframes history-spin {
  to { transform: rotate(360deg); }
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

/* Thread pill — a collapsed fork off the conversation. Indented to sit under the
   message column (past the avatar gutter), centred at the dock width. */
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
  padding: 6px 12px 6px 7px;
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

.thread-pill--new { border-color: color-mix(in oklab, var(--status-main) 40%, transparent); }
.thread-pill--thread { border-color: color-mix(in oklab, var(--violet) 35%, transparent); }

.thread-pill__icon {
  flex-shrink: 0;
  width: 22px;
  height: 22px;
  border-radius: 7px;
  display: grid;
  place-items: center;
  background: color-mix(in oklab, var(--violet) 18%, transparent);
  color: var(--violet-light);
}

.thread-pill__label {
  flex-shrink: 0;
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
}

.thread-pill__summary {
  min-width: 0;
  font-size: 13px;
  color: var(--text-tertiary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.thread-pill__dot {
  flex-shrink: 0;
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

/* Inline (in-pill) the unread dot is the muted rose — NOT the bright magenta;
   that glow is reserved for the activity rail / search dots. */
.thread-pill--new .thread-pill__dot {
  background: var(--status-main);
}

.thread-pill--thread .thread-pill__dot {
  background: var(--violet);
  box-shadow: 0 0 8px color-mix(in oklab, var(--violet) 40%, transparent);
}

.thread-pill--idle .thread-pill__dot {
  background: transparent;
  border: 1.5px solid color-mix(in oklab, var(--text-primary) 30%, transparent);
}

.thread-pill__badge {
  flex-shrink: 0;
  font-size: 10px;
  font-weight: 600;
  padding: 1px 7px;
  border-radius: 9px;
  background: color-mix(in oklab, var(--status-main) 16%, transparent);
  color: color-mix(in oklab, var(--status-main) 70%, var(--text-primary));
}
</style>

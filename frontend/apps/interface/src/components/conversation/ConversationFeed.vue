<!-- Renders the thread feed: single-exchange turns inline as Weave avatar rows,
     every other thread as a collapsed Thread pill, plus the live in-flight turn.
     Pills and the reply action open the thread in the slide-over panel. -->
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

// A thread renders INLINE (Weave avatar rows) only when it is a single exchange
// — one user message — and is not currently held open in the panel. Everything
// else collapses to a Thread pill. The live in-flight turn is always inline.
type FeedEntry =
  | { kind: 'pill'; thread: ThreadListItem }
  | { kind: 'inline'; id: number; forms: ConversationForm[]; working: boolean };

const feedEntries = computed<FeedEntry[]>(() => {
  const entries: FeedEntry[] = [];
  for (const t of conversationStore.threads) {
    const forms = t.expanded ? conversationStore.forms.filter((f) => f.turnId === t.turn_id) : [];
    const singleExchange = forms.filter((f) => f.kind === 'user').length === 1;
    if (forms.length && singleExchange && t.turn_id !== session.panelThreadId) {
      // A bound turn's spinner is driven by its own `working` signal (isTurnWorking
      // inside TurnView); only the unbound live turn below needs the prop.
      entries.push({ kind: 'inline', id: forms[0].id, forms, working: false });
    } else {
      entries.push({ kind: 'pill', thread: t });
    }
  }
  // The live turn has no turn_id yet, so its spinner can't come from isTurnWorking
  // — drive it from this surface's send guard until `working` binds the thread.
  if (liveTurn.value) {
    entries.push({ kind: 'inline', id: liveTurn.value.id, forms: liveTurn.value.forms, working: session.isSending });
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
  if (thread.loading || thread.turn_id == null) return;
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

    <template v-for="entry in feedEntries" :key="entry.kind === 'pill' ? `p-${entry.thread.turn_id}` : `i-${entry.id}`">
      <!-- Collapsed thread → Weave pill: fork glyph + label + gist + status. -->
      <div v-if="entry.kind === 'pill'" class="feed-pill-row">
        <button
          class="thread-pill"
          :class="`thread-pill--${pillStatus(entry.thread)}`"
          type="button"
          :disabled="entry.thread.loading"
          @click="onPillClick(entry.thread)"
        >
          <span class="thread-pill__icon" aria-hidden="true">
            <svg
              width="13"
              height="13"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <circle cx="6" cy="5" r="2.5" />
              <circle cx="18" cy="19" r="2.5" />
              <path d="M6 7.5v5a4 4 0 0 0 4 4h4.5" />
            </svg>
          </span>
          <span class="thread-pill__label">Thread</span>
          <span class="thread-pill__summary">{{ entry.thread.gist || entry.thread.preview || 'Conversation' }}</span>
          <span class="thread-pill__dot" aria-hidden="true" />
          <span v-if="entry.thread.unread" class="thread-pill__badge">{{ entry.thread.unread }} new</span>
        </button>
      </div>

      <!-- Single-exchange or live turn → inline avatar rows. -->
      <TurnView v-else :forms="entry.forms" :working="entry.working" @reply="onReply" />
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
  color: var(--violet);
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

.thread-pill--new .thread-pill__dot {
  background: var(--status-new);
  box-shadow: 0 0 10px color-mix(in oklab, var(--status-new) 60%, transparent);
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

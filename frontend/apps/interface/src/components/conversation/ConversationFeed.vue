<!-- Renders the thread-list feed: collapsed gist rows + expanded threads,
     plus the live in-flight turn. Drives thread-list pagination. -->
<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted, onBeforeUnmount } from 'vue';
import { ChevronRight, ChevronDown } from '@lucide/vue';
import { useConversationStore } from '../../stores/conversation';
import type { ConversationForm, UserForm, ChalieForm, ActForm, ThreadListItem } from '../../stores/conversation';
import { useSessionStore } from '../../stores/session';
import { useAutoscroll } from '../../composables/useAutoscroll';
import UserBubble from './UserBubble.vue';
import ChalieBubble from './ChalieBubble.vue';
import ActCycle from './ActCycle.vue';
import ActCycleGroup from './ActCycleGroup.vue';
import InputDock from '../layout/InputDock.vue';

const conversationStore = useConversationStore();
const session = useSessionStore();

// Consecutive superseded ACT cycles fold into one group; everything else
// (including a live, non-collapsed act) renders on its own.
type RenderRow =
  | { type: 'single'; id: number; form: ConversationForm }
  | { type: 'act-group'; id: number; forms: ActForm[] };

function groupRows(forms: ConversationForm[]): RenderRow[] {
  const rows: RenderRow[] = [];
  for (const form of forms) {
    const last = rows[rows.length - 1];
    if (form.kind === 'act' && form.collapsed) {
      if (last?.type === 'act-group') last.forms.push(form);
      else rows.push({ type: 'act-group', id: form.id, forms: [form] });
    } else {
      rows.push({ type: 'single', id: form.id, form });
    }
  }
  return rows;
}

/** Extract the turn_id shared by all forms in a turn (null for legacy rows). */
function threadTurnId(turn: { forms: ConversationForm[] }): number | null {
  for (const f of turn.forms) {
    if (f.turnId != null) return f.turnId;
  }
  return null;
}

/** True while this thread has a live (unresolved) ACT group — i.e. Chalie is
 *  mid-loop on it. Drives the header spinner + pulsing title. */
function isLooping(forms: ConversationForm[]): boolean {
  return forms.some((f) => f.kind === 'act' && !f.collapsed);
}

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
  const id = forms[0].id;
  return { id, forms };
});

// The ordered feed: interleave collapsed thread rows and expanded thread
// render groups following the thread list order.
type FeedEntry =
  | { kind: 'collapsed'; thread: ThreadListItem }
  | { kind: 'expanded'; turn: { id: number; forms: ConversationForm[] }; thread: ThreadListItem | null };

const feedEntries = computed<FeedEntry[]>(() => {
  const entries: FeedEntry[] = [];
  for (const t of conversationStore.threads) {
    if (t.expanded) {
      // Find the forms for this thread.
      const forms = conversationStore.forms.filter((f) => f.turnId === t.turn_id);
      if (forms.length) entries.push({ kind: 'expanded', turn: { id: forms[0].id, forms }, thread: t });
      else entries.push({ kind: 'collapsed', thread: t });
    } else {
      entries.push({ kind: 'collapsed', thread: t });
    }
  }
  // Live turn (in-flight) renders at the end — no thread row yet.
  if (liveTurn.value) entries.push({ kind: 'expanded', turn: liveTurn.value, thread: null });
  return entries;
});

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

function onThreadClick(thread: ThreadListItem): void {
  if (thread.loading) return;
  void session.expandThread(thread.turn_id!);
}

function onCollapse(turnId: number): void {
  session.collapseThread(turnId);
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

    <!-- Thread feed: collapsed gist rows + expanded threads, in list order. -->
    <template v-for="entry in feedEntries" :key="entry.kind === 'collapsed' ? `t-${entry.thread.turn_id}` : `e-${entry.turn.id}`">
      <!-- Collapsed thread: one-line gist row. Active threads expand on click;
           inactive (>1hr) threads show the gist only. -->
      <div
        v-if="entry.kind === 'collapsed'"
        class="thread-row"
        :class="{ 'thread-row--active': conversationStore.isThreadActive(entry.thread.last_activity_at) }"
      >
        <button
          class="thread-row__expand"
          type="button"
          :aria-label="'Expand thread'"
          :disabled="entry.thread.loading"
          @click="onThreadClick(entry.thread)"
        >
          <ChevronRight
            class="thread-row__caret"
            :class="{ 'thread-row__caret--spin': entry.thread.loading }"
            :size="14"
            aria-hidden="true"
          />
        </button>
        <div class="thread-row__body" @click="onThreadClick(entry.thread)">
          <span v-if="entry.thread.gist" class="thread-row__gist">{{ entry.thread.gist }}</span>
          <span v-else class="thread-row__gist thread-row__gist--placeholder">{{ entry.thread.preview || 'Conversation' }}</span>
        </div>
      </div>

      <!-- Expanded thread: a self-contained card. The header names what started
           the thread and carries the collapse control; the body holds the
           starter message and its replies; the shared composer (the dock with a
           turn_id) replies into the thread. -->
      <section
        v-else
        class="thread-card"
        :class="{ 'thread-card--active': entry.thread && conversationStore.isThreadActive(entry.thread.last_activity_at) }"
      >
        <header
          v-if="threadTurnId(entry.turn) != null"
          class="thread-card__head"
          role="button"
          tabindex="0"
          aria-label="Collapse thread"
          @click="onCollapse(threadTurnId(entry.turn)!)"
          @keydown.enter="onCollapse(threadTurnId(entry.turn)!)"
        >
          <output
            v-if="isLooping(entry.turn.forms)"
            class="thread-card__spinner"
            aria-label="Working"
          />
          <ChevronDown v-else class="thread-card__caret" :size="15" aria-hidden="true" />
          <span
            class="thread-card__starter"
            :class="{ 'thread-card__starter--loading': isLooping(entry.turn.forms) }"
          >{{ entry.thread?.gist || entry.thread?.preview || 'Thread' }}</span>
        </header>
        <div class="thread-card__body">
          <!-- Same .turn flex grouping the flat feed uses, so replies keep the
               main-chat spacing/alignment — the card is only a frame around it. -->
          <div class="turn">
            <template v-for="row in groupRows(entry.turn.forms)" :key="row.id">
              <ActCycleGroup
                v-if="row.type === 'act-group'"
                :forms="row.forms"
              />
              <template v-else>
                <UserBubble
                  v-if="row.form.kind === 'user'"
                  :form="(row.form as UserForm)"
                />
                <ChalieBubble
                  v-else-if="row.form.kind === 'chalie'"
                  :form="(row.form as ChalieForm)"
                />
                <ActCycle
                  v-else-if="row.form.kind === 'act'"
                  :form="(row.form as ActForm)"
                />
              </template>
            </template>
          </div>
        </div>
        <!-- Reply composer — the same dock, only difference is supplying the
             thread's turn_id. Hidden on legacy NULL-turn_id singletons. -->
        <InputDock
          v-if="threadTurnId(entry.turn) != null"
          :turn-id="threadTurnId(entry.turn)!"
        />
      </section>
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

/* Collapsed thread row — one-line gist with expand/reply affordances. */
.thread-row {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-sm) var(--space-md);
  margin: var(--space-xs) 0;
  border-radius: var(--radius-sm);
  background: var(--bg-surface);
  border: 1px solid var(--border);
  transition: border-color var(--duration-fast) ease, background var(--duration-fast) ease;
}

.thread-row:hover {
  border-color: color-mix(in oklab, var(--violet) 25%, transparent);
  background: color-mix(in oklab, var(--violet) 4%, transparent);
}

.thread-row--active {
  border-left: 2px solid var(--violet);
}

.thread-row__expand {
  flex-shrink: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: none;
  cursor: pointer;
  color: var(--text-tertiary);
  border-radius: var(--radius-sm);
  transition: color var(--duration-fast);
}

.thread-row__expand:hover {
  color: var(--accent-primary);
}

.thread-row__caret {
  transition: transform 180ms var(--ease-out);
}

.thread-row__caret--spin {
  animation: history-spin 0.7s linear infinite;
}

.thread-row__body {
  flex: 1;
  min-width: 0;
  cursor: pointer;
  display: flex;
  align-items: center;
}

.thread-row__gist {
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.5;
}

.thread-row__gist--placeholder {
  color: var(--text-tertiary);
  font-style: italic;
}

/* The faint tertiary reads as washed-out on the dark surface — lift the gist
   preview to the secondary tone so it stays legible. */
[data-theme='dark'] .thread-row__gist--placeholder {
  color: var(--text-secondary);
}

/* Expanded thread — a contained card grouping a starter message with its
   replies, visually distinct from the flat collapsed rows around it. */
.thread-card {
  margin: var(--space-md) 0;
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  background: var(--bg-surface);
  overflow: hidden;
}

.thread-card--active {
  border-left: 3px solid var(--violet);
}

/* The whole header is the expand/collapse affordance — click it to collapse. */
.thread-card__head {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-sm) var(--space-md);
  background: color-mix(in oklab, var(--violet) 5%, transparent);
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  transition: background var(--duration-fast);
}

.thread-card__head:hover {
  background: color-mix(in oklab, var(--violet) 9%, transparent);
}

.thread-card__caret {
  flex-shrink: 0;
  color: var(--text-tertiary);
}

/* Live-loop affordance: a small spinner replaces the caret while a step runs. */
.thread-card__spinner {
  flex-shrink: 0;
  width: 13px;
  height: 13px;
  border: 2px solid color-mix(in oklab, var(--violet) 25%, transparent);
  border-top-color: var(--violet);
  border-radius: 50%;
  animation: history-spin 0.7s linear infinite;
}

.thread-card__starter {
  flex: 1;
  min-width: 0;
  font-size: var(--font-size-sm);
  font-weight: 600;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Gentle opacity oscillation signals the thread is actively working. */
.thread-card__starter--loading {
  animation: thread-pulse 1.4s ease-in-out infinite;
}

@keyframes thread-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.55; }
}

/* Just a frame: the inner .turn carries the flex layout, gap and alignment. */
.thread-card__body {
  padding: var(--space-md);
}
</style>
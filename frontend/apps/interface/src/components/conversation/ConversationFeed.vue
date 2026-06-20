<!--
  ConversationFeed — Task A1.

  Renders the ordered list of conversation forms from the conversation store,
  drives history pagination via the session store, and manages autoscroll.

  Port sources:
    - renderer.js: forceScrollToBottom, _initScrollTracking (via useAutoscroll)
    - chat.js: loadRecentConversation, init() scroll handler
-->
<script setup lang="ts">
import { ref, watch, nextTick, onMounted, onBeforeUnmount } from 'vue';
import { useConversationStore } from '../../stores/conversation';
import type { ConversationForm, UserForm, ChalieForm, ActForm, ErrorForm } from '../../stores/conversation';
import { useSessionStore } from '../../stores/session';
import { useAutoscroll } from '../../composables/useAutoscroll';
import UserBubble from './UserBubble.vue';
import ChalieBubble from './ChalieBubble.vue';
import ActCycle from './ActCycle.vue';
import ActCycleGroup from './ActCycleGroup.vue';
import ErrorFormVue from './ErrorForm.vue';

const conversationStore = useConversationStore();
const session = useSessionStore();

// ── Render rows ───────────────────────────────────────────────────────────────
// Consecutive superseded (collapsed) ACT cycles fold into one collapsible group;
// everything else — including a live, non-collapsed act — renders on its own.
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

// ── Feed element ref ──────────────────────────────────────────────────────────

const feedRef = ref<HTMLElement | null>(null);

// ── Autoscroll composable ─────────────────────────────────────────────────────

const { scrollToBottom, forceScrollToBottom } = useAutoscroll(feedRef);

// ── History pagination — scroll-up trigger ────────────────────────────────────
//
// Port of chat.js init() scroll handler (lines 46-54).
// Fires on window scroll; when within 150px of the top and history is not
// already loading / exhausted, anchor-preserve then paginate.

let _paginating = false;

async function _onScrollPaginate(): Promise<void> {
  if (_paginating) return;
  if (session.historyLoading || session.historyExhausted) return;

  const scrollable = document.documentElement.scrollHeight > window.innerHeight + 100;
  if (!scrollable) return;
  if (window.scrollY >= 150) return;

  _paginating = true;
  try {
    // Anchor-preserve: capture height AND scroll position BEFORE the prepend
    // (chat.js lines 49-52: `anchor = body.scrollHeight - window.scrollY`
    // then after load: `scrollTo(0, body.scrollHeight - anchor)`).
    // `prevScrollY` MUST be read before the await — Chromium scroll-anchoring
    // (overflow-anchor, on by default) shifts scrollY once nodes land above the
    // viewport, so reading it post-prepend would double-count the offset.
    // Restoring to `prevScrollY + added` is algebraically identical to legacy:
    //   newHeight - anchor = newHeight - (oldHeight - prevScrollY) = prevScrollY + added
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

// ── Turn-end autoscroll ───────────────────────────────────────────────────────
//
// document events are raw CustomEvents (not in the typed event bus).
// session:turn-done → always force-scroll (spec: "always force-scroll on turn-end").
// session:history-initial-loaded → force-scroll after initial load.

function _onTurnDone(): void {
  forceScrollToBottom();
}

function _onHistoryInitialLoaded(): void {
  forceScrollToBottom();
}

// ── Deep-watch forms for incremental reactive scrolling ───────────────────────
//
// Legacy calls _scrollToBottom() after EVERY content mutation — new form,
// ACT narration, tool-pill add/resolve (renderer.js:170,239,261,302,374). A
// count-only watch would miss narration/pill growth inside an in-flight ACT
// form (forms.length is unchanged), so the feed would stop following the trail
// mid-cycle. A deep watch fires on those nested mutations too. flush:'post'
// runs the callback after Vue patches the DOM, so the rAF inside scrollToBottom
// measures the settled height. scrollToBottom is the GUARDED smooth variant —
// it self-skips when the user has scrolled up (turn-end uses the force variant).

watch(
  () => conversationStore.forms,
  () => {
    scrollToBottom();
  },
  { deep: true, flush: 'post' },
);

// ── Lifecycle ─────────────────────────────────────────────────────────────────

onMounted(async () => {
  // Wire document events for force-scroll signals
  document.addEventListener('session:turn-done', _onTurnDone);
  document.addEventListener('session:history-initial-loaded', _onHistoryInitialLoaded);

  // Initial history load — this is the ONLY trigger (App.vue does NOT call it)
  await session.loadRecentConversation();

  // Wire the pagination scroll listener ONLY AFTER the initial load + scroll-to-
  // bottom (legacy chat.js init() ordering): registering it earlier lets a short
  // conversation — where scrollY is 0 < 150 during the load — fire a pagination
  // cascade on startup.
  window.addEventListener('scroll', _onScrollPaginate, { passive: true });
});

onBeforeUnmount(() => {
  window.removeEventListener('scroll', _onScrollPaginate);
  document.removeEventListener('session:turn-done', _onTurnDone);
  document.removeEventListener('session:history-initial-loaded', _onHistoryInitialLoaded);
});
</script>

<template>
  <main id="conversationFeed" ref="feedRef" class="conversation-spine">
    <!-- History loader — shown at TOP while fetching older history pages -->
    <div v-if="session.historyLoading" class="history-loader">
      <span class="history-loader__spinner" aria-label="Loading history" role="status" />
    </div>

    <!-- "End of working history" pill — shown once history is fully exhausted -->
    <div v-if="session.historyExhausted" class="history-end-pill">
      <span class="history-end-pill__label">End of working history</span>
    </div>

    <!-- Forms grouped into turns: the user message + the assistant rows / act
         groups it produced share one `.turn` wrapper, so they render as a single
         visually-tight unit (intra-turn spacing < inter-turn spacing). -->
    <div v-for="turn in conversationStore.turns" :key="turn.id" class="turn">
      <template v-for="row in groupRows(turn.forms)" :key="row.id">
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
          <ErrorFormVue
            v-else-if="row.form.kind === 'error'"
            :form="(row.form as ErrorForm)"
          />
        </template>
      </template>
    </div>
  </main>
</template>

<style scoped lang="scss">
/*
  History loader and end-pill styles.
  `.conversation-spine` layout is owned by interface.scss (Task S5) — not duplicated here.
*/

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
  to {
    transform: rotate(360deg);
  }
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

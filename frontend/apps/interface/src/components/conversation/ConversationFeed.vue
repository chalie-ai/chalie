<!-- Renders conversation forms, drives history pagination, manages autoscroll. -->
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

const feedRef = ref<HTMLElement | null>(null);

const { scrollToBottom, forceScrollToBottom } = useAutoscroll(feedRef);

// History pagination: on scroll within 150px of the top (and not already
// loading/exhausted), anchor-preserve then paginate.
let _paginating = false;

async function _onScrollPaginate(): Promise<void> {
  if (_paginating) return;
  if (session.historyLoading || session.historyExhausted) return;

  const scrollable = document.documentElement.scrollHeight > window.innerHeight + 100;
  if (!scrollable) return;
  if (window.scrollY >= 150) return;

  _paginating = true;
  try {
    // Anchor-preserve: capture height AND scrollY BEFORE the prepend.
    // `prevScrollY` MUST be read before the await — Chromium scroll-anchoring
    // (overflow-anchor, on by default) shifts scrollY once nodes land above the
    // viewport, so reading it post-prepend would double-count the offset.
    // Restoring to `prevScrollY + added` keeps the visible content fixed.
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

  // Initial history load — this is the ONLY trigger (App.vue does NOT call it).
  await session.loadRecentConversation();

  // Wire the pagination listener ONLY AFTER the initial load: registering it
  // earlier lets a short conversation (scrollY 0 < 150 during load) fire a
  // pagination cascade on startup.
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
      <span class="history-loader__spinner" aria-label="Loading history" role="status" />
    </div>

    <div v-if="session.historyExhausted" class="history-end-pill">
      <span class="history-end-pill__label">End of working history</span>
    </div>

    <!-- Each turn shares one `.turn` wrapper so intra-turn spacing < inter-turn. -->
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
/* Loader + end-pill only; `.conversation-spine` layout is owned by interface.scss. */
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

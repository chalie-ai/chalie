<!-- Renders a turn block as gutterless speaker rows: assistant prose runs open
     to the left, the user message sits as a right-set bubble, and speaker-change
     spacing gives the rhythm (no avatar gutter). Shared by the main feed (inline
     turns) and the thread panel so both keep identical row rhythm. -->
<script setup lang="ts">
import { computed } from 'vue';
import { ConfigType } from '@chalie/shared';
import type { ConversationMessage, ConversationTurnBlock } from '../../api/conversation';
import type { LiveToolPill } from '../../utils/liveActTrail';
import { liveTrailsFor } from '../../utils/liveActTrail';
import UserBubble from './UserBubble.vue';
import ChalieBubble from './ChalieBubble.vue';
import ActCycle from './ActCycle.vue';
import ActCycleGroup from './ActCycleGroup.vue';
import BubbleFooter from './BubbleFooter.vue';

const props = withDefaults(
  defineProps<{
    block: ConversationTurnBlock;
    canReply?: boolean;
    type?: string;
    fullThread?: boolean;
    // When this turn is a forked thread, the spine passes its collapsed thread
    // pill (status + gist label) so it can ride INLINE on the settle0 footer's
    // meta line, next to the trace — no separate pill row. Null on the thread
    // panel (fullThread) and on non-forked turns.
    threadPill?: { status: 'working' | 'done' | 'thread' | 'idle'; label: string } | null;
  }>(),
  { canReply: true, type: ConfigType.USER, fullThread: false, threadPill: null },
);

const emit = defineEmits<{ reply: [turnId: number]; openThread: [turnId: number] }>();

/** A forked thread carries at least one row past its settle0 (see
 *  ConversationMessage.thread_message) — derived directly off the prop, no
 *  store/composable needed. */
const isForkedThread = computed(() => props.block.messages.some((m) => m.thread_message));

// The live act-trail is derived from WS signals (spec §6.5). While the turn
// works, append one transient non-collapsed ActRow carrying the turn's live tool
// pills (each driven started→done/error by the single tool-call frame). Before the
// first pill lands, a bare anchor renders the "thinking…" placeholder.
interface LiveActRow {
  kind: 'live-act';
  rowId: number;
  pills: LiveToolPill[];
}

interface MsgRow {
  kind: 'msg';
  message: ConversationMessage;
}

/** One tool call's collapsed summary, as carried by the API model. */
type ToolSummary = { tool_name: string; summary: string; state: string; ended_at: string | null };

interface CollapsedGroupRow {
  kind: 'collapsed-group';
  id: string;
  summaries: ToolSummary[];
}

// The completion-time footer — one per turn_exchange. It carries that exchange's
// aggregated tool calls (`toolCalls`) so its meta line can show the "N tools used"
// trace inline (expandable), matching the design: assistant prose → footer(trace).
interface FooterRow {
  kind: 'footer';
  message: ConversationMessage;
  toolCalls: ToolSummary[];
}

type DisplayRow = MsgRow | LiveActRow | CollapsedGroupRow | FooterRow;

const displayRows = computed<DisplayRow[]>(() => {
  const rows: DisplayRow[] = [];
  // A turn_exchange opens at each USER message and runs through its assistant
  // reply(-ies). Every tool call in it — pre-turn calls on the user row and calls
  // on each assistant step alike — is buffered here and RIDES ON that exchange's
  // footer as its aggregated "N tools used" trace (inline, expandable), so the
  // paint order within one exchange is: assistant prose → footer(with its trace),
  // never a tool chip hoisted past a later exchange. A still-streaming exchange
  // has no footer yet; its already-finished calls fall back to a collapsed-group
  // row so they are never dropped while the reply streams.
  let pendingTools: ToolSummary[] = [];
  let exchangeLastAssistant: ConversationMessage | null = null;

  // The turn's still-streaming reply is its LAST message; that exchange has
  // nothing "complete" to timestamp yet, so its footer waits for the turn to
  // settle.
  const streamingReply = props.block.working
    ? props.block.messages[props.block.messages.length - 1]
    : null;

  for (const message of props.block.messages) {
    // Spine renders only through settle0 — drop thread reply rows. The thread
    // panel (fullThread) renders the WHOLE thread, continuations included.
    if (!props.fullThread && message.thread_message) continue;

    // A new user message opens the next exchange: close the previous one — its
    // footer carries that exchange's aggregated tool calls beneath its last reply,
    // before this row.
    if (message.role === 'user') {
      if (exchangeLastAssistant) {
        rows.push({ kind: 'footer', message: exchangeLastAssistant, toolCalls: pendingTools });
        exchangeLastAssistant = null;
      } else if (pendingTools.length) {
        // Tool calls with no assistant reply to anchor a footer (rare) — keep
        // them visible as their own collapsed row rather than drop them.
        rows.push({ kind: 'collapsed-group', id: `pre-${message.id}`, summaries: pendingTools });
      }
      pendingTools = [];
    }

    rows.push({ kind: 'msg', message });
    if (message.role === 'assistant') exchangeLastAssistant = message;

    if (message.tool_calls?.length) pendingTools.push(...message.tool_calls);
  }

  // Tail — the last exchange. A settled one gets its footer (carrying its trace);
  // a settled earlier exchange that only reaches this tail push — on the spine the
  // streaming continuation is dropped, leaving the opener as the last VISIBLE
  // reply — keeps its footer while the fork streams, so the spine always shows
  // exactly one footer per turn_id. The still-streaming exchange has no footer,
  // so its finished calls fall back to a collapsed-group row.
  if (exchangeLastAssistant && exchangeLastAssistant !== streamingReply) {
    rows.push({ kind: 'footer', message: exchangeLastAssistant, toolCalls: pendingTools });
  } else if (pendingTools.length) {
    rows.push({ kind: 'collapsed-group', id: 'streaming', summaries: pendingTools });
  }

  // Live trails: appended at the tail while the turn is working, but only when
  // this render is the authoritative live view of the turn (thread panel, or a
  // non-forked turn in the spine). Forked turns in the spine show the thread
  // pill's animated dot instead — rendering "thinking..." inline would duplicate
  // that indicator and misattribute thread activity to the top-level timeline.
  if (props.block.working && (props.fullThread || !isForkedThread.value)) {
    const trails = liveTrailsFor(props.type, props.block.turn_id);
    if (trails.length) {
      for (const t of trails) {
        rows.push({ kind: 'live-act', rowId: t.rowId, pills: t.pills });
      }
    } else {
      rows.push({ kind: 'live-act', rowId: -1, pills: [] });
    }
  }

  return rows;
});

// Speaker-role grouping for the row rhythm.
type RowRole = 'user' | 'chalie';

interface RowEntry {
  key: string;
  role: RowRole;
  isLead: boolean;
  row: DisplayRow;
}

/** Key for a non-message row — collapsed tool group, live act-trail anchor, or footer. */
function nonMsgKey(row: LiveActRow | CollapsedGroupRow | FooterRow): string {
  if (row.kind === 'collapsed-group') return `cg-${row.id}`;
  if (row.kind === 'footer') return `footer-${row.message.id}`;
  return `live-${row.rowId}`;
}

const rowEntries = computed<RowEntry[]>(() => {
  let prevRole: RowRole | null = null;
  return displayRows.value.map((row) => {
    // A footer row has no user branch to match, so it falls into 'chalie' —
    // same as the collapsed-group/live-act rows — keeping it grouped under
    // the exchange's chalie rows so speaker-change spacing stays correct.
    const role: RowRole = row.kind === 'msg' && row.message.role === 'user' ? 'user' : 'chalie';
    const key = row.kind === 'msg' ? `msg-${row.message.id}` : nonMsgKey(row);
    const isLead = role !== prevRole;
    prevRole = role;
    return { key, role, isLead, row };
  });
});

// The thread pill rides on the exchange-closing footer. On the spine there is
// exactly one footer per turn (thread replies are dropped), but keying off the
// LAST footer stays correct if an exchange ever renders more than one.
const lastFooterKey = computed<string | null>(() => {
  const entries = rowEntries.value;
  for (let i = entries.length - 1; i >= 0; i--) {
    if (entries[i].row.kind === 'footer') return entries[i].key;
  }
  return null;
});

// A crashed turn that surfaced no assistant reply text settles to working:false
// with nothing but (maybe) a tool-trace footer — indistinguishable from a normal
// empty turn. Show an explicit note in exactly that case. A crash that DID leave
// a reply keeps it and needs no note. Gated on the VISIBLE rows (thread
// continuations are spine-dropped), so a fork-reply crash never mislabels a
// completed opener as failed.
const showCrashNote = computed<boolean>(() =>
  (props.block.crashed ?? false)
  && !displayRows.value.some(
    (r) => r.kind === 'msg' && r.message.role === 'assistant' && r.message.content.trim().length > 0,
  ),
);

function onReply(): void {
  emit('reply', props.block.turn_id);
}

function onOpenThread(): void {
  emit('openThread', props.block.turn_id);
}
</script>

<template>
  <div
    class="turn-view"
    :data-turn-id="block.turn_id"
    :data-type="type"
    :data-forked="isForkedThread || undefined"
    :data-gist="block.gist ?? undefined"
    :data-preview="block.preview"
    :data-last-activity="block.last_activity_at ?? undefined"
  >
    <div
      v-for="ar in rowEntries"
      :key="ar.key"
      class="msg-row"
      :class="[`msg-row--${ar.role}`, ar.isLead ? 'msg-row--lead' : 'msg-row--cont']"
    >
      <!-- Collapsed tool-call group -->
      <ActCycleGroup
        v-if="ar.row.kind === 'collapsed-group'"
        :summaries="(ar.row as CollapsedGroupRow).summaries"
      />

      <!-- Live act-trail anchor -->
      <ActCycle v-else-if="ar.row.kind === 'live-act'" :pills="(ar.row as LiveActRow).pills" />

      <!-- Completion-time footer — one per turn_exchange, below its act-trail.
           Its meta line carries the exchange's aggregated tool trace inline, and
           (on the settle0 footer of a forked turn) the collapsed thread pill. -->
      <BubbleFooter
        v-else-if="ar.row.kind === 'footer'"
        :message="(ar.row as FooterRow).message"
        :tool-calls="(ar.row as FooterRow).toolCalls"
        :can-reply="canReply"
        :thread-pill="ar.key === lastFooterKey ? threadPill : null"
        @reply="onReply"
        @open-thread="onOpenThread"
      />

      <!-- Message rows -->
      <template v-else>
        <UserBubble
          v-if="(ar.row as MsgRow).message.role === 'user'"
          :message="(ar.row as MsgRow).message"
        />
        <ChalieBubble v-else :message="(ar.row as MsgRow).message" />
      </template>
    </div>

    <!-- A crash that produced no reply — name the absence rather than leave a
         bare tool-trace footer read as an empty answer. -->
    <div v-if="showCrashNote" class="msg-row msg-row--chalie msg-row--lead">
      <p class="turn-crashed">This turn ended unexpectedly.</p>
    </div>
  </div>
</template>

<style scoped lang="scss">
.turn-view {
  display: flex;
  flex-direction: column;
}

/* Turn row: centred at the dock width, no gutter.
   Speaker-change rhythm — new speaker 30px, same-speaker continuation 6px. */
.msg-row {
  display: flex;
  width: 100%;
  max-width: var(--dock-width);
  margin-inline: auto;
}

.msg-row--user {
  justify-content: flex-end;
}

.msg-row--lead {
  margin-top: 30px;
}
.msg-row--cont {
  margin-top: 6px;
}

// A settled turn that failed with no reply — a muted, unobtrusive note (not an
// alarm banner); it explains an otherwise-blank exchange, matching the feed's
// restrained tone.
.turn-crashed {
  margin: 0;
  font-size: 13px;
  font-style: italic;
  color: var(--text-secondary);
}
</style>

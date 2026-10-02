<!-- Renders a turn block as gutterless speaker rows: assistant prose runs open
     to the left, the user message sits as a right-set bubble, and speaker-change
     spacing gives the rhythm (no avatar gutter). Shared by the main feed (inline
     turns) and the thread panel so both keep identical row rhythm. -->
<script setup lang="ts">
import { computed } from 'vue';
import { ConfigType } from '@chalie/shared';
import type {
  ConversationMessage,
  ConversationTurnBlock,
  DelegateTurnBlock,
} from '../../api/conversation';
import type { LiveToolPill } from '../../utils/liveActTrail';
import { liveTrailsFor } from '../../utils/liveActTrail';
import UserBubble from './UserBubble.vue';
import ChalieBubble from './ChalieBubble.vue';
import ActCycle from './ActCycle.vue';
import BubbleFooter from './BubbleFooter.vue';

const props = withDefaults(
  defineProps<{
    block: ConversationTurnBlock | DelegateTurnBlock;
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

/** Set when this block is a delegate (subagent) turn, read by its channel: the
 *  turn has no type, its stop undoes nothing, and its user-role rows are the
 *  task Chalie handed over. */
const channel = computed(() => props.block.channel ?? null);

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

// The meta line under one row, carrying that row's OWN tool calls and reasoning
// — never another row's. On the row that closes its exchange (`closing`) it is
// the full footer: timestamp, actions, thread pill. `standalone` marks one with
// no bubble of its own above it — a step row that only called tools, or a user
// row whose chips were stored before step rows existed.
interface FooterRow {
  kind: 'footer';
  message: ConversationMessage;
  closing: boolean;
  standalone: boolean;
}

type DisplayRow = MsgRow | LiveActRow | FooterRow;

/** Ids of the rows that close an exchange. An exchange opens at each user row
 *  and closes on its settled row. One that ended with none — cancelled, or
 *  crashed — closes on its last row with text, so its reply still gets a
 *  timestamp and actions, or on its last row when none said anything. The
 *  exchange a working turn is still running has no closing row yet. */
const closingRowIds = computed<Set<string>>(() => {
  const ids = new Set<string>();
  let settled = false;
  let lastWithText: string | null = null;
  let lastRow: string | null = null;
  const closeUnsettled = (): void => {
    const id = lastWithText ?? lastRow;
    if (!settled && id) ids.add(id);
  };
  for (const message of props.block.messages) {
    if (message.role === 'user') {
      closeUnsettled();
      settled = false;
      lastWithText = null;
      lastRow = null;
    } else if (message.settled) {
      ids.add(message.id);
      settled = true;
    } else {
      lastRow = message.id;
      if (message.content.trim().length > 0) lastWithText = message.id;
    }
  }
  if (!props.block.working) closeUnsettled();
  return ids;
});

const displayRows = computed<DisplayRow[]>(() => {
  const rows: DisplayRow[] = [];
  // One provider call is one assistant row, drawn on its own: its bubble when it
  // said something, then its footer when it closes its exchange or carries a
  // trace. A step row that only asked for tools has no text, so its trace stands
  // alone — no empty bubble.
  for (const message of props.block.messages) {
    // Spine renders only through settle0 — drop thread reply rows. The thread
    // panel (fullThread) renders the WHOLE thread, continuations included.
    if (!props.fullThread && message.thread_message) continue;

    const hasBubble = message.role === 'user' || message.content.trim().length > 0;
    if (hasBubble) rows.push({ kind: 'msg', message });

    const closing = closingRowIds.value.has(message.id);
    if (closing || message.tool_calls?.length || message.thinking) {
      rows.push({ kind: 'footer', message, closing, standalone: message.role === 'user' || !hasBubble });
    }
  }

  // Live trails: appended at the tail while the turn is working, but only when
  // this render is the authoritative live view of the turn (thread panel, or a
  // non-forked turn in the spine). Forked turns in the spine show the thread
  // pill's animated dot instead — rendering "thinking..." inline would duplicate
  // that indicator and misattribute thread activity to the top-level timeline.
  if (props.block.working && (props.fullThread || !isForkedThread.value)) {
    const trails = liveTrailsFor(channel.value ?? props.type, props.block.turn_id);
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

/** Key for a non-message row — a row's footer or a live act-trail anchor. */
function nonMsgKey(row: LiveActRow | FooterRow): string {
  return row.kind === 'footer' ? `footer-${row.message.id}` : `live-${row.rowId}`;
}

const rowEntries = computed<RowEntry[]>(() => {
  let prevRole: RowRole | null = null;
  return displayRows.value.map((row) => {
    // A footer row has no user branch to match, so it falls into 'chalie' —
    // same as the live-act rows, even under a user row's chips — keeping it
    // grouped with the chalie rows so speaker-change spacing stays correct.
    const role: RowRole = row.kind === 'msg' && row.message.role === 'user' ? 'user' : 'chalie';
    const key = row.kind === 'msg' ? `msg-${row.message.id}` : nonMsgKey(row);
    const isLead = role !== prevRole;
    prevRole = role;
    return { key, role, isLead, row };
  });
});

// The thread pill rides on the last closing footer. On the spine that is the
// opener's (thread replies are dropped), but keying off the LAST one stays
// correct when a render shows more than one exchange.
const lastFooterKey = computed<string | null>(() => {
  const entries = rowEntries.value;
  for (let i = entries.length - 1; i >= 0; i--) {
    const row = entries[i].row;
    if (row.kind === 'footer' && row.closing) return entries[i].key;
  }
  return null;
});

// A crashed turn that surfaced no assistant reply text settles to working:false
// with nothing but (maybe) tool traces — indistinguishable from a normal empty
// turn. Show an explicit note in exactly that case. A crash that DID leave a
// reply keeps it and needs no note. Gated on the VISIBLE rows (an assistant row
// only draws a bubble when it has text, and thread continuations are
// spine-dropped), so a fork-reply crash never mislabels a completed opener as
// failed.
const showCrashNote = computed<boolean>(() =>
  (props.block.crashed ?? false)
  && !displayRows.value.some((r) => r.kind === 'msg' && r.message.role === 'assistant'),
);

// A subagent the user stopped ends with nothing after its task — say so, or
// the panel reads as an empty answer.
const showStoppedNote = computed<boolean>(() => channel.value != null && (props.block.cancelled ?? false));

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
    :data-type="channel ? undefined : type"
    :data-channel="channel ?? undefined"
    :data-forked="isForkedThread || undefined"
    :data-gist="block.gist ?? undefined"
    :data-preview="block.preview"
    :data-last-activity="block.last_activity_at ?? undefined"
  >
    <div
      v-for="ar in rowEntries"
      :key="ar.key"
      class="msg-row"
      :class="[
        `msg-row--${ar.role}`,
        ar.isLead ? 'msg-row--lead' : 'msg-row--cont',
        { 'msg-row--standalone': ar.row.kind === 'footer' && (ar.row as FooterRow).standalone },
      ]"
    >
      <!-- Live act-trail anchor -->
      <ActCycle
        v-if="ar.row.kind === 'live-act'"
        :pills="(ar.row as LiveActRow).pills"
        :undoable="channel == null"
      />

      <!-- A row's footer: its own tool trace and reasoning. Only the row that
           closes its exchange adds the timestamp and actions, and (on the
           settle0 footer of a forked turn) the collapsed thread pill. -->
      <BubbleFooter
        v-else-if="ar.row.kind === 'footer'"
        :message="(ar.row as FooterRow).message"
        :trace-only="!(ar.row as FooterRow).closing"
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
          :label="channel ? 'Task from Chalie' : null"
        />
        <ChalieBubble v-else :message="(ar.row as MsgRow).message" />
      </template>
    </div>

    <!-- A crash that produced no reply — name the absence rather than leave a
         bare tool-trace footer read as an empty answer. -->
    <div v-if="showCrashNote" class="msg-row msg-row--chalie msg-row--lead">
      <p class="turn-crashed">This turn ended unexpectedly.</p>
    </div>
    <div v-if="showStoppedNote" class="msg-row msg-row--chalie msg-row--lead">
      <p class="turn-stopped">This subagent was stopped.</p>
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

// The meta's top margin spaces a footer from its own bubble; a footer with no
// bubble above it takes the row rhythm alone.
.msg-row--standalone :deep(.speech-form__meta) {
  margin-top: 0;
}

// A settled turn that failed with no reply, or a stopped subagent — a muted,
// unobtrusive note (not an alarm banner); it explains an otherwise-blank
// exchange, matching the feed's restrained tone.
.turn-crashed,
.turn-stopped {
  margin: 0;
  font-size: 13px;
  font-style: italic;
  color: var(--text-secondary);
}
</style>

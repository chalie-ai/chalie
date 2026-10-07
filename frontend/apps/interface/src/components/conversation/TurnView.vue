<!-- Renders a turn block as gutterless speaker rows: each segment of the turn
     reads as the user bubble, ONE quiet activity line (the segment's thinking,
     mid-turn prose and tool calls folded into steps), the answer bubble, and
     ONE footer — instead of one bubble + "N tools used" pill per LLM call.
     Speaker-change spacing gives the rhythm (no avatar gutter). Shared by the
     main feed (inline turns) and the thread panel so both keep identical row
     rhythm. -->
<script setup lang="ts">
import { computed } from 'vue';
import { ConfigType } from '@chalie/shared';
import type {
  ConversationMessage,
  ConversationTurnBlock,
  DelegateTurnBlock,
} from '../../api/conversation';
import { usePermissionsStore } from '../../stores/permissions';
import type { PermissionRequest } from '../../stores/permissions';
import type { LiveToolPill } from '../../utils/liveActTrail';
import { liveTrailsFor } from '../../utils/liveActTrail';
import UserBubble from './UserBubble.vue';
import ChalieBubble from './ChalieBubble.vue';
import ActivityLine from './ActivityLine.vue';
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

const permissions = usePermissionsStore();

/** This turn's parked permission ask, if any: the first queued gate whose
 *  `origin` names this view's (turn_id, type) identity — the same pair the
 *  lane resolver matches a card's origin on. A subagent's gate inherits its
 *  caller's origin, so it waits on the caller's line, never a delegate's. */
const pendingAsk = computed<PermissionRequest | null>(() =>
  channel.value == null
    ? (permissions.queue.find((r) => r.origin?.turn_id === props.block.turn_id && r.origin.type === props.type) ?? null)
    : null,
);

/** The still-working exchange, drawn as ONE quiet, expandable live line:
 *  every row of the segment — thinking, mid-turn prose, tool calls — folds
 *  into its steps, tracking the turn's act-trail `pills` (derived from WS
 *  signals, spec §6.5) and carrying the stop; before the first pill lands it
 *  is the "thinking…" anchor. When the exchange settles its work moves into
 *  the footer below the answer, so no line stands above it. */
interface ActivityRow {
  kind: 'activity';
  key: string;
  messages: ConversationMessage[];
  pills: LiveToolPill[];
}

interface MsgRow {
  kind: 'msg';
  message: ConversationMessage;
}

// The meta line under the row that closes an exchange: timestamp, actions, and
// (on the last footer of a forked turn) the collapsed thread pill. The work
// behind it lives HERE — the exchange's rows and duration behind the work
// toggle. `standalone` marks one with no bubble of its own above it — a
// closing row that said nothing.
interface FooterRow {
  kind: 'footer';
  message: ConversationMessage;
  standalone: boolean;
  work: ConversationMessage[];
  durationMs: number | null;
}

type DisplayRow = MsgRow | ActivityRow | FooterRow;

/** Ids of the rows that close an exchange. An exchange opens at each user row
 *  — except one that joined the running turn, which stays inside the exchange
 *  it joined — and closes on its settled row. One that ended with none — cancelled, or
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
      if (message.joined) continue;
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

/** Ids of the user rows a cancel of this working turn would remove: the
 *  trailing run of user rows nothing has answered yet, stopping at a row that
 *  joined the running turn — a cancel keeps that one and everything before it.
 *  The same rule the backend trims a cancelled turn by. A stop hands the text
 *  back to the dock only when its row is one of these. */
const droppedOnCancel = computed<Set<string>>(() => {
  const ids = new Set<string>();
  if (!props.block.working) return ids;
  const messages = props.block.messages;
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.role !== 'user' || message.joined) break;
    ids.add(message.id);
  }
  return ids;
});

const displayRows = computed<DisplayRow[]>(() => {
  // One segment is one exchange: it opens at a user row and closes on the row
  // that closes it (closingRowIds). The answer keeps its bubble when it said
  // something, and the closing row carries the exchange's single footer — the
  // exchange's work rows and duration folded into it. While the exchange is
  // still working, its open work stands as the live activity line, keyed the
  // same live and settled (anchor + segment index) so it does not remount
  // when the answer lands.
  const rows: DisplayRow[] = [];
  // anchor = id of the most recent user row ('start' before the first one);
  // segIndex = segment counter within the current anchor (reset at each user
  // row); pending = the work rows of the still-open exchange; lastFooter =
  // the footer of the last CLOSED exchange since the current user row —
  // trailing work after an exchange's answer stays in that exchange's footer;
  // exchangeStart = created_at of the row that opened the open exchange, for
  // its duration.
  let anchor = 'start';
  let segIndex = 0;
  let pending: ConversationMessage[] = [];
  let lastFooter: FooterRow | null = null;
  let exchangeStart: string | null = null;

  for (const message of props.block.messages) {
    // The spine renders only through settle0 — drop thread reply rows. The
    // thread panel (fullThread) renders the WHOLE thread, continuations included.
    if (!props.fullThread && message.thread_message) continue;

    if (message.role === 'user') {
      if (message.joined) {
        // A joined row continues the running exchange: its work and key stay
        // the open segment's.
        rows.push({ kind: 'msg', message });
        lastFooter = null;
        if (exchangeStart === null) exchangeStart = message.created_at;
        continue;
      }
      // A new exchange — the previous segment already closed on its closing
      // row (closingRowIds), so nothing flushes. The user row's own tool_calls
      // are the framework's turn-zero memory seed, never steps, so the
      // exchange opens with empty work.
      rows.push({ kind: 'msg', message });
      anchor = message.id;
      segIndex = 0;
      lastFooter = null;
      pending = [];
      exchangeStart = message.created_at;
      continue;
    }

    if (closingRowIds.value.has(message.id)) {
      // The exchange closes: its whole work — this closing row included —
      // hands to the footer under the answer, which keeps its bubble when it
      // said something; a closing row that only called tools leaves its footer
      // standing alone.
      pending.push(message);
      const hasBubble = message.content.trim().length > 0;
      if (hasBubble) rows.push({ kind: 'msg', message });
      let durationMs: number | null = null;
      if (exchangeStart !== null) {
        const start = Date.parse(exchangeStart);
        const end = Date.parse(message.created_at);
        if (!Number.isNaN(start) && !Number.isNaN(end) && end - start >= 0) {
          durationMs = end - start;
        }
      }
      const footer: FooterRow = { kind: 'footer', message, standalone: !hasBubble, work: pending, durationMs };
      rows.push(footer);
      lastFooter = footer;
      segIndex += 1;
      pending = [];
      exchangeStart = null;
      continue;
    }

    // A work row of the open exchange — or, after a closed one, trailing work
    // that stays with that exchange's footer.
    if (pending.length === 0 && lastFooter) {
      lastFooter.work.push(message);
    } else {
      if (exchangeStart === null) exchangeStart = message.created_at;
      pending.push(message);
    }
  }

  // Still working: the open exchange renders as the LIVE activity line —
  // carrying the act-trail pills (or a bare "thinking…" anchor before the
  // first pill lands) — on a key stable across live re-renders (anchor +
  // segment index), so it does not remount as the pills tick. Only when this
  // render is the authoritative live view of the turn (thread panel, or a
  // non-forked turn in the spine): a forked turn in the spine shows the thread
  // pill's animated dot instead, so an inline live line would duplicate that
  // indicator and misattribute thread activity to the top-level timeline.
  const liveView = props.block.working && (props.fullThread || !isForkedThread.value);
  if (liveView) {
    const pills = liveTrailsFor(channel.value ?? props.type, props.block.turn_id).flatMap((t) => t.pills);
    rows.push({ kind: 'activity', key: `activity-${anchor}-${segIndex}`, messages: pending, pills });
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

const rowEntries = computed<RowEntry[]>(() => {
  let prevRole: RowRole | null = null;
  return displayRows.value.map((row) => {
    // Activity lines and footers have no user branch to match, so they fall
    // into 'chalie' — keeping them grouped with the chalie rows so
    // speaker-change spacing stays correct.
    const role: RowRole = row.kind === 'msg' && row.message.role === 'user' ? 'user' : 'chalie';
    const key = row.kind === 'msg' ? `msg-${row.message.id}` : row.kind === 'activity' ? row.key : `footer-${row.message.id}`;
    const isLead = role !== prevRole;
    prevRole = role;
    return { key, role, isLead, row };
  });
});

// The thread pill rides on the last footer — every footer closes its exchange
// now, so the last one. On the spine that is the opener's (thread replies are
// dropped), but keying off the LAST one stays correct when a render shows more
// than one exchange.
const lastFooterKey = computed<string | null>(() => {
  const entries = rowEntries.value;
  for (let i = entries.length - 1; i >= 0; i--) {
    if (entries[i].row.kind === 'footer') return entries[i].key;
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
      <!-- The still-working exchange's LIVE activity line: its thinking,
           mid-turn prose and tool calls folded into steps, tracking the
           act-trail pill and carrying the stop. Settled, the exchange's work
           lives in the footer under the answer instead. -->
      <ActivityLine
        v-if="ar.row.kind === 'activity'"
        :messages="(ar.row as ActivityRow).messages"
        :pills="(ar.row as ActivityRow).pills"
        :undoable="channel == null"
        :ask="pendingAsk"
      />

      <!-- The segment's single footer: timestamp and actions — and (on the
           last footer of a forked turn) the collapsed thread pill. -->
      <BubbleFooter
        v-else-if="ar.row.kind === 'footer'"
        :message="(ar.row as FooterRow).message"
        :work="(ar.row as FooterRow).work"
        :duration-ms="(ar.row as FooterRow).durationMs"
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
          :data-dropped-on-cancel="droppedOnCancel.has((ar.row as MsgRow).message.id) || undefined"
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
  font-size: var(--fs-body);
  font-style: italic;
  color: var(--muted);
}
</style>

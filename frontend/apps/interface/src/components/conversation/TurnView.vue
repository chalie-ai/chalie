<!-- Renders a turn block as gutterless speaker rows: each segment of the turn
     reads as the user bubble, then each of Chalie's steps as a chat message of
     its own — its thinking and tool calls on one line, then what it said — and
     a footer under the step that closes the segment.
     Message spacing gives the rhythm (no avatar gutter). Shared by the main
     feed (inline turns) and the thread panel so both keep identical row
     rhythm. -->
<script setup lang="ts">
import { computed } from 'vue';
import { ConfigType } from '@chalie/shared';
import type {
  ConversationMessage,
  ConversationTurnBlock,
  DelegateTurnBlock,
  PlanTask,
} from '../../api/conversation';
import { usePermissionsStore } from '../../stores/permissions';
import type { PermissionRequest } from '../../stores/permissions';
import { liveTrailsFor } from '../../utils/liveActTrail';
import { didWork } from '../../utils/turnSteps';
import UserBubble from './UserBubble.vue';
import ChalieBubble from './ChalieBubble.vue';
import ActivityLine from './ActivityLine.vue';
import BubbleFooter from './BubbleFooter.vue';
import PlanCard from './PlanCard.vue';

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

// Only the authoritative live view of the turn (thread panel, or a non-forked
// turn in the spine) draws the live line: a forked turn in the spine shows the
// thread pill's animated dot instead, so an inline live line would duplicate
// that indicator and misattribute thread activity to the top-level timeline.
const liveView = computed(() => props.block.working && (props.fullThread || !isForkedThread.value));

/** The running step's live line: its calls are the turn's act-trail pills
 *  (derived from WS signals), and it carries the stop and the parked ask. */
const liveLine = computed(() => ({
  pills: liveView.value ? liveTrailsFor(channel.value ?? props.type, props.block.turn_id).flatMap((t) => t.pills) : [],
  undoable: channel.value == null,
  ask: pendingAsk.value,
}));

/** The running step: the row a live call was dropped from, else — before a
 *  refetch returned that row — none yet, so its block is the live line alone,
 *  reading "Thinking…" until the first pill lands. Its key never changes, so the
 *  live line never remounts as the step moves. */
interface ActivityRow {
  kind: 'activity';
  key: string;
  message: ConversationMessage | null;
}

/** A row as its speaker's message — an assistant row as its step's block. */
interface MsgRow {
  kind: 'msg';
  key: string;
  message: ConversationMessage;
}

// The meta line that ends a segment, under the row that closes it: timestamp,
// the segment's duration when it did work, the plan its exchange kept behind
// the plan link, actions, and (on the last footer of a forked turn) the
// collapsed thread pill.
interface FooterRow {
  kind: 'footer';
  key: string;
  message: ConversationMessage;
  durationMs: number | null;
  plan: PlanTask[];
}

/** The still-working exchange's plan, standing sticky under the message it
 *  plans and ticking as the agent works it. Settled, it moves behind the
 *  footer's plan link. */
interface PlanRow {
  kind: 'plan';
  key: string;
  tasks: PlanTask[];
}

type DisplayRow = MsgRow | ActivityRow | FooterRow | PlanRow;

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
  // Every row stands in order, by one rule: a user row as its bubble; an
  // assistant row as its step's block — its work, then its text — and, when it
  // settled, the footer that closes its segment. An exchange opens at each
  // user row — except one that joined the running turn, which stays in the
  // exchange it joined. One that ended with no settled row — cancelled or
  // crashed — closes on its last row with text, else its last row, its footer
  // right under that row. The exchange a working turn is still running has no
  // closing row yet: its live plan stands under its prompt and the running
  // step ends the list.
  const rows: DisplayRow[] = [];

  // A running call shows as its pill, and a refetch also returns it as a
  // `started` chip: a chip whose pill is live is dropped, so the call shows
  // once — the pill while it runs, the stored chip once the pill clears. The
  // row it was dropped from is the running step.
  const liveIds = new Set(liveLine.value.pills.map((p) => p.id));
  const activity: ActivityRow = { kind: 'activity', key: 'activity', message: null };

  // The spine renders only through settle0 — drop thread reply rows: every row
  // past settle0, so the exchange before them has settled. The thread panel
  // (fullThread) renders the WHOLE thread, continuations included.
  const messages = props.fullThread ? props.block.messages : props.block.messages.filter((m) => !m.thread_message);

  // The open exchange: its rows after the prompt, its plan, whether it settled,
  // and the footers its last row and last row with text would close it with.
  // The open segment: its start, whether it worked.
  let body: DisplayRow[] = [];
  let plan: PlanTask[] = [];
  let settled = false;
  let last: { at: number; footer: FooterRow } | undefined;
  let lastText: typeof last;
  let start = '';
  let worked = false;

  const flush = (open: boolean): void => {
    const closing = lastText ?? last;
    if (!settled && !open && closing) body.splice(closing.at, 0, closing.footer);
    rows.push(...body);
  };

  for (const stored of messages) {
    const calls = stored.tool_calls ?? [];
    const kept = calls.filter((c) => !liveIds.has(String(c.id)));
    const message = kept.length === calls.length ? stored : { ...stored, tool_calls: kept };
    const own: MsgRow = { kind: 'msg', key: `msg-${message.id}`, message };
    start ||= message.created_at;
    // A user row's tool calls are the turn-zero memory seed, never a step.
    if (message.role === 'user') {
      if (message.joined) {
        body.push(own);
      } else {
        flush(false);
        rows.push(own);
        body = [];
        plan = message.plan ?? [];
        settled = worked = false;
        last = lastText = undefined;
        start = message.created_at;
      }
      continue;
    }
    const live = !activity.message && kept.length < calls.length;
    const work = live || didWork(message);
    const text = message.content.trim();
    if (live) activity.message = message;
    if (work || text) body.push(live ? activity : own);
    worked ||= work;
    // A segment shows its duration only when it did work; the footer drops a
    // NaN or negative one.
    const durationMs = worked ? Date.parse(message.created_at) - Date.parse(start) : null;
    const footer: FooterRow = { kind: 'footer', key: `footer-${message.id}`, message, durationMs, plan };
    last = { at: body.length, footer };
    if (text) lastText = last;
    if (message.settled) {
      body.push(footer);
      settled = true;
      worked = false;
      start = '';
    }
  }

  if (liveView.value && !settled && plan.length > 0) rows.push({ kind: 'plan', key: 'plan', tasks: plan });
  // Only a live view's last exchange is still open: a forked spine shows just
  // the opener, which the thread past it closed.
  flush(liveView.value);
  if (liveView.value && !activity.message) rows.push(activity);
  return rows;
});

// Speaker-role grouping for the row rhythm.
type RowRole = 'user' | 'chalie';

interface RowEntry {
  role: RowRole;
  isLead: boolean;
  row: DisplayRow;
}

const rowEntries = computed<RowEntry[]>(() => {
  let prevRole: RowRole | null = null;
  return displayRows.value.map((row) => {
    // The running step, footers and plans have no user branch to match, so
    // they fall into 'chalie' — keeping them grouped with the chalie rows so
    // speaker-change spacing stays correct.
    const role: RowRole = row.kind === 'msg' && row.message.role === 'user' ? 'user' : 'chalie';
    const isLead = role !== prevRole;
    prevRole = role;
    return { role, isLead, row };
  });
});

// The thread pill rides on the last footer. On the spine that is the opener's
// (thread replies are dropped), but keying off the LAST one stays correct when
// a render shows more than one exchange.
const lastFooterKey = computed(() => displayRows.value.filter((r) => r.kind === 'footer').at(-1)?.key);

// A crashed turn that surfaced no assistant reply text settles to working:false
// with nothing but (maybe) tool traces — indistinguishable from a normal empty
// turn. Show an explicit note in exactly that case. A crash that DID leave a
// reply keeps it and needs no note. Gated on the VISIBLE rows (thread
// continuations are spine-dropped), so a fork-reply crash never mislabels a
// completed opener as failed.
const showCrashNote = computed<boolean>(() =>
  (props.block.crashed ?? false)
  && !displayRows.value.some((r) => r.kind === 'msg' && r.message.role === 'assistant' && r.message.content.trim() !== ''),
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
    :class="{ 'turn-view--panel': fullThread }"
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
      :key="ar.row.key"
      class="msg-row"
      :class="[`msg-row--${ar.role}`, ar.isLead ? 'msg-row--lead' : 'msg-row--cont', `msg-row--${ar.row.kind}`]"
    >
      <PlanCard v-if="ar.row.kind === 'plan'" :tasks="(ar.row as PlanRow).tasks" />

      <!-- The segment's footer: timestamp, duration, plan link and actions —
           and (on the last footer of a forked turn) the collapsed thread
           pill. -->
      <BubbleFooter
        v-else-if="ar.row.kind === 'footer'"
        :message="(ar.row as FooterRow).message"
        :duration-ms="(ar.row as FooterRow).durationMs"
        :plan="(ar.row as FooterRow).plan"
        :can-reply="canReply"
        :thread-pill="ar.row.key === lastFooterKey ? threadPill : null"
        @reply="onReply"
        @open-thread="onOpenThread"
      />

      <UserBubble
        v-else-if="ar.role === 'user'"
        :message="(ar.row as MsgRow).message"
        :label="channel ? 'Task from Chalie' : null"
        :data-dropped-on-cancel="droppedOnCancel.has((ar.row as MsgRow).message.id) || undefined"
      />

      <!-- A step's block. The running step's work line is the LIVE line: it
           tracks the act-trail pills and carries the stop. -->
      <ChalieBubble v-else :message="(ar.row as MsgRow | ActivityRow).message">
        <template v-if="ar.row.kind === 'activity'" #step>
          <ActivityLine v-bind="liveLine" :work="(ar.row as ActivityRow).message" />
        </template>
      </ChalieBubble>
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

// Each step is a message of its own, spaced from the one before it like one;
// a footer stays close under the step it closes.
.msg-row--chalie.msg-row--cont:not(.msg-row--footer) {
  margin-top: var(--space-lg);
}

// The live plan stays in view under the fixed 56px top bar while the work runs
// — or at the top of the thread panel, which scrolls its own body. It opens
// Chalie's side.
.msg-row--plan {
  position: sticky;
  top: 56px;
  z-index: 3;
}

.turn-view--panel .msg-row--plan {
  top: 0;
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

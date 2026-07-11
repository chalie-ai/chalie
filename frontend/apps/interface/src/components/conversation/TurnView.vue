<!-- Renders a turn block as Weave avatar rows: a 32px avatar gutter that
     shows on speaker change, then the message body. Shared by the main feed
     (inline turns) and the thread panel so both keep identical row rhythm. -->
<script setup lang="ts">
import { computed } from 'vue';
import { User } from '@lucide/vue';
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
  }>(),
  { canReply: true, type: ConfigType.USER, fullThread: false },
);

const emit = defineEmits<{ reply: [turnId: number] }>();

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

interface CollapsedGroupRow {
  kind: 'collapsed-group';
  id: string;
  summaries: { tool_name: string; summary: string; state: string; ended_at: string | null }[];
}

// The completion-time footer (timestamp/speak/reply) — one per turn_exchange,
// emitted as its own tail row AFTER that exchange's act-trail flush so it
// paints below the tool-call chips, never above them.
interface FooterRow {
  kind: 'footer';
  message: ConversationMessage;
}

type DisplayRow = MsgRow | LiveActRow | CollapsedGroupRow | FooterRow;

const displayRows = computed<DisplayRow[]>(() => {
  const rows: DisplayRow[] = [];
  // Tool-call chips render at the BOTTOM of the turn_exchange they belong to —
  // below that exchange's final reply, never interleaved between an interim
  // step and the reply, and never hoisted past a later exchange. A turn_exchange
  // opens at each USER message and runs through its assistant reply(-ies); every
  // tool group in it — pre-turn chips on the user row and chips on each assistant
  // step alike — is buffered here and flushed when the next exchange opens (or at
  // the end, for the last/still-working exchange), so chips are never dropped.
  // The exchange's footer flushes right after its act-trail, so within one
  // exchange the paint order is: assistant prose → act-trail → footer.
  let pending: CollapsedGroupRow[] = [];
  let exchangeLastAssistant: ConversationMessage | null = null;

  for (const message of props.block.messages) {
    // Spine renders only through settle0 — drop thread reply rows. The thread
    // panel (fullThread) renders the WHOLE thread, continuations included.
    if (!props.fullThread && message.thread_message) continue;

    // A new user message opens the next exchange: flush the previous exchange's
    // buffered tool chips beneath its last reply, then that exchange's footer,
    // before this row.
    if (message.role === 'user') {
      rows.push(...pending);
      pending = [];
      if (exchangeLastAssistant) {
        rows.push({ kind: 'footer', message: exchangeLastAssistant });
        exchangeLastAssistant = null;
      }
    }

    rows.push({ kind: 'msg', message });
    if (message.role === 'assistant') exchangeLastAssistant = message;

    if (message.tool_calls?.length) {
      pending.push({
        kind: 'collapsed-group',
        id: message.id,
        summaries: message.tool_calls,
      });
    }
  }

  rows.push(...pending);
  // The turn's still-streaming reply is its LAST message; that exchange has
  // nothing "complete" to timestamp yet, so its footer waits for the turn to
  // settle. But a settled earlier exchange that only reaches this tail push —
  // on the spine the streaming continuation is dropped, leaving the opener as
  // the last VISIBLE reply — is already complete and keeps its footer while the
  // fork streams, so the spine always shows exactly one footer per turn_id.
  const streamingReply = props.block.working
    ? props.block.messages[props.block.messages.length - 1]
    : null;
  if (exchangeLastAssistant && exchangeLastAssistant !== streamingReply) {
    rows.push({ kind: 'footer', message: exchangeLastAssistant });
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

// Avatar-role grouping for the Weave rhythm.
type AvatarRole = 'user' | 'chalie';

interface AvatarRow {
  key: string;
  role: AvatarRole;
  showAvatar: boolean;
  row: DisplayRow;
}

/** Key for a non-message row — collapsed tool group, live act-trail anchor, or footer. */
function nonMsgKey(row: LiveActRow | CollapsedGroupRow | FooterRow): string {
  if (row.kind === 'collapsed-group') return `cg-${row.id}`;
  if (row.kind === 'footer') return `footer-${row.message.id}`;
  return `live-${row.rowId}`;
}

const avatarRows = computed<AvatarRow[]>(() => {
  let prevRole: AvatarRole | null = null;
  return displayRows.value.map((row) => {
    // A footer row has no user branch to match, so it falls into 'chalie' —
    // same as the collapsed-group/live-act rows — keeping it grouped under
    // the exchange's chalie rows with no duplicate avatar.
    const role: AvatarRole = row.kind === 'msg' && row.message.role === 'user' ? 'user' : 'chalie';
    const key = row.kind === 'msg' ? `msg-${row.message.id}` : nonMsgKey(row);
    const ar: AvatarRow = { key, role, showAvatar: role !== prevRole, row };
    prevRole = role;
    return ar;
  });
});

function onReply(): void {
  emit('reply', props.block.turn_id);
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
      v-for="ar in avatarRows"
      :key="ar.key"
      class="msg-row"
      :class="[`msg-row--${ar.role}`, ar.showAvatar ? 'msg-row--lead' : 'msg-row--cont']"
    >
      <div class="msg-row__gutter" aria-hidden="true">
        <span v-if="ar.showAvatar && ar.role === 'chalie'" class="msg-avatar msg-avatar--chalie">
          <img src="/icons/icon.png" alt="" />
        </span>
        <span v-else-if="ar.showAvatar" class="msg-avatar msg-avatar--user">
          <User :size="15" />
        </span>
      </div>

      <div class="msg-row__body">
        <!-- Collapsed tool-call group -->
        <ActCycleGroup
          v-if="ar.row.kind === 'collapsed-group'"
          :summaries="(ar.row as CollapsedGroupRow).summaries"
        />

        <!-- Live act-trail anchor -->
        <ActCycle v-else-if="ar.row.kind === 'live-act'" :pills="(ar.row as LiveActRow).pills" />

        <!-- Completion-time footer — one per turn_exchange, below its act-trail -->
        <BubbleFooter
          v-else-if="ar.row.kind === 'footer'"
          :message="(ar.row as FooterRow).message"
          :can-reply="canReply"
          @reply="onReply"
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
    </div>
  </div>
</template>

<style scoped lang="scss">
.turn-view {
  display: flex;
  flex-direction: column;
}

/* Weave message row: centred at the dock width, 32px avatar gutter + 18px gap.
   Speaker-change rhythm — new speaker 30px, same-speaker continuation 6px. */
.msg-row {
  display: flex;
  gap: 18px;
  width: 100%;
  max-width: var(--dock-width);
  margin-inline: auto;
}

.msg-row--lead {
  margin-top: 30px;
}
.msg-row--cont {
  margin-top: 6px;
}

.msg-row__gutter {
  width: var(--avatar-size);
  flex-shrink: 0;
  display: flex;
  justify-content: center;
  padding-top: 1px;
}

.msg-row__body {
  flex: 1;
  min-width: 0;
}

.msg-avatar {
  width: var(--avatar-size);
  height: var(--avatar-size);
  display: grid;
  place-items: center;
  flex-shrink: 0;
}

// Chalie's mark is the bare gradient logo — no badge, no glow, no clip.
.msg-avatar--chalie img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

// Only the generic person avatar is a clipped circular badge.
.msg-avatar--user {
  border-radius: 50%;
  overflow: hidden;
  background: var(--bg-surface-2);
  border: 1px solid var(--border-strong);
  color: var(--text-secondary);
}
</style>

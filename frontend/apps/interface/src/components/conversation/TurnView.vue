<!-- Renders a sequence of forms as Weave avatar rows: a 32px avatar gutter that
     shows on speaker change, then the message body. Shared by the main feed
     (inline turns) and the thread panel so both keep identical row rhythm. -->
<script setup lang="ts">
import { computed } from 'vue';
import { User } from '@lucide/vue';
import { useConversationStore } from '../../stores/conversation';
import type { ConversationForm, UserForm, ChalieForm, ActForm } from '../../stores/conversation';
import UserBubble from './UserBubble.vue';
import ChalieBubble from './ChalieBubble.vue';
import ActCycle from './ActCycle.vue';
import ActCycleGroup from './ActCycleGroup.vue';

const props = withDefaults(
  defineProps<{ forms: ConversationForm[]; canReply?: boolean; working?: boolean }>(),
  { canReply: true, working: false },
);

const emit = defineEmits<{ reply: [turnId: number] }>();

const convo = useConversationStore();

// Sentinel id for the synthetic working anchor — negative so it never collides
// with a store-minted form id.
const WORKING_ANCHOR_ID = -1;

const turnId = computed<number | null>(() => {
  for (const f of props.forms) if (f.turnId != null) return f.turnId;
  return null;
});

// The spinner is no rendered form — it is derived from the turn's `working`
// signal. Append a transient ActForm anchor whose placeholder is the latest
// `tool_invoked` summary; ActCycle renders it as the live step.
const displayForms = computed<ConversationForm[]>(() => {
  if (!(props.working || convo.isTurnWorking(turnId.value))) return props.forms;
  const anchor: ActForm = {
    kind: 'act',
    id: WORKING_ANCHOR_ID,
    tools: [],
    collapsed: false,
    turnId: turnId.value,
    placeholder: convo.liveSummaryFor(turnId.value),
  };
  return [...props.forms, anchor];
});

// Consecutive superseded ACT cycles fold into one group; everything else
// (including a live, non-collapsed act) renders on its own row.
type RowBase =
  | { type: 'single'; id: number; form: ConversationForm }
  | { type: 'act-group'; id: number; forms: ActForm[] };
type AvatarRow = RowBase & { role: 'user' | 'chalie'; showAvatar: boolean };

const rows = computed<AvatarRow[]>(() => {
  const grouped: RowBase[] = [];
  for (const form of displayForms.value) {
    const last = grouped[grouped.length - 1];
    if (form.kind === 'act' && form.collapsed) {
      if (last?.type === 'act-group') last.forms.push(form);
      else grouped.push({ type: 'act-group', id: form.id, forms: [form] });
    } else {
      grouped.push({ type: 'single', id: form.id, form });
    }
  }
  let prevRole: 'user' | 'chalie' | null = null;
  return grouped.map((row) => {
    const role: 'user' | 'chalie' = row.type === 'single' && row.form.kind === 'user' ? 'user' : 'chalie';
    const annotated: AvatarRow = { ...row, role, showAvatar: role !== prevRole };
    prevRole = role;
    return annotated;
  });
});

function onReply(): void {
  if (turnId.value != null) emit('reply', turnId.value);
}
</script>

<template>
  <div class="turn-view">
    <div
      v-for="row in rows"
      :key="row.id"
      class="msg-row"
      :class="[`msg-row--${row.role}`, row.showAvatar ? 'msg-row--lead' : 'msg-row--cont']"
    >
      <div class="msg-row__gutter" aria-hidden="true">
        <span v-if="row.showAvatar && row.role === 'chalie'" class="msg-avatar msg-avatar--chalie">
          <img src="/icons/icon.png" alt="" />
        </span>
        <span v-else-if="row.showAvatar" class="msg-avatar msg-avatar--user">
          <User :size="15" />
        </span>
      </div>

      <div class="msg-row__body">
        <ActCycleGroup v-if="row.type === 'act-group'" :forms="row.forms" />
        <template v-else>
          <UserBubble v-if="row.form.kind === 'user'" :form="(row.form as UserForm)" />
          <ChalieBubble
            v-else-if="row.form.kind === 'chalie'"
            :form="(row.form as ChalieForm)"
            :can-reply="canReply"
            @reply="onReply"
          />
          <ActCycle v-else-if="row.form.kind === 'act'" :form="(row.form as ActForm)" />
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

.msg-row--lead { margin-top: 30px; }
.msg-row--cont { margin-top: 6px; }

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

// Chalie's mark is the bare gradient logo — no badge, no glow, no clip — so the
// transparent icon reads as the brand itself on both the dark and light scrims.
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

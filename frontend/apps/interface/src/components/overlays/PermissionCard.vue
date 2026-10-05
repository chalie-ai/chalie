<script setup lang="ts">
/**
 * PermissionCard — one pending permission gate: the model's one-line summary
 * of the gated action as the title, the gated permission beneath it, and
 * Allow / Deny (answered straight into the permissions store).
 *
 * Lane-agnostic: PermissionStack.vue decides which stack the card sits in.
 * When the card is on the spine but belongs to a thread or scheduled turn
 * (that turn's panel is not open) the stack passes `label` — "Thread ·
 * <heading>" / "Scheduled task · <heading>" — and the card shows it with an
 * Open button that asks the stack to open that turn's panel, where the card
 * then moves.
 *
 * The enter/leave transition classes live here with the card they animate:
 * the stack's TransitionGroup is `name="perm-card"` and this card is its
 * direct child, so the classes land on this root.
 */
import { computed } from 'vue';
import { Info } from '@lucide/vue';
import { usePermissionsStore, type PermissionRequest } from '../../stores/permissions';

const props = defineProps<{
  req: PermissionRequest;
  /** Where the card belongs while it is away from its turn; null when at home. */
  label: string | null;
}>();

const emit = defineEmits<{
  /** The user asked to open the turn this card belongs to. */
  open: [req: PermissionRequest];
}>();

const permissions = usePermissionsStore();

const ACTION_LABELS: Record<string, string> = {
  // email/calendar/contacts gate at the OUTER `pim` permission only — the inner
  // tools are INTERNAL on the backend and can never raise a permission_request.
  pim: 'Access Email, Calendar & Contacts',
  code_agent: 'Coding Agent',
  'browser.render': 'Read Webpage',
  'browser.interact': 'Interact with Webpage',
  'browser.screenshot': 'Screenshot Webpage',
  'browser.monitor': 'Monitor Webpage',
  'list.delete': 'Delete List',
  'memory.store': 'Store Memory',
  'memory.recall': 'Recall Memory',
  'memory.forget': 'Forget Memory',
  'memory.reflect': 'Reflect on Memory',
  'schedule.create': 'Create Schedule',
  'schedule.cancel': 'Cancel Schedule',
  'schedule.list': 'List Schedules',
  'schedule.search': 'Search Schedules',
  news: 'Fetch News',
  search: 'Web Search',
  weather: 'Check Weather',
  timer: 'Set Timer',
};

/** Readable label for an action_id; falls back to formatting the id (dots/underscores → spaces, title case). */
function actionLabel(actionId: string): string {
  if (ACTION_LABELS[actionId]) return ACTION_LABELS[actionId];
  return actionId.replace(/[._]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

/** The model's summary of the action leads; the permission's label stands in when the backend sent none. */
const title = computed(() => props.req.summary || actionLabel(props.req.action_id));
</script>

<template>
  <div
    class="perm-card"
    role="dialog"
    aria-label="Permission request"
    :data-request-id="req.request_id"
  >
    <div class="perm-card__body">
      <div v-if="label" class="perm-card__lane">
        <span class="perm-card__lane-text">{{ label }}</span>
        <button type="button" class="perm-card__lane-open" @click="emit('open', req)">
          Open
        </button>
      </div>

      <div class="perm-card__header">
        <span class="perm-card__icon" aria-hidden="true">
          <Info :size="14" />
        </span>
        <p class="perm-card__title">{{ title }}</p>
      </div>

      <p v-if="req.summary" class="perm-card__desc">{{ actionLabel(req.action_id) }}</p>

      <div class="perm-card__actions">
        <button
          class="perm-card__btn perm-card__btn--deny"
          @click="permissions.respond(req.request_id, false)"
        >
          Deny
        </button>
        <button
          class="perm-card__btn perm-card__btn--allow"
          @click="permissions.respond(req.request_id, true)"
        >
          Allow
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
.perm-card {
  // Both teleport targets (#permStack, #permStackPanel) are pointer-events:none
  // so their empty area doesn't block the chat behind them; re-enable here or
  // the Allow/Deny clicks fall through to the turn underneath. The transition
  // states below re-disable it mid enter/leave, which is intentional.
  pointer-events: auto;
  background: color-mix(in oklab, var(--surface, var(--surface)) 95%, transparent);
  border: 1px solid var(--line);
  overflow: hidden;

  // Light theme: soften the lift so it reads as depth, not a dark halo.
  // Plain `[data-theme] &` — :global() drops the `&` and leaks this onto <html>.
  [data-theme='light'] & {
  }
}

.perm-card-enter-from,
.perm-card-leave-to {
  opacity: 0;
  transform: translateY(20px);
}

.perm-card-leave-to {
  transform: translateY(-20px);
}

.perm-card-enter-active,
.perm-card-leave-active {
  transition:
    transform var(--dur-2) ease,
    opacity var(--dur-2) ease;
  pointer-events: none;
}

.perm-card-enter-to {
  opacity: 1;
  transform: none;
}

.perm-card__body {
  padding: var(--space-md);
}

// "Thread · heading" / "Scheduled task · heading" + Open — only on a spine
// card whose turn is shown elsewhere.
.perm-card__lane {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-sm);
  margin-bottom: var(--space-xs);
  font-size: var(--font-size-xs);
  color: var(--muted);
}

.perm-card__lane-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.perm-card__lane-open {
  flex-shrink: 0;
  padding: 2px var(--space-xs);
  font-size: var(--font-size-xs);
  font-weight: 500;
  line-height: 1.4;
  border: 1px solid var(--line);
  background: transparent;
  color: var(--pink-text);
  cursor: pointer;

  &:hover {
    background: color-mix(in oklab, var(--pink) 10%, transparent);
  }
}

.perm-card__header {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  margin-bottom: var(--space-xs);
}

.perm-card__icon {
  color: var(--pink-text);
  flex-shrink: 0;
  line-height: 1;
}

.perm-card__title {
  font-size: var(--font-size-sm);
  font-weight: 600;
  color: var(--text);
  margin: 0;
}

.perm-card__desc {
  font-size: var(--font-size-sm);
  color: var(--muted);
  margin: 0 0 var(--space-sm);
  line-height: 1.45;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.perm-card__actions {
  display: flex;
  gap: var(--space-xs);
  justify-content: flex-end;
}

.perm-card__btn {
  padding: 5px var(--space-sm);
  font-size: var(--font-size-sm);
  font-weight: 500;
  border: 1px solid transparent;
  cursor: pointer;
  transition:
    background var(--dur-1),
    border-color var(--dur-1),
    color var(--dur-1);
  line-height: 1.4;

  &--allow {
    background: color-mix(in oklab, var(--allow) 15%, transparent);
    border-color: color-mix(in oklab, var(--allow-text) 35%, transparent);
    color: var(--allow-text);

    &:hover {
      background: color-mix(in oklab, var(--allow) 25%, transparent);
      border-color: color-mix(in oklab, var(--allow-text) 55%, transparent);
    }
  }

  &--deny {
    background: color-mix(in oklab, var(--text) 5%, transparent);
    border-color: var(--line);
    color: var(--muted);

    &:hover {
      background: color-mix(in oklab, var(--text) 9%, transparent);
      border-color: color-mix(in oklab, var(--text) 15%, transparent);
      color: var(--text);
    }
  }
}
</style>

<script setup lang="ts">
/**
 * PermissionStack — slide-up card stack for backend `permission_request` events.
 *
 * Port source: frontend/interface/permission_notifications.js
 *
 * Renders into the `#permStack` teleport target provided by App.vue.
 * Cards are stacked vertically (newest on top via flex-direction: column-reverse
 * in the parent `.permission-stack`), each with Allow / Deny buttons.
 * No auto-deny — the card waits indefinitely for user input.
 *
 * The session store wires WS `permission_request` frames to
 * `usePermissionsStore().enqueue(data)` before this component mounts.
 */
import { usePermissionsStore } from '../../stores/permissions';
import type { PermissionRequest } from '../../stores/permissions';

const permissions = usePermissionsStore();

// ── Action label map — verbatim port from permission_notifications.js ─────────

const ACTION_LABELS: Record<string, string> = {
  'email.read':            'Read Email',
  'email.search':          'Search Email',
  'email.manage':          'Manage Email',
  'email.draft':           'Draft Email',
  'email.send':            'Send Email',
  'email.reply':           'Reply to Email',
  'email.forward':         'Forward Email',
  'calendar.list_events':  'List Calendar Events',
  'calendar.get_event':    'Read Calendar Event',
  'calendar.update_event': 'Update Calendar Event',
  'calendar.create_event': 'Create Calendar Event',
  'code_eval':             'Execute Code',
  'browser.render':        'Read Webpage',
  'browser.interact':      'Interact with Webpage',
  'browser.screenshot':    'Screenshot Webpage',
  'browser.monitor':       'Monitor Webpage',
  'document.search':       'Search Documents',
  'document.list':         'List Documents',
  'document.view':         'View Document',
  'document.create':       'Create Document',
  'document.delete':       'Delete Document',
  'document.restore':      'Restore Document',
  'list.delete':           'Delete List',
  'memory.store':          'Store Memory',
  'memory.recall':         'Recall Memory',
  'memory.forget':         'Forget Memory',
  'memory.reflect':        'Reflect on Memory',
  'schedule.create':       'Create Schedule',
  'schedule.cancel':       'Cancel Schedule',
  'schedule.list':         'List Schedules',
  'schedule.search':       'Search Schedules',
  'contacts':              'Access Contacts',
  'news':                  'Fetch News',
  'search':                'Web Search',
  'weather':               'Check Weather',
  'timer':                 'Set Timer',
};

/**
 * Convert an action_id to a readable label.
 * Falls back to formatting the id itself (dots/underscores → spaces, title case).
 * Matches the legacy `actionLabel()` function exactly.
 */
function actionLabel(actionId: string): string {
  if (ACTION_LABELS[actionId]) return ACTION_LABELS[actionId];
  return actionId
    .replace(/[._]/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// ── Handlers ──────────────────────────────────────────────────────────────────

function allow(req: PermissionRequest): void {
  void permissions.respond(req.request_id, true);
}

function deny(req: PermissionRequest): void {
  void permissions.respond(req.request_id, false);
}
</script>

<template>
  <Teleport to="#permStack">
    <TransitionGroup
      tag="div"
      name="perm-card"
      class="perm-stack-inner"
      aria-live="assertive"
      aria-label="Permission requests"
    >
      <div
        v-for="req in permissions.queue"
        :key="req.request_id"
        class="perm-card"
        role="dialog"
        aria-label="Permission request"
        :data-request-id="req.request_id"
      >
        <div class="perm-card__body">
          <!-- Header: info icon + action label -->
          <div class="perm-card__header">
            <span class="perm-card__icon" aria-hidden="true">
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
            </span>
            <p class="perm-card__title">{{ actionLabel(req.action_id) }}</p>
          </div>

          <!-- Description — shown only when the backend provides it -->
          <p v-if="req.description" class="perm-card__desc">{{ req.description }}</p>

          <!-- Deny first, Allow second — matches legacy DOM order -->
          <div class="perm-card__actions">
            <button
              class="perm-card__btn perm-card__btn--deny"
              @click="deny(req)"
            >
              Deny
            </button>
            <button
              class="perm-card__btn perm-card__btn--allow"
              @click="allow(req)"
            >
              Allow
            </button>
          </div>
        </div>
      </div>
    </TransitionGroup>
  </Teleport>
</template>

<style scoped lang="scss">
/*
 * The outer .permission-stack container is provided by App.vue (id="permStack").
 * The inner wrapper here carries the TransitionGroup tag so enter/leave
 * transforms apply directly to .perm-card elements.
 *
 * Legacy animation used CSS class toggling (perm-card--visible / perm-card--leaving).
 * In Vue we map that to TransitionGroup v-enter-from / v-leave-to.
 */

.perm-stack-inner {
  display: flex;
  flex-direction: column-reverse;
  gap: var(--space-sm);
}

// ── Card shell ────────────────────────────────────────────────────────────────

.perm-card {
  background: color-mix(in oklab, var(--surface, var(--bg-chalie)) 95%, transparent);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  box-shadow:
    0 8px 32px rgba(0, 0, 0, 0.35),
    0 2px 8px rgba(0, 0, 0, 0.20);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  overflow: hidden;

  // Light theme: soften the lift so it reads as depth, not a dark halo.
  // plain `[data-theme] &` — :global() drops the `&` and leaks this onto <html>.
  [data-theme="light"] & {
    box-shadow:
      0 2px 8px rgba(0, 0, 0, 0.08),
      0 8px 32px rgba(0, 0, 0, 0.12);
  }
}

// ── TransitionGroup enter / leave — mirrors perm-card--visible / perm-card--leaving ──

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
    transform var(--duration-normal) ease,
    opacity var(--duration-normal) ease;
  pointer-events: none;
}

.perm-card-enter-to {
  opacity: 1;
  transform: none;
}

// ── Card body & layout ────────────────────────────────────────────────────────

.perm-card__body {
  padding: var(--space-md);
}

.perm-card__header {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  margin-bottom: var(--space-xs);
}

.perm-card__icon {
  color: var(--accent-primary);
  flex-shrink: 0;
  line-height: 1;
}

.perm-card__title {
  font-size: var(--font-size-sm);
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.perm-card__desc {
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
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

// ── Buttons ───────────────────────────────────────────────────────────────────

.perm-card__btn {
  padding: 5px var(--space-sm);
  font-size: var(--font-size-sm);
  font-weight: 500;
  border-radius: var(--radius-sm);
  border: 1px solid transparent;
  cursor: pointer;
  transition:
    background var(--duration-fast),
    border-color var(--duration-fast),
    color var(--duration-fast);
  line-height: 1.4;

  &--allow {
    background: color-mix(in oklab, var(--success) 15%, transparent);
    border-color: color-mix(in oklab, var(--success) 35%, transparent);
    color: var(--success);

    &:hover {
      background: color-mix(in oklab, var(--success) 25%, transparent);
      border-color: color-mix(in oklab, var(--success) 55%, transparent);
    }
  }

  &--deny {
    background: color-mix(in oklab, var(--text) 5%, transparent);
    border-color: var(--border-subtle);
    color: var(--text-secondary);

    &:hover {
      background: color-mix(in oklab, var(--text) 9%, transparent);
      border-color: color-mix(in oklab, var(--text) 15%, transparent);
      color: var(--text-primary);
    }
  }
}
</style>

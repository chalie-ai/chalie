<script setup lang="ts">
/**
 * PermissionStack — the pending permission cards, each in the lane its turn
 * renders in: a gate bubbles up to the channel the request came from, for
 * every tool.
 *
 * One store queue, two lanes, decided per card by `resolvePermissionLane`:
 *   - spine → the `#permStack` target above the main dock (App.vue): main-
 *             spine turns, plus thread/scheduled turns whose panel is not
 *             open on them — those carry a "Thread · <heading>" /
 *             "Scheduled task · <heading>" label and an Open button.
 *   - panel → the `#permStackPanel` target ThreadPanel.vue renders above its
 *             own dock, for the turn the panel is open on.
 * The lane is a reactive decision (panel identity × origin), so opening the
 * panel on a card's turn moves the card in and closing it moves the card back
 * out — never a duplicate: the store holds one entry per request.
 *
 * The panel Teleport is `v-if`-gated on the panel being open AND deferred: a
 * Teleport whose target is missing when it mounts never re-resolves it, and
 * ThreadPanel renders its target only while open.
 */
import { computed } from 'vue';
import { useSessionStore } from '../../stores/session';
import { usePermissionsStore, type PermissionRequest } from '../../stores/permissions';
import { resolvePermissionLane } from '../../utils/permissionLane';
import { getTurnEl } from '../../utils/turnDom';
import PermissionCard from './PermissionCard.vue';

const session = useSessionStore();
const permissions = usePermissionsStore();

interface LaneCard {
  req: PermissionRequest;
  label: string | null;
}

const resolved = computed(() =>
  permissions.queue.map((req) => ({
    req,
    ...resolvePermissionLane(req.origin, {
      panelThreadId: session.panelThreadId,
      panelType: session.panelType,
    }),
  })),
);

const spineCards = computed<LaneCard[]>(() =>
  resolved.value
    .filter((c) => c.lane === 'spine')
    .map((c) => ({
      req: c.req,
      label: c.label == null ? null : `${c.label} · ${turnHeading(c.req)}`,
    })),
);

const panelCards = computed(() =>
  resolved.value.filter((c) => c.lane === 'panel').map((c) => c.req),
);

const panelOpen = computed(() => session.panelThreadId != null);

/**
 * The turn's heading as the spine already shows it (the TurnView host's
 * data-gist / data-preview); "#<turn_id>" when the turn is not rendered on the
 * spine — scheduled turns never are.
 */
function turnHeading(req: PermissionRequest): string {
  if (req.origin == null) return '';
  const el = getTurnEl(req.origin.turn_id, req.origin.type);
  return el?.dataset.gist || el?.dataset.preview || `#${req.origin.turn_id}`;
}

/** Open the panel on the card's turn — the card then re-resolves into the panel lane. */
function open(req: PermissionRequest): void {
  if (req.origin == null) return;
  session.openThreadPanel(req.origin.turn_id, req.origin.type);
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
      <PermissionCard
        v-for="card in spineCards"
        :key="card.req.request_id"
        :req="card.req"
        :label="card.label"
        @open="open"
      />
    </TransitionGroup>
  </Teleport>

  <Teleport v-if="panelOpen" to="#permStackPanel" defer>
    <TransitionGroup
      tag="div"
      name="perm-card"
      class="perm-stack-inner"
      aria-live="assertive"
      aria-label="Permission requests for this thread"
    >
      <PermissionCard
        v-for="req in panelCards"
        :key="req.request_id"
        :req="req"
        :label="null"
        @open="open"
      />
    </TransitionGroup>
  </Teleport>
</template>

<style scoped lang="scss">
// The inner wrapper carries the TransitionGroup tag so enter/leave transforms
// apply directly to the cards (the outer targets are in App.vue / ThreadPanel.vue).

.perm-stack-inner {
  display: flex;
  flex-direction: column-reverse;
  gap: var(--space-sm);
}
</style>

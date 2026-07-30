/**
 * Reactive "is this dock's lane currently working" read, derived from the
 * DOM contract's `data-working` attribute (D3), and sharing its single busy
 * authority (`isLaneWorking`) with the send/queue gate so the visual and the
 * gate can never disagree. `turnId() === null` means the spine's lane, which
 * has no turn_id of its own and so takes the reserved pair; otherwise the
 * lane is that thread's own turn.
 *
 * Initializes from a live DOM query, then stays in sync via the
 * `turn-state-changed` CustomEvent turnDom's setTurnWorking dispatches (same
 * reactive-via-DOM-event pattern established by SpineTurn.vue), plus a
 * `watchEffect` so a change to the dock's OWN target (e.g. the thread panel
 * switching threads) re-reads immediately.
 */
import { onBeforeUnmount, onMounted, ref, watchEffect } from 'vue';
import type { Ref } from 'vue';
import { isLaneWorking, SPINE_LANE_TURN_ID, SPINE_LANE_TYPE } from '../utils/turnDom';

export function useDockBusy(turnId: () => number | null, type: () => string): Ref<boolean> {
  const busy = ref(false);

  function refresh(): void {
    const id = turnId();
    busy.value =
      id == null ? isLaneWorking(SPINE_LANE_TYPE, SPINE_LANE_TURN_ID) : isLaneWorking(type(), id);
  }

  // 'turn-upserted' matters too: `stampWorking` applies data-working at
  // mount time (the fetch-render after a brand-new turn's 'working' frame)
  // without a state-changed dispatch of its own.
  watchEffect(refresh);
  onMounted(() => {
    document.addEventListener('turn-state-changed', refresh);
    document.addEventListener('turn-upserted', refresh);
  });
  onBeforeUnmount(() => {
    document.removeEventListener('turn-state-changed', refresh);
    document.removeEventListener('turn-upserted', refresh);
  });

  return busy;
}

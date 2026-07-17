/**
 * Reactive "is this dock's target currently working" read, derived from the
 * DOM contract's `data-working` attribute (D3). `turnId() === null` means
 * the main spine (no stable turn_id to key off until a brand-new send's POST
 * resolves one, so busy = anything working within the registered spine
 * surface); otherwise busy = that specific turn's own working state.
 *
 * Initializes from a live DOM query, then stays in sync via the
 * `turn-state-changed` CustomEvent turnDom's setTurnWorking dispatches (same
 * reactive-via-DOM-event pattern established by SpineTurn.vue), plus a
 * `watchEffect` so a change to the dock's OWN target (e.g. the thread panel
 * switching threads) re-reads immediately.
 */
import { onBeforeUnmount, onMounted, ref, watchEffect } from 'vue';
import type { Ref } from 'vue';
import { isSurfaceWorking, isTurnWorking, SPINE_SURFACE_ID } from '../utils/turnDom';

export function useDockBusy(turnId: () => number | null, type: () => string): Ref<boolean> {
  const busy = ref(false);

  function refresh(): void {
    const id = turnId();
    busy.value = id == null ? isSurfaceWorking(SPINE_SURFACE_ID) : isTurnWorking(id, type());
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

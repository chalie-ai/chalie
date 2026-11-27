import { onUnmounted, ref, watch } from 'vue';
import type { Ref } from 'vue';

/**
 * The clock ask rows age their "waiting for you" note from: it ticks once a
 * second while `count()` rows are showing and stops when none are, or when
 * the calling component unmounts.
 */
export function useWaitClock(count: () => number): Ref<number> {
  const now = ref(Date.now());
  let timer: ReturnType<typeof setInterval> | null = null;

  function stopClock(): void {
    if (timer) clearInterval(timer);
    timer = null;
  }

  watch(
    count,
    (n) => {
      if (n > 0 && !timer) {
        timer = setInterval(() => {
          now.value = Date.now();
        }, 1000);
      } else if (n === 0) {
        stopClock();
      }
    },
    { immediate: true },
  );

  onUnmounted(stopClock);
  return now;
}

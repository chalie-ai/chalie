/**
 * useAutoscroll — port of renderer.js forceScrollToBottom + _initScrollTracking.
 *
 * The feed layout scrolls at document/window level (no overflow:auto on the
 * <main> element), so scroll events and position measurements use window /
 * document, exactly as the legacy renderer does.  `feedRef` is used only for
 * querying <img> elements inside the spine (matching renderer.js line 579:
 * `this._spine?.querySelectorAll('img')`).
 */
import { ref, onMounted, onBeforeUnmount } from 'vue';
import type { Ref } from 'vue';

export function useAutoscroll(feedRef: Ref<HTMLElement | null>) {
  const userScrolledUp = ref(false);

  // ── Scroll-position tracking ───────────────────────────────────────────────

  function _onScroll(): void {
    const scrollBottom =
      document.documentElement.scrollHeight - window.scrollY - window.innerHeight;
    userScrolledUp.value = scrollBottom > 100;
  }

  onMounted(() => {
    window.addEventListener('scroll', _onScroll, { passive: true });
  });

  onBeforeUnmount(() => {
    window.removeEventListener('scroll', _onScroll);
  });

  // ── Force-scroll to bottom ─────────────────────────────────────────────────

  /**
   * Unconditionally scroll to the very bottom.
   *
   * Port of renderer.js forceScrollToBottom (lines 571-593):
   *   1. Reset the userScrolledUp guard.
   *   2. Two nested rAFs — first fires after the browser applies layout for
   *      just-appended nodes; second fires a frame later for any remaining
   *      paint work.
   *   3. After the inner rAF, query every <img> inside the feed and re-scroll
   *      once each late-loading image fires `load` or `error` — images arrive
   *      async and shift scrollHeight after the initial scroll.
   *
   * Safe when feedRef.value is null (guards at each access point).
   */
  function forceScrollToBottom(): void {
    userScrolledUp.value = false;

    const scroll = (): void => {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'instant' });
    };

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        scroll();

        const spine = feedRef.value;
        if (!spine) return;

        const imgs = spine.querySelectorAll<HTMLImageElement>('img');
        for (const img of imgs) {
          if (img.complete) continue;
          // Re-scroll once the image lands, but only if the user hasn't
          // scrolled away since — matching renderer.js line 585-589.
          const retry = (): void => {
            if (!userScrolledUp.value) scroll();
          };
          img.addEventListener('load', retry, { once: true });
          img.addEventListener('error', retry, { once: true });
        }
      });
    });
  }

  return { forceScrollToBottom, userScrolledUp };
}

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
    userScrolledUp.value =
      document.documentElement.scrollHeight - window.scrollY - window.innerHeight > 100;
  }

  onMounted(() => {
    window.addEventListener('scroll', _onScroll, { passive: true });
  });

  onBeforeUnmount(() => {
    window.removeEventListener('scroll', _onScroll);
  });

  // ── Guarded scroll to bottom (incremental appends) ─────────────────────────

  /**
   * Smooth scroll to the bottom, but ONLY if the user has not scrolled up.
   *
   * Port of renderer.js _scrollToBottom (lines 550-558): used for every
   * incremental content mutation — new form, ACT narration, tool-pill add /
   * resolve — so the feed follows growing content without yanking a user who
   * has deliberately scrolled up to read history. A single rAF defers the
   * measurement until the browser has laid out the just-appended nodes.
   *
   * (The legacy method also calls _updateAmbientBloom(); that speaker-bloom
   * highlight is a separate concern wired by the ambient task, not autoscroll.)
   */
  function scrollToBottom(): void {
    if (userScrolledUp.value) return;
    requestAnimationFrame(() => {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    });
  }

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

        for (const img of spine.querySelectorAll<HTMLImageElement>('img')) {
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

  return { scrollToBottom, forceScrollToBottom, userScrolledUp };
}

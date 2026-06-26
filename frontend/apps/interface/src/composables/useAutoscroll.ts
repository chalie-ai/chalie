/**
 * Document/window-level autoscroll: the feed has no overflow:auto, so scroll
 * events and position measurements use window/document. `feedRef` is used only
 * for querying <img> elements inside the spine.
 */
import { ref, onMounted, onBeforeUnmount } from 'vue';
import type { Ref } from 'vue';

export function useAutoscroll(feedRef: Ref<HTMLElement | null>) {
  const userScrolledUp = ref(false);

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

  /**
   * Expanded thread cards cap at 50vh and scroll their own body, so the latest
   * reply lives inside `.thread-card__body`, not the window. Pin each body to
   * its bottom. `force` overrides the per-body guard that otherwise leaves a
   * body alone once the user has scrolled up within it (reading that thread's
   * history), so live appends follow the trail without yanking a reader.
   */
  function _pinCardBodies(force: boolean): void {
    const spine = feedRef.value;
    if (!spine) return;
    for (const body of spine.querySelectorAll<HTMLElement>('.thread-card__body')) {
      const atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 100;
      if (force || atBottom) body.scrollTop = body.scrollHeight;
    }
  }

  /**
   * Smooth scroll to bottom on incremental appends, but ONLY if the user hasn't
   * scrolled up (so reading history isn't yanked). The rAF defers measurement
   * until the just-appended nodes are laid out.
   */
  function scrollToBottom(): void {
    if (userScrolledUp.value) return;
    requestAnimationFrame(() => {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
      _pinCardBodies(false);
    });
  }

  /**
   * Unconditionally scroll to the very bottom. Two nested rAFs straddle layout
   * + paint of just-appended nodes; then re-scroll once each late-loading <img>
   * fires load/error, since async images shift scrollHeight after the scroll.
   */
  function forceScrollToBottom(): void {
    userScrolledUp.value = false;

    const scroll = (): void => {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'instant' });
      _pinCardBodies(true);
    };

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        scroll();

        const spine = feedRef.value;
        if (!spine) return;

        for (const img of spine.querySelectorAll<HTMLImageElement>('img')) {
          if (img.complete) continue;
          // Re-scroll once the image lands, unless the user scrolled away since.
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

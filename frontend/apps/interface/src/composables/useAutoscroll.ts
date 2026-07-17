/**
 * Document/window-level autoscroll: the feed has no overflow:auto, so scroll
 * events and position measurements use window/document. `feedRef` is used only
 * for querying <img> elements inside the spine.
 */
import type { Ref } from 'vue';
import { onBeforeUnmount, onMounted, ref } from 'vue';

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
   * Smooth scroll to bottom on incremental appends, but ONLY if the user hasn't
   * scrolled up (so reading history isn't yanked). The rAF defers measurement
   * until the just-appended nodes are laid out.
   */
  function scrollToBottom(): void {
    if (userScrolledUp.value) return;
    requestAnimationFrame(() => {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    });
  }

  /** Re-scroll once each late-loading <img> inside `spine` lands, unless the
   *  user scrolled away since — split out of `forceScrollToBottom` so its
   *  nested callbacks stay within the nesting-depth limit. */
  function _retryImageScroll(spine: HTMLElement, scroll: () => void): void {
    for (const img of spine.querySelectorAll<HTMLImageElement>('img')) {
      if (img.complete) continue;
      const retry = (): void => {
        if (!userScrolledUp.value) scroll();
      };
      img.addEventListener('load', retry, { once: true });
      img.addEventListener('error', retry, { once: true });
    }
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
    };

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        scroll();

        const spine = feedRef.value;
        if (!spine) return;

        _retryImageScroll(spine, scroll);
      });
    });
  }

  return { scrollToBottom, forceScrollToBottom, userScrolledUp };
}

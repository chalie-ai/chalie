import { useAsyncResource, type AsyncResource } from '@chalie/shared';
import { apiErrorMessage } from '../api/http';
import { useToast } from './useToast';

/**
 * Brain flavour of `useAsyncResource`: a fetch failure is surfaced as a toast
 * (`apiErrorMessage(e, failMsg)`), which is the contract every brain list panel
 * repeated by hand. Read-only sub-views that render an inline empty/failed
 * state instead of toasting should use `useAsyncResource` directly and read
 * its `error` ref.
 */
export function useBrainResource<T>(
  fetcher: () => Promise<T>,
  opts: { initial: T; failMsg: string; immediate?: boolean },
): AsyncResource<T> {
  const { show } = useToast();
  return useAsyncResource(fetcher, {
    initial: opts.initial,
    immediate: opts.immediate,
    onError: (e) => show(apiErrorMessage(e, opts.failMsg), 'error'),
  });
}

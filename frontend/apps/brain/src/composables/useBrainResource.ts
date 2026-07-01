import { type AsyncResource, useAsyncResource } from '@chalie/shared';
import { apiErrorMessage } from '../api/http';
import { useToast } from './useToast';

/**
 * `useAsyncResource` that toasts fetch failures. Sub-views wanting an inline
 * empty/failed state should use `useAsyncResource` directly and read `error`.
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

import { apiErrorMessage } from '../api/http';
import { useToast } from './useToast';

/**
 * Write-path counterpart to `useBrainResource`: runs a mutation with uniform
 * toast feedback — an optional success toast on resolve, and an
 * `apiErrorMessage(e, failMsg, networkMsg)` error toast on reject. Returns
 * `{ ok, data }` so callers can gate follow-up work (reloading, closing a
 * dialog, updating local state) on the mutation actually succeeding.
 */
export function useBrainAction() {
  const { show } = useToast();

  async function run<T>(
    fn: () => Promise<T>,
    opts: { success?: string; failMsg: string; networkMsg?: string },
  ): Promise<{ ok: true; data: T } | { ok: false; data?: undefined }> {
    try {
      const data = await fn();
      if (opts.success) show(opts.success, 'success');
      return { ok: true, data };
    } catch (e) {
      show(apiErrorMessage(e, opts.failMsg, opts.networkMsg), 'error');
      return { ok: false };
    }
  }

  return { run };
}

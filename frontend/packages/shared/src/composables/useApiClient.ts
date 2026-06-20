import { ApiClient } from '../services/ApiClient';
import { getHost } from '../config/host';

let client: ApiClient | null = null;

/** Singleton ApiClient bound to the shared configurable host. */
export function useApiClient(): ApiClient {
  return (client ??= new ApiClient(getHost));
}

/**
 * Ready-to-use singleton ApiClient — the same instance `useApiClient()` returns.
 *
 * Prefer this at call sites: `import { api } from '@chalie/shared'` then
 * `api.del('/documents/' + id)`. The client deals with auth itself (sends the
 * session cookie; redirects to /login/ on 401 unless the caller opts out), so
 * there is no per-call auth wrapper to remember.
 */
export const api: ApiClient = useApiClient();

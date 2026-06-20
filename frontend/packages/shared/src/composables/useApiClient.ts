import { ApiClient } from '../services/ApiClient';
import { getHost } from '../config/host';

let client: ApiClient | null = null;

/** Singleton ApiClient bound to the shared configurable host. */
export function useApiClient(): ApiClient {
  return (client ??= new ApiClient(getHost));
}

/**
 * Same singleton `useApiClient()` returns; prefer this at call sites. The client
 * handles auth itself (session cookie; redirects to /login/ on 401 unless the
 * caller opts out), so there is no per-call auth wrapper.
 */
export const api: ApiClient = useApiClient();

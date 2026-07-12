import { ApiClient } from '../services/ApiClient';
import { getHost, getToken } from '../config/host';

/**
 * Shared singleton ApiClient bound to the configurable host. The client
 * handles auth itself (session cookie; redirects to /login/ on 401 unless the
 * caller opts out), so there is no per-call auth wrapper.
 */
export const api: ApiClient = new ApiClient(getHost, getToken);

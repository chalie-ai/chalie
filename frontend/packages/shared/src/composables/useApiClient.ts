import { ApiClient } from '../services/ApiClient';
import { getHost } from '../config/host';

let client: ApiClient | null = null;

/** Singleton ApiClient bound to the shared configurable host. */
export function useApiClient(): ApiClient {
  if (!client) client = new ApiClient(getHost);
  return client;
}

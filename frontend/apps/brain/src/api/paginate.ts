/**
 * Page-walking helper for the Endpoint-contract listing routes.
 *
 * Listing responses are { success, result: [...], pagination: { page, limit,
 * total } }. `fetchAllPages` fetches `basePath` at the backend's max limit
 * (100) and loops pages until `pagination.total` rows are collected. Extra
 * per-request query params (e.g. `include_deleted`) are threaded through
 * `extraParams` and preserved across every page fetch.
 */
import { api } from '@chalie/shared';

export interface ListingEnvelope<T> {
  success: true;
  result: T[];
  pagination: { page: number; limit: number; total: number };
}

export async function fetchAllPages<T>(
  basePath: string,
  extraParams?: Record<string, string>,
): Promise<T[]> {
  const all: T[] = [];
  let page = 1;
  const suffix = extraParams
    ? Object.entries(extraParams)
        .map(([k, v]) => `&${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
        .join('')
    : '';
  while (true) {
    const body = await api.get<ListingEnvelope<T>>(
      `${basePath}?page=${page}&limit=100${suffix}`,
    );
    const items = body.result;
    if (items.length === 0) break; // stale total must not loop forever
    const total = body.pagination?.total ?? items.length;
    all.push(...items);
    if (all.length >= total) break;
    page++;
  }
  return all;
}

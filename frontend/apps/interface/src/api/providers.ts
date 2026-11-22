import { api } from '@chalie/shared';

/** A configured model, as the dock's model menu lists it. */
export interface Provider {
  id: number;
  name: string;
  /** Its context window in tokens; null until the model is first used. */
  context_window: number | null;
}

interface ListingEnvelope<T> {
  success: true;
  result: T[];
  pagination: { page: number; limit: number; total: number };
}

/** The most rows the server returns on one page. */
const PAGE_LIMIT = 100;

export const providers = {
  /** GET /api/providers/all — every configured model, read page by page until
   *  the listing's total is reached. */
  async all(): Promise<Provider[]> {
    const rows: Provider[] = [];
    for (let page = 1; ; page++) {
      const body = await api.get<ListingEnvelope<Provider>>(`/api/providers/all?page=${page}&limit=${PAGE_LIMIT}`);
      rows.push(...body.result);
      if (body.result.length === 0 || rows.length >= body.pagination.total) return rows;
    }
  },
};

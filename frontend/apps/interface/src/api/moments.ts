import { api } from '@chalie/shared';

/** A single moment as returned by the moments API. */
export interface Moment {
  id: number | null;
  transcript_id: number | null;
  key: string;
  value: string;
  /** Alias for `value` — moment_search.js renders item.message_text. */
  message_text: string;
  created_at: string | null;
}

export const moments = {
  /** GET /moments/search?q=<query> — semantic search over pinned moments. */
  search(q: string): Promise<{ items: Moment[] }> {
    return api.get(`/moments/search?q=${encodeURIComponent(q)}`);
  },

  /** POST /moments — pin an assistant message; body { message_text }. */
  pin(content: string): Promise<{ item: Moment; duplicate: boolean }> {
    return api.post('/moments', { message_text: content });
  },

  /** POST /moments/<tid>/forget — soft-delete a moment by transcript ID. */
  forget(tid: number): Promise<{ ok: boolean }> {
    return api.post(`/moments/${tid}/forget`, {});
  },
};

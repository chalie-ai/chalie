import { useApiClient } from '@chalie/shared';

/** A single moment as returned by the moments API. */
export interface Moment {
  id: number | null;
  transcript_id: number | null;
  key: string;
  value: string;
  /** Alias for value — rendered by moment_search.js as item.message_text. */
  message_text: string;
  created_at: string | null;
}

export const moments = {
  /**
   * Semantic search over pinned moments.
   * GET /moments/search?q=<query>
   */
  search(q: string): Promise<{ items: Moment[] }> {
    const api = useApiClient();
    return api.get(`/moments/search?q=${encodeURIComponent(q)}`);
  },

  /**
   * Pin an assistant message as a moment.
   * POST /moments — body: { message_text } or { transcript_id }
   */
  pin(content: string): Promise<{ item: Moment; duplicate: boolean }> {
    const api = useApiClient();
    return api.post('/moments', { message_text: content });
  },

  /**
   * Soft-delete (forget) a moment by its transcript ID.
   * POST /moments/<tid>/forget
   */
  forget(tid: number): Promise<{ ok: boolean }> {
    const api = useApiClient();
    return api.post(`/moments/${tid}/forget`, {});
  },
};

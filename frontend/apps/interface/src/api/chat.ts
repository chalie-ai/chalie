import { api } from '@chalie/shared';

/** GET /chat/status response. */
export interface ChatStatus {
  in_progress: boolean;
  started_at?: string;
}

export const chat = {
  /**
   * Whether a UMP turn is currently in flight. Called on page mount (after
   * history load) to decide re-attach vs. interrupted-error rendering.
   */
  status(): Promise<ChatStatus> {
    return api.get('/chat/status');
  },
};

import { api } from '@chalie/shared';

/** Response from GET /auth/status. */
export interface AuthStatus {
  has_master_account: boolean;
  has_providers: boolean;
  has_session: boolean;
  vault_state: 'unlocked' | 'locked' | 'uninitialized';
}

/** Response from GET /ready — backend readiness probe. */
export interface ReadyStatus {
  ready: boolean;
}

// authStatus / readyCheck / heartbeat are probes, so they opt out of the
// client's redirect-on-401 (the gate inspects the 401 itself; /ready and
// /health are public). The authenticated settings calls below keep the default:
// a 401 there means the session expired → redirect to /login/.
const NO_REDIRECT = { redirectOnAuthError: false } as const;

/** A thread's model and thinking level, as the thread settings API answers.
 *  `provider_id` is null only when no model is configured at all; `turn_id` is
 *  the thread they belong to, null on a channel with no thread yet. */
export interface ThreadSettings {
  provider_id: number | null;
  thinking_level: string;
  turn_id: number | null;
}

interface ThreadSettingsEnvelope {
  success: true;
  result: ThreadSettings;
}

export const system = {
  /** GET /auth/status — readiness probe used by the auth gate in main.ts. */
  authStatus(): Promise<AuthStatus> {
    return api.get('/api/auth/status', NO_REDIRECT);
  },

  /**
   * GET /ready — readiness probe for the loading overlay. NEVER rejects: any
   * failure resolves to { ready: false } so the overlay's poll loop can retry.
   */
  async readyCheck(): Promise<ReadyStatus> {
    try {
      return { ready: Boolean((await api.get<ReadyStatus>('/ready', NO_REDIRECT))?.ready) };
    } catch {
      return { ready: false };
    }
  },

  /** POST /health — heartbeat with client telemetry; no auth required. */
  heartbeat(payload: Record<string, unknown>): Promise<{ status: string; version: string }> {
    return api.post('/health', payload, NO_REDIRECT);
  },

  /** GET /api/threads/settings/<turnId>?type= — the model and thinking level
   *  the thread's next message runs on; -1 names the channel's latest thread. */
  threadSettings(type: string, turnId: number): Promise<ThreadSettings> {
    return api.get<ThreadSettingsEnvelope>(`/api/threads/settings/${turnId}?type=${type}`).then((body) => body.result);
  },

  /** POST /api/threads/settings/<turnId>?type= — change a thread's model and
   *  thinking level, from its next message on; -1 names the channel's latest
   *  thread. */
  saveThreadSettings(type: string, turnId: number, settings: { provider_id: number; thinking_level: string }): Promise<ThreadSettings> {
    return api.post<ThreadSettingsEnvelope>(`/api/threads/settings/${turnId}?type=${type}`, settings).then((body) => body.result);
  },
};

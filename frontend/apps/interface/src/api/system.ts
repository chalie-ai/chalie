import { api } from '@chalie/shared';

/** Response from GET /auth/status. */
export interface AuthStatus {
  has_master_account: boolean;
  has_providers: boolean;
  has_session: boolean;
  vault_state: 'unlocked' | 'locked' | 'uninitialized';
}

/** Response from GET /system/context-usage. */
export interface ContextUsage {
  last_request_tokens: number | null;
  context_window: number | null;
}

/** Response from POST /system/update/apply. */
export interface UpdateApplyResult {
  ok: boolean;
  message: string;
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

  /** GET /system/context-usage — last request token count + context window. */
  contextUsage(): Promise<ContextUsage> {
    return api.get('/api/system/context-usage');
  },

  /** GET /system/settings/thinking_level_override */
  thinkingLevel(): Promise<{ key: string; value: string | null }> {
    return api.get('/api/system/settings/thinking_level_override');
  },

  /**
   * PUT /system/settings/thinking_level_override — empty value string deletes
   * the row (reverts to auto).
   */
  setThinkingLevel(value: string): Promise<Record<string, unknown>> {
    return api.put('/api/system/settings/thinking_level_override', { value });
  },

  /** POST /system/update/apply — apply an in-place installer update. */
  updateApply(tag: string): Promise<UpdateApplyResult> {
    return api.post('/api/system/update/apply', { tag });
  },
};

import { useApiClient } from '@chalie/shared';

/** Response from GET /auth/status. */
export interface AuthStatus {
  has_master_account: boolean;
  has_providers: boolean;
  has_session: boolean;
  vault_state: 'unlocked' | 'locked' | 'uninitialized';
  has_vision_provider: boolean;
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

export const system = {
  /**
   * GET /auth/status — check master-account / session / provider readiness.
   * Used by the auth gate in main.ts.
   */
  authStatus(): Promise<AuthStatus> {
    const api = useApiClient();
    return api.get('/auth/status');
  },

  /**
   * GET /ready — backend readiness probe for the loading overlay.
   *
   * NEVER rejects: any failure resolves to { ready: false } so the overlay's
   * poll loop can simply retry (port of legacy api.js readyCheck, lines 87-90).
   */
  async readyCheck(): Promise<ReadyStatus> {
    const api = useApiClient();
    try {
      const result = await api.get<ReadyStatus>('/ready');
      return { ready: Boolean(result?.ready) };
    } catch {
      return { ready: false };
    }
  },

  /**
   * POST /health — heartbeat with client telemetry payload.
   * No auth required; always returns { status: 'ok', version: string }.
   */
  heartbeat(payload: Record<string, unknown>): Promise<{ status: string; version: string }> {
    const api = useApiClient();
    return api.post('/health', payload);
  },

  /**
   * GET /system/context-usage — last request token count + context window.
   */
  contextUsage(): Promise<ContextUsage> {
    const api = useApiClient();
    return api.get('/system/context-usage');
  },

  /**
   * GET /system/settings/thinking_level_override
   */
  thinkingLevel(): Promise<{ key: string; value: string | null }> {
    const api = useApiClient();
    return api.get('/system/settings/thinking_level_override');
  },

  /**
   * PUT /system/settings/thinking_level_override
   * Empty value string deletes the row (reverts to auto).
   */
  setThinkingLevel(value: string): Promise<Record<string, unknown>> {
    const api = useApiClient();
    return api.put('/system/settings/thinking_level_override', { value });
  },

  /**
   * POST /system/update/apply — apply an in-place installer update.
   */
  updateApply(tag: string): Promise<UpdateApplyResult> {
    const api = useApiClient();
    return api.post('/system/update/apply', { tag });
  },
};

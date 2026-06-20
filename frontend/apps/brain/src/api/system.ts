import { api } from '@chalie/shared';

/** Response from GET /auth/status. */
export interface AuthStatus {
  has_master_account: boolean;
  has_providers: boolean;
  has_session: boolean;
  vault_state: 'unlocked' | 'locked' | 'uninitialized';
  has_vision_provider: boolean;
}

export const system = {
  /**
   * GET /auth/status — used by the router auth gate.
   * Opts out of the client's redirect-on-401: the gate inspects the result to
   * decide where to route, rather than treating a 401 as session expiry.
   */
  authStatus(): Promise<AuthStatus> {
    return api.get('/auth/status', { redirectOnAuthError: false });
  },

  /**
   * GET /auth/username — the master LOGIN username, embedded in the pairing QR
   * so the device's UnlockVault needs only a password. Cookie-session only
   * (backend rejects bearer callers), which the Brain dashboard always is.
   */
  username(): Promise<{ username: string }> {
    return api.get('/auth/username');
  },
};

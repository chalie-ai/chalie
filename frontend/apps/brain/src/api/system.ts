import { api } from '@chalie/shared';

/** Response from GET /auth/status. */
export interface AuthStatus {
  has_master_account: boolean;
  has_providers: boolean;
  has_session: boolean;
  vault_state: 'unlocked' | 'locked' | 'uninitialized';
  has_vision_provider: boolean;
  internal_dev: boolean;
}

export interface NetworkConfig {
  deployment_domain: string;
  ssl_enabled: boolean;
  ssl_cert_present: boolean;
}

export interface NetworkSaveResult {
  ssl_enabled: boolean;
  restarting: boolean;
}

export interface NetworkSavePayload {
  deployment_domain: string;
  ssl_enabled: boolean;
  ssl_cert?: File;
  ssl_key?: File;
}

export const system = {
  /**
   * GET /auth/status — used by the router auth gate.
   * Opts out of the client's redirect-on-401: the gate inspects the result to
   * decide where to route, rather than treating a 401 as session expiry.
   */
  authStatus(): Promise<AuthStatus> {
    return api.get('/api/auth/status', { redirectOnAuthError: false });
  },

  /**
   * GET /auth/username — the master LOGIN username, embedded in the pairing QR
   * so the device's UnlockVault needs only a password. Cookie-session only
   * (backend rejects bearer callers), which the Brain dashboard always is.
   */
  username(): Promise<{ username: string }> {
    return api.get('/api/auth/username');
  },

  getNetwork(): Promise<NetworkConfig> {
    return api.get('/api/system/network');
  },

  async saveNetwork(payload: NetworkSavePayload): Promise<NetworkSaveResult> {
    const form = new FormData();
    form.append('deployment_domain', payload.deployment_domain);
    form.append('ssl_enabled', String(payload.ssl_enabled));
    if (payload.ssl_cert) form.append('ssl_cert', payload.ssl_cert);
    if (payload.ssl_key) form.append('ssl_key', payload.ssl_key);
    return api.putForm<NetworkSaveResult>('/api/system/network', form);
  },
};

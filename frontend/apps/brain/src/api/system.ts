import { useApiClient } from '@chalie/shared';

/** Response from GET /auth/status. */
export interface AuthStatus {
  has_master_account: boolean;
  has_providers: boolean;
  has_session: boolean;
  vault_state: 'unlocked' | 'locked' | 'uninitialized';
  has_vision_provider: boolean;
}

export const system = {
  /** GET /auth/status — used by the router auth gate. */
  authStatus(): Promise<AuthStatus> {
    const api = useApiClient();
    return api.get('/auth/status');
  },
};

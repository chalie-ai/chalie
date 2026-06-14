import { useApiClient, AuthError, HttpError } from '@chalie/shared';

/** Response from GET /auth/status. */
export interface AuthStatus {
  has_master_account: boolean;
  has_session: boolean;
  has_providers: boolean;
}

export const auth = {
  /** GET /auth/status — check master-account / session / provider readiness. */
  authStatus(): Promise<AuthStatus> {
    const api = useApiClient();
    return api.get<AuthStatus>('/auth/status');
  },

  /** POST /auth/login — authenticate with username + password. */
  login(username: string, password: string): Promise<void> {
    const api = useApiClient();
    return api.post<void>('/auth/login', { username, password });
  },

  // register() and setVoiceEnabled() back the on-boarding multi-page entry,
  // not the login page — they share this auth module deliberately.

  /** POST /auth/register — create the master account. */
  register(username: string, password: string): Promise<void> {
    const api = useApiClient();
    return api.post<void>('/auth/register', { username, password });
  },

  /** PUT /api/voice-settings — enable or disable voice. */
  setVoiceEnabled(enabled: boolean): Promise<void> {
    const api = useApiClient();
    return api.put<void>('/api/voice-settings', { enabled });
  },
};

export { AuthError, HttpError };

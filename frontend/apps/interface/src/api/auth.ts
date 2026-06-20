import { api, AuthError, HttpError } from '@chalie/shared';

/** Response from GET /auth/status. */
export interface AuthStatus {
  has_master_account: boolean;
  has_session: boolean;
  has_providers: boolean;
}

// Auth/onboarding endpoints: a 401 here is bad credentials / no session yet /
// a routing probe, never "session expired mid-app" — so they opt out of the
// client's redirect-on-401 and let the caller read the error.
const NO_REDIRECT = { redirectOnAuthError: false } as const;

export const auth = {
  /** GET /auth/status — master-account / session / provider readiness. */
  authStatus(): Promise<AuthStatus> {
    return api.get<AuthStatus>('/auth/status', NO_REDIRECT);
  },

  /** POST /auth/login */
  login(username: string, password: string): Promise<void> {
    return api.post<void>('/auth/login', { username, password }, NO_REDIRECT);
  },

  // register() and setVoiceEnabled() back the onboarding multi-page entry, not
  // the login page — they share this auth module deliberately.

  /** POST /auth/register — create the master account. */
  register(username: string, password: string): Promise<void> {
    return api.post<void>('/auth/register', { username, password }, NO_REDIRECT);
  },

  /** PUT /api/voice-settings */
  setVoiceEnabled(enabled: boolean): Promise<void> {
    return api.put<void>('/api/voice-settings', { enabled }, NO_REDIRECT);
  },
};

export { AuthError, HttpError };

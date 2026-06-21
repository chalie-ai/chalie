import { api, AuthError, HttpError } from '@chalie/shared';

/** POST /auth/login response (existing backend contract — do not rename keys). */
export interface LoginResult {
  ok?: boolean;
  vault_state?: 'unlocked' | 'locked' | 'uninitialized';
  error?: string;
  onboarding_required?: boolean;
}

// Auth/onboarding endpoints: a 401 here is bad credentials / no session yet /
// a routing probe, never "session expired mid-app" — so they opt out of the
// client's redirect-on-401 and let the caller read the error.
const NO_REDIRECT = { redirectOnAuthError: false } as const;

export const auth = {
  /** POST /auth/login — unseal the vault with the stored login username + typed
   *  password. Existing endpoint; called here only from the UnlockVault overlay. */
  login(username: string, password: string): Promise<LoginResult> {
    return api.post<LoginResult>('/auth/login', { username, password }, NO_REDIRECT);
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

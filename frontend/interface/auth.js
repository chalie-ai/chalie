/**
 * Authentication — session checking and login redirect.
 */
export class Auth {
  constructor() {}

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Check /auth/status.
   * - No account  → redirect to /on-boarding/
   * - No session  → redirect to /login/
   * - Authenticated → resolve immediately
   */
  async checkSession() {
    try {
      const r = await fetch('/auth/status', { credentials: 'same-origin' });
      if (r.ok) {
        const data = await r.json();
        if (!data.has_master_account) {
          window.location.replace('/on-boarding/');
          return;
        }
        if (!data.has_session) {
          // Standalone login page — macOS Keychain/autofill needs the
          // form visible on page load to offer password fill.
          window.location.replace('/login/');
          return;
        }
      } else {
        window.location.replace('/on-boarding/');
        return;
      }
    } catch (_) { /* backend unreachable — let the app handle it normally */ }
  }

  /**
   * Called by other modules on 401.
   */
  handleAuthFailure() {
    window.location.replace('/login/');
  }
}

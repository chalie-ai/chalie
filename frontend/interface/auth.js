/**
 * Authentication — session checking and login dialog.
 */
export class Auth {
  /**
   * @param {object} options
   * @param {() => string} options.getHost     — returns the backend host URL
   * @param {() => void}  [options.onBeforeShow] — called before the login dialog opens
   *   (used by ChalieApp to clear the task-strip poll interval so it stops firing 401s)
   */
  constructor({ getHost, onBeforeShow }) {
    this._getHost = getHost;
    this._onBeforeShow = onBeforeShow || null;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Check /auth/status.
   * - No account  → redirect to /on-boarding/
   * - No session  → show login dialog, then return
   * - Authenticated → resolve immediately
   * Returns a Promise that resolves when auth is ready (or after a successful login).
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
          await this._showLoginDialog();
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
   * Guards against double-open (e.g. two concurrent 401s firing at once).
   */
  handleAuthFailure() {
    if (this._onBeforeShow) this._onBeforeShow();
    const dialog = document.getElementById('loginDialog');
    if (dialog?.open) return;
    this._showLoginDialog();
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  _showLoginDialog() {
    return new Promise((resolve) => {
      const dialog     = document.getElementById('loginDialog');
      const submitBtn  = document.getElementById('loginSubmitBtn');
      const statusEl   = document.getElementById('loginStatus');
      const usernameEl = document.getElementById('loginUsername');
      const passwordEl = document.getElementById('loginPassword');

      statusEl.textContent = '';
      statusEl.className = 'api-key-dialog__status';

      const doLogin = async () => {
        const username = usernameEl.value.trim();
        const password = passwordEl.value;
        if (!username || !password) {
          statusEl.textContent = 'Username and password required.';
          statusEl.className = 'api-key-dialog__status api-key-dialog__status--error';
          return;
        }
        submitBtn.disabled = true;
        submitBtn.textContent = 'Logging in...';
        try {
          const res = await fetch(this._getHost() + '/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'same-origin',
            body: JSON.stringify({ username, password }),
          });
          if (res.ok) {
            dialog.close();
            resolve();
            // Reload the page — the first checkSession() returned early (before any _initXxx()
            // calls) so the app shell was never wired up. A clean reload is more reliable than
            // trying to re-bootstrap in-place with leftover timers and partial state.
            window.location.reload();
          } else {
            statusEl.textContent = res.status === 401 ? 'Invalid credentials.' : 'Login failed.';
            statusEl.className = 'api-key-dialog__status api-key-dialog__status--error';
            submitBtn.disabled = false;
            submitBtn.textContent = 'Login';
          }
        } catch {
          statusEl.textContent = 'Network error.';
          statusEl.className = 'api-key-dialog__status api-key-dialog__status--error';
          submitBtn.disabled = false;
          submitBtn.textContent = 'Login';
        }
      };

      const form = document.getElementById('loginForm');
      form.onsubmit = (e) => { e.preventDefault(); doLogin(); };

      dialog.showModal();
    });
  }
}

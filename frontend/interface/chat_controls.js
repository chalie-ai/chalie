/**
 * ChatControls — the two composer controls under the input dock:
 *   • thinking-level override dropdown (left): auto / medium / high, persisted
 *     in the backend `settings` table so it applies to every MessageProcessor
 *     and shows the correct value on refresh / other devices.
 *   • context-size indicator (right): `X.X/YY.Yk` = last chat request tokens /
 *     context window. Re-fetched on page load and on every inbound WS message.
 */
const THINKING_KEY = 'thinking_level_override';
const LEVELS = ['auto', 'medium', 'high'];

export class ChatControls {
  /** @param {{ api: object, ws: object }} deps */
  constructor({ api, ws }) {
    this._api = api;
    this._ws = ws;
    this._current = 'auto';
    this._refreshing = false;   // a context fetch is in flight
    this._refreshQueued = false; // ≥1 WS message arrived during that fetch
  }

  init() {
    this._trigger = document.getElementById('thinkingTrigger');
    this._menu = document.getElementById('thinkingMenu');
    this._label = document.getElementById('thinkingLabel');
    this._indicator = document.getElementById('contextIndicator');
    if (!this._trigger || !this._menu || !this._label || !this._indicator) return;

    this._wireDropdown();
    this._loadThinking();

    // Context indicator: load once, then refresh on every WS message.
    this.refreshContext();
    this._ws.onAny(() => this.refreshContext());
  }

  // ── Thinking-level dropdown ───────────────────────────────────────────────

  _wireDropdown() {
    this._trigger.addEventListener('click', (e) => {
      e.stopPropagation();
      const open = this._menu.classList.toggle('hidden') === false;
      this._trigger.setAttribute('aria-expanded', String(open));
    });

    this._menu.querySelectorAll('[data-level]').forEach((btn) => {
      btn.addEventListener('click', () => this._select(btn.dataset.level));
    });

    // Click-outside closes the menu.
    document.addEventListener('click', (e) => {
      if (!this._menu.contains(e.target) && e.target !== this._trigger) {
        this._closeMenu();
      }
    });
  }

  _closeMenu() {
    this._menu.classList.add('hidden');
    this._trigger.setAttribute('aria-expanded', 'false');
  }

  async _loadThinking() {
    try {
      const data = await this._api.getSetting(THINKING_KEY);
      this._setLabel(LEVELS.includes(data?.value) ? data.value : 'auto');
    } catch {
      this._setLabel('auto');
    }
  }

  async _select(level) {
    this._closeMenu();
    if (!LEVELS.includes(level)) return;
    const previous = this._current;
    this._setLabel(level);
    try {
      // Empty value clears the override server-side → 'auto' (gate decides).
      await this._api.setSetting(THINKING_KEY, level === 'auto' ? '' : level);
    } catch {
      this._setLabel(previous); // persist failed — don't leave a label that lies
    }
  }

  _setLabel(level) {
    this._current = level;
    this._label.textContent = level.charAt(0).toUpperCase() + level.slice(1);
    this._menu.querySelectorAll('[data-level]').forEach((b) => {
      b.classList.toggle('active', b.dataset.level === level);
    });
  }

  // ── Context-size indicator ────────────────────────────────────────────────

  async refreshContext() {
    // Coalesce: a WS burst (many act_* events per turn) must not open a fetch
    // per message. While one is in flight, record that more arrived and run
    // exactly one trailing fetch when it settles — so the final value after a
    // burst is always painted, without flooding the endpoint.
    if (this._refreshing) {
      this._refreshQueued = true;
      return;
    }
    this._refreshing = true;
    try {
      let data;
      try {
        data = await this._api.getContextUsage();
      } catch {
        return; // transient/auth miss — leave the last painted value in place
      }
      const tokens = data?.last_request_tokens;
      const contextWindow = data?.context_window;
      if (tokens == null || contextWindow == null) return;
      this._indicator.textContent =
        `${(tokens / 1000).toFixed(1)}/${(contextWindow / 1000).toFixed(1)}k`;
      this._indicator.classList.remove('hidden');
    } finally {
      this._refreshing = false;
      if (this._refreshQueued) {
        this._refreshQueued = false;
        this.refreshContext(); // trailing run for messages that arrived mid-fetch
      }
    }
  }
}

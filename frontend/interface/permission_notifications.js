/**
 * Permission request notifications — slide-up cards for backend permission_request events.
 *
 * Renders a stack of cards above the input dock. Each card shows:
 *   - Human-readable action label
 *   - Key/value params preview
 *   - Allow / Deny buttons
 *   - 30-second countdown progress bar that auto-denies on expiry
 *
 * Multiple concurrent requests stack vertically, newest on top.
 */

const ACTION_LABELS = {
  'email.manage':          'Manage email',
  'email.draft':           'Draft email',
  'email.send':            'Send email',
  'calendar.update_event': 'Update calendar event',
  'calendar.create_event': 'Create calendar event',
  'code_eval':             'Execute code',
  'browser.interact':      'Interact with webpage',
  'document.delete':       'Delete document',
  'list.delete':           'Delete list',
  'memory.forget':         'Forget memory',
  'schedule.create':       'Create schedule',
  'schedule.cancel':       'Cancel schedule',
};

const TIMEOUT_MS = 30_000;

/**
 * Convert an action_id to a readable label.
 * Falls back to formatting the id itself (dots/underscores → spaces, title case).
 */
function actionLabel(actionId) {
  if (ACTION_LABELS[actionId]) return ACTION_LABELS[actionId];
  return actionId
    .replace(/[._]/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Render params_preview as a small key/value block.
 * Returns null if params is empty / not an object.
 */
function buildParamsEl(params) {
  if (!params || typeof params !== 'object' || Array.isArray(params)) return null;
  const entries = Object.entries(params);
  if (!entries.length) return null;

  const dl = document.createElement('dl');
  dl.className = 'perm-card__params';

  for (const [k, v] of entries) {
    const dt = document.createElement('dt');
    dt.className = 'perm-card__param-key';
    dt.textContent = k;

    const dd = document.createElement('dd');
    dd.className = 'perm-card__param-val';
    // Truncate very long values so the card stays compact
    const str = typeof v === 'string' ? v : JSON.stringify(v);
    dd.textContent = str.length > 80 ? str.slice(0, 77) + '…' : str;

    dl.appendChild(dt);
    dl.appendChild(dd);
  }

  return dl;
}

export class PermissionNotifications {
  /**
   * @param {{ ws: import('./ws.js').WSClient }} options
   */
  constructor({ ws }) {
    this._ws = ws;
    /** @type {Map<string, { el: HTMLElement, timer: number, interval: number }>} */
    this._active = new Map();
    this._container = null;
  }

  /**
   * Call once after the DOM is ready to mount the container.
   */
  init() {
    this._container = document.createElement('div');
    this._container.className = 'perm-stack';
    this._container.setAttribute('aria-live', 'assertive');
    this._container.setAttribute('aria-label', 'Permission requests');
    document.body.appendChild(this._container);
  }

  /**
   * Handle a permission_request event from the backend.
   * @param {{ request_id: string, action_id: string, context?: string, params_preview?: object }} data
   */
  handleRequest(data) {
    const { request_id, action_id, params_preview } = data;
    if (!request_id || !action_id) return;

    // Deduplicate — if already shown, ignore
    if (this._active.has(request_id)) return;

    const card = this._buildCard(request_id, action_id, params_preview);
    this._container.prepend(card);  // newest on top

    const startTime = Date.now();

    // Countdown: update the progress bar every 100ms
    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const remaining = Math.max(0, TIMEOUT_MS - elapsed);
      const pct = (remaining / TIMEOUT_MS) * 100;
      const bar = card.querySelector('.perm-card__progress-fill');
      if (bar) bar.style.width = pct + '%';
    }, 100);

    // Auto-deny after timeout
    const timer = setTimeout(() => {
      this._respond(request_id, false, true);
    }, TIMEOUT_MS);

    this._active.set(request_id, { el: card, timer, interval });

    // Trigger slide-in animation
    requestAnimationFrame(() => card.classList.add('perm-card--visible'));
  }

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  _buildCard(requestId, actionId, paramsPreview) {
    const card = document.createElement('div');
    card.className = 'perm-card';
    card.dataset.requestId = requestId;
    card.setAttribute('role', 'dialog');
    card.setAttribute('aria-label', 'Permission request');

    // Header
    const header = document.createElement('div');
    header.className = 'perm-card__header';

    const icon = document.createElement('span');
    icon.className = 'perm-card__icon';
    icon.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
      aria-hidden="true">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="8" x2="12" y2="12"/>
      <line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>`;

    const title = document.createElement('p');
    title.className = 'perm-card__title';

    const intro = document.createElement('span');
    intro.className = 'perm-card__intro';
    intro.textContent = 'Chalie wants to: ';

    const action = document.createElement('strong');
    action.className = 'perm-card__action';
    action.textContent = actionLabel(actionId);

    title.appendChild(intro);
    title.appendChild(action);

    header.appendChild(icon);
    header.appendChild(title);
    card.appendChild(header);

    // Params preview
    const paramsEl = buildParamsEl(paramsPreview);
    if (paramsEl) {
      card.appendChild(paramsEl);
    }

    // Actions
    const actions = document.createElement('div');
    actions.className = 'perm-card__actions';

    const allowBtn = document.createElement('button');
    allowBtn.className = 'perm-card__btn perm-card__btn--allow';
    allowBtn.textContent = 'Allow';
    allowBtn.addEventListener('click', () => this._respond(requestId, true, false));

    const denyBtn = document.createElement('button');
    denyBtn.className = 'perm-card__btn perm-card__btn--deny';
    denyBtn.textContent = 'Deny';
    denyBtn.addEventListener('click', () => this._respond(requestId, false, false));

    actions.appendChild(denyBtn);
    actions.appendChild(allowBtn);
    card.appendChild(actions);

    // Progress bar (countdown depletes left to right)
    const progressTrack = document.createElement('div');
    progressTrack.className = 'perm-card__progress';

    const progressFill = document.createElement('div');
    progressFill.className = 'perm-card__progress-fill';
    progressFill.style.width = '100%';

    progressTrack.appendChild(progressFill);
    card.appendChild(progressTrack);

    return card;
  }

  /**
   * Send a permission response and dismiss the card.
   * @param {string} requestId
   * @param {boolean} approved
   * @param {boolean} timedOut  — adds a brief "timed out" flash before removal
   */
  _respond(requestId, approved, timedOut) {
    const entry = this._active.get(requestId);
    if (!entry) return;

    clearTimeout(entry.timer);
    clearInterval(entry.interval);
    this._active.delete(requestId);

    // Send the response via WebSocket
    if (this._ws.isConnected) {
      this._ws.sendProtocol({ type: 'permission_response', request_id: requestId, approved });
    }

    // Animate out
    const { el } = entry;
    if (timedOut) {
      el.classList.add('perm-card--timeout');
      setTimeout(() => this._removeCard(el), 600);
    } else {
      this._removeCard(el);
    }
  }

  _removeCard(el) {
    el.classList.remove('perm-card--visible');
    el.classList.add('perm-card--leaving');
    // Remove from DOM after transition completes
    el.addEventListener('transitionend', () => el.remove(), { once: true });
    // Safety fallback in case transitionend doesn't fire
    setTimeout(() => el.remove(), 400);
  }
}

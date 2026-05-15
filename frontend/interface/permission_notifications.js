/**
 * Permission request notifications — slide-up cards for backend permission_request events.
 *
 * Renders a stack of cards above the input dock. Each card shows:
 *   - Human-readable action label
 *   - One-line description of what the action does
 *   - Allow / Deny buttons
 *   - No auto-deny — waits indefinitely for user response
 *
 * Multiple concurrent requests stack vertically, newest on top.
 */

const ACTION_LABELS = {
  'email.read':            'Read Email',
  'email.search':          'Search Email',
  'email.manage':          'Manage Email',
  'email.draft':           'Draft Email',
  'email.send':            'Send Email',
  'email.reply':           'Reply to Email',
  'email.forward':         'Forward Email',
  'calendar.list_events':  'List Calendar Events',
  'calendar.get_event':    'Read Calendar Event',
  'calendar.update_event': 'Update Calendar Event',
  'calendar.create_event': 'Create Calendar Event',
  'code_eval':             'Execute Code',
  'browser.render':        'Read Webpage',
  'browser.interact':      'Interact with Webpage',
  'browser.screenshot':    'Screenshot Webpage',
  'browser.monitor':       'Monitor Webpage',
  'document.search':       'Search Documents',
  'document.list':         'List Documents',
  'document.view':         'View Document',
  'document.create':       'Create Document',
  'document.delete':       'Delete Document',
  'document.restore':      'Restore Document',
  'list.delete':           'Delete List',
  'memory.store':          'Store Memory',
  'memory.recall':         'Recall Memory',
  'memory.forget':         'Forget Memory',
  'memory.reflect':        'Reflect on Memory',
  'schedule.create':       'Create Schedule',
  'schedule.cancel':       'Cancel Schedule',
  'schedule.list':         'List Schedules',
  'schedule.search':       'Search Schedules',
  'contacts':              'Access Contacts',
  'news':                  'Fetch News',
  'search':                'Web Search',
  'weather':               'Check Weather',
  'timer':                 'Set Timer',
};

// No auto-deny — the card stays until the user explicitly clicks Allow or Deny.
// The backend polls indefinitely (1-hour safety net).

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

export class PermissionNotifications {
  constructor() {
    /** @type {Map<string, { el: HTMLElement }>} */
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
   * @param {{ request_id: string, action_id: string, description?: string, context?: string }} data
   */
  handleRequest(data) {
    const { request_id, action_id, description } = data;
    if (!request_id || !action_id) return;

    // Deduplicate — if already shown, ignore
    if (this._active.has(request_id)) return;

    const card = this._buildCard(request_id, action_id, description);
    this._container.prepend(card);  // newest on top

    this._active.set(request_id, { el: card });

    // Trigger slide-in animation
    requestAnimationFrame(() => card.classList.add('perm-card--visible'));
  }

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  _buildCard(requestId, actionId, description) {
    const card = document.createElement('div');
    card.className = 'perm-card';
    card.dataset.requestId = requestId;
    card.setAttribute('role', 'dialog');
    card.setAttribute('aria-label', 'Permission request');

    // Body wrapper — provides padding around content
    const body = document.createElement('div');
    body.className = 'perm-card__body';

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
    title.textContent = actionLabel(actionId);

    header.appendChild(icon);
    header.appendChild(title);
    body.appendChild(header);

    // Description — one-line summary of what the action does
    if (description) {
      const desc = document.createElement('p');
      desc.className = 'perm-card__desc';
      desc.textContent = description;
      body.appendChild(desc);
    }

    // Actions
    const actions = document.createElement('div');
    actions.className = 'perm-card__actions';

    const allowBtn = document.createElement('button');
    allowBtn.className = 'perm-card__btn perm-card__btn--allow';
    allowBtn.textContent = 'Allow';
    allowBtn.addEventListener('click', () => this._respond(requestId, true));

    const denyBtn = document.createElement('button');
    denyBtn.className = 'perm-card__btn perm-card__btn--deny';
    denyBtn.textContent = 'Deny';
    denyBtn.addEventListener('click', () => this._respond(requestId, false));

    actions.appendChild(denyBtn);
    actions.appendChild(allowBtn);
    body.appendChild(actions);

    card.appendChild(body);

    return card;
  }

  /**
   * Send a permission response and dismiss the card.
   * @param {string} requestId
   * @param {boolean} approved
   */
  _respond(requestId, approved) {
    const entry = this._active.get(requestId);
    if (!entry) return;

    this._active.delete(requestId);

    // Send via REST — the WebSocket receive loop is blocked during chat processing
    fetch('/api/policies/respond', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_id: requestId, approved }),
    }).catch((e) => console.warn('[PermNotif] respond failed:', e));

    this._removeCard(entry.el);
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

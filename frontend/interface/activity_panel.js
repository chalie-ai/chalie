/**
 * Activity Panel — slide-out log of recent Chalie activity.
 *
 * Displays a reverse-chronological feed of memory, goal, proactive, and task
 * events. Follows the same open/hidden CSS-class toggle pattern used by
 * AppsPanel. Panel slides in from the right on toggle and stacks below the
 * Apps panel in z-index order (240 vs 250).
 *
 * DOM dependencies (must exist in index.html):
 *   #activityBtn         — title-bar toggle button
 *   #activityPanel       — the <aside> panel element (class: activity-panel)
 *   #activityPanelList   — container for entry elements
 *   #closeActivityPanel  — close button inside the panel
 */
export class ActivityPanel {
  /**
   * Initialise state. Call {@link init} after the DOM is ready.
   */
  constructor() {
    /** @type {boolean} Whether the panel is currently open. */
    this._open = false;

    /**
     * Bounded ring-buffer of activity entries (newest first), capped at 100.
     * @type {Array<{id: number, type: string, message: string, timestamp: Date}>}
     */
    this._entries = [];

    /** Monotonically incrementing entry id used as a DOM key. */
    this._nextId = 1;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Wire DOM event listeners and pre-populate the panel with mock entries.
   * Must be called once after the document is ready.
   *
   * @returns {void}
   */
  init() {
    const toggleBtn = document.getElementById('activityBtn');
    const closeBtn  = document.getElementById('closeActivityPanel');

    toggleBtn?.addEventListener('click', () => this._toggle());
    closeBtn?.addEventListener('click',  () => this._toggle(false));

    this._seedMockEntries();
  }

  /**
   * Add a new activity entry and prepend it to the visible list.
   *
   * Entries exceeding 100 are dropped from the tail (oldest).
   *
   * @param {'memory'|'goal'|'proactive'|'task'} type       Entry category.
   * @param {string}                              message    Human-readable description.
   * @param {Date}                                [timestamp=new Date()] Event time.
   * @returns {void}
   */
  addEntry(type, message, timestamp = new Date()) {
    const entry = { id: this._nextId++, type, message, timestamp };

    this._entries.unshift(entry);

    // Cap ring-buffer at 100 entries.
    if (this._entries.length > 100) {
      this._entries.length = 100;
    }

    this._prependEntryElement(entry);
  }

  // ---------------------------------------------------------------------------
  // Private — panel toggle
  // ---------------------------------------------------------------------------

  /**
   * Open or close the activity panel.
   *
   * @param {boolean} [force] When provided, explicitly sets the open state
   *   (`true` = open, `false` = close). When omitted the state is flipped.
   * @returns {void}
   */
  _toggle(force) {
    const panel     = document.getElementById('activityPanel');
    const toggleBtn = document.getElementById('activityBtn');
    if (!panel) return;

    const shouldOpen = force !== undefined ? force : !this._open;
    this._open = shouldOpen;

    // Reflect active state on the toggle button.
    toggleBtn?.classList.toggle('active', shouldOpen);

    if (shouldOpen) {
      panel.classList.remove('hidden');
      requestAnimationFrame(() => panel.classList.add('open'));
    } else {
      panel.classList.remove('open');
      panel.addEventListener('transitionend', () => {
        if (!this._open) panel.classList.add('hidden');
      }, { once: true });
    }
  }

  // ---------------------------------------------------------------------------
  // Private — rendering
  // ---------------------------------------------------------------------------

  /**
   * Create a DOM element for a single entry and prepend it to the list container.
   *
   * The element receives the BEM classes `activity-entry` and
   * `activity-entry--{type}`. The left-border accent colour is applied as an
   * inline CSS variable so the stylesheet rule stays generic.
   *
   * @param {{id: number, type: string, message: string, timestamp: Date}} entry
   * @returns {void}
   */
  _prependEntryElement(entry) {
    const list = document.getElementById('activityPanelList');
    if (!list) return;

    const el = this._buildEntryElement(entry);

    // Insert before the first child so newest appears at the top.
    list.insertBefore(el, list.firstChild);
  }

  /**
   * Build and return the DOM element for a single activity entry.
   *
   * @param {{id: number, type: string, message: string, timestamp: Date}} entry
   * @returns {HTMLDivElement}
   */
  _buildEntryElement(entry) {
    const BORDER_COLORS = {
      memory:    '#00F0FF',
      goal:      '#8A5CFF',
      proactive: '#FF00AA',
      task:      '#888888',
    };

    const color = BORDER_COLORS[entry.type] ?? '#888888';

    const el = document.createElement('div');
    el.className = `activity-entry activity-entry--${entry.type}`;
    el.dataset.entryId = entry.id;
    el.style.setProperty('--entry-accent', color);

    const body = document.createElement('div');
    body.className = 'activity-entry__body';

    const msg = document.createElement('span');
    msg.className = 'activity-entry__message';
    msg.textContent = entry.message;
    body.appendChild(msg);

    const ts = document.createElement('span');
    ts.className = 'activity-entry__time';
    ts.textContent = this._relativeTime(entry.timestamp);
    ts.title = entry.timestamp.toLocaleString();
    body.appendChild(ts);

    el.appendChild(body);
    return el;
  }

  /**
   * Render a human-friendly relative time string (e.g. "2 min ago").
   *
   * @param {Date} date  The reference date to compare against now.
   * @returns {string}   A short relative-time label.
   */
  _relativeTime(date) {
    const diffMs  = Date.now() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);

    if (diffSec < 5)   return 'just now';
    if (diffSec < 60)  return `${diffSec}s ago`;

    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60)  return `${diffMin} min ago`;

    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24)   return `${diffHr}h ago`;

    const diffDay = Math.floor(diffHr / 24);
    return `${diffDay}d ago`;
  }

  // ---------------------------------------------------------------------------
  // Private — mock data
  // ---------------------------------------------------------------------------

  /**
   * Pre-populate the panel with 10 representative mock entries so the panel
   * is not empty on first open during development.
   *
   * @returns {void}
   */
  _seedMockEntries() {
    const now = Date.now();

    /** @type {Array<{type: string, message: string, offsetMs: number}>} */
    const seeds = [
      { type: 'memory',    message: 'Recalled 4 relevant episodes for context',           offsetMs:   0 },
      { type: 'goal',      message: 'Goal "Learn Spanish" progressed — evidence added',   offsetMs:   2 * 60 * 1000 },
      { type: 'proactive', message: 'Suppressed nudge: social cost too high (0.82)',       offsetMs:   5 * 60 * 1000 },
      { type: 'task',      message: 'Background task "summarise_notes" completed',         offsetMs:   9 * 60 * 1000 },
      { type: 'memory',    message: 'Consolidated 12 short-term memories into episodes',  offsetMs:  18 * 60 * 1000 },
      { type: 'goal',      message: 'Goal "Read more" created from conversation',          offsetMs:  32 * 60 * 1000 },
      { type: 'proactive', message: 'Triggered proactive thought — confidence 0.74',      offsetMs:  47 * 60 * 1000 },
      { type: 'task',      message: 'Curiosity thread "quantum computing" started',        offsetMs:  65 * 60 * 1000 },
      { type: 'memory',    message: 'Surprising connection found: music ↔ mathematics',   offsetMs:  90 * 60 * 1000 },
      { type: 'goal',      message: 'Goal "Exercise daily" decay threshold reached',       offsetMs: 120 * 60 * 1000 },
    ];

    // Add in reverse order so the most-recent entry (index 0) ends up on top
    // after insertBefore-to-firstChild in _prependEntryElement.
    for (const seed of seeds) {
      this.addEntry(seed.type, seed.message, new Date(now - seed.offsetMs));
    }
  }
}

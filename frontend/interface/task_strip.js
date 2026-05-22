import { escHtml, relativeTime, lsGet, lsSet } from './utils.js';

/**
 * Task drawer — displays active subagents and pending reminders in a slideout
 * panel that slides in from the right edge of the viewport.
 *
 * Subagents are tracked via WS push events (subagent_start / subagent_end)
 * rather than polling. Reminders are fetched every 60 seconds.
 */
export class TaskStrip {
  constructor({ api }) {
    this._api = api;
    this._taskStripInterval = null;
    this._onAuthFailureCb = null;
    /** @type {Map<string, {agent_type: string, description: string}>} */
    this._activeSubagents = new Map();
    /** @type {Array} Last fetched reminders, used for re-render on subagent events. */
    this._cachedReminders = [];
  }

  /**
   * Bind trigger / close controls and start 60s polling interval.
   */
  init() {
    const triggerBtn = document.getElementById('taskDrawerBtn');
    const closeBtn   = document.getElementById('taskDrawerClose');
    const scrim      = document.getElementById('taskDrawerScrim');
    if (!triggerBtn || !scrim) return;

    triggerBtn.addEventListener('click', () => this._openDrawer());
    if (closeBtn) closeBtn.addEventListener('click', () => this._closeDrawer());
    scrim.addEventListener('click',      () => this._closeDrawer());

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') this._closeDrawer();
    });

    this.loadActiveTasks();
    this._taskStripInterval = setInterval(() => this.loadActiveTasks(), 60_000);
  }

  /**
   * Register a callback for 401 auth failures.
   * @param {Function} cb
   */
  onAuthFailure(cb) {
    this._onAuthFailureCb = cb;
  }

  /**
   * Handle a subagent_start WS event: add the subagent to the active map
   * and re-render the drawer.
   * @param {{ sub_id: string, agent_type: string, description: string }} data
   */
  onSubagentStart(data) {
    this._activeSubagents.set(data.sub_id, {
      agent_type: data.agent_type,
      description: data.description,
    });
    this._renderFromCache();
  }

  /**
   * Handle a subagent_end WS event: remove the subagent from the active map
   * and re-render the drawer.
   * @param {{ sub_id: string, status: string }} data
   */
  onSubagentEnd(data) {
    this._activeSubagents.delete(data.sub_id);
    this._renderFromCache();
  }

  /**
   * Fetch and render active tasks and pending reminders.
   * Public — called on init and from the event router.
   */
  async loadActiveTasks() {
    try {
      const schedData = await this._api._get('/scheduler?status=pending').catch((e) => { if (e?.message === 'AUTH') throw e; return {}; });
      const reminders = (schedData.items || []).filter(
        r => r.status === 'pending' && r.due_at
      );
      this._cachedReminders = reminders;
      this._render(reminders);
    } catch (e) {
      if (e?.message === 'AUTH') {
        this._onAuthFailureCb?.();
      }
      // Other errors: silently fail — drawer is supplementary
    }
  }

  /**
   * Clear the polling interval.
   */
  destroy() {
    clearInterval(this._taskStripInterval);
    this._taskStripInterval = null;
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  _openDrawer() {
    const drawer = document.getElementById('taskDrawer');
    const scrim  = document.getElementById('taskDrawerScrim');
    if (!drawer) return;

    scrim.classList.remove('hidden');
    drawer.classList.remove('hidden');
    requestAnimationFrame(() => {
      drawer.classList.add('open');
    });
  }

  _closeDrawer() {
    const drawer = document.getElementById('taskDrawer');
    const scrim  = document.getElementById('taskDrawerScrim');
    if (!drawer) return;

    drawer.classList.remove('open');
    scrim.classList.add('hidden');
    drawer.addEventListener('transitionend', () => {
      if (!drawer.classList.contains('open')) {
        drawer.classList.add('hidden');
      }
    }, { once: true });
  }

  /**
   * Re-render using the last fetched reminders and current subagent map.
   * Called on subagent_start / subagent_end events without a fresh API call.
   */
  _renderFromCache() {
    this._render(this._cachedReminders);
  }

  _render(reminders = []) {
    const drawer   = document.getElementById('taskDrawer');
    const list     = document.getElementById('taskDrawerList');
    const trigger  = document.getElementById('taskDrawerBtn');
    if (!drawer || !list) return;

    const totalCount = reminders.length + this._activeSubagents.size;

    // Show trigger when there is anything to display.
    trigger?.classList.toggle('hidden', totalCount === 0);

    // Auto-close the drawer when all items are gone.
    if (totalCount === 0) {
      this._closeDrawer();
      list.innerHTML = '';
      return;
    }

    let html = '';

    // Render pending reminders (top section).
    for (const r of reminders) {
      const msg = r.message || '';
      const due = r.due_at ? relativeTime(r.due_at) : '';

      html += `<div class="task-drawer__item task-drawer__item--reminder">
        <span class="task-drawer__msg">${escHtml(msg)}</span>
        ${due ? `<span class="task-drawer__due">${escHtml(due)}</span>` : ''}
      </div>`;
    }

    // Render active subagents (bottom section).
    if (this._activeSubagents.size > 0) {
      if (reminders.length > 0) {
        html += '<div class="task-drawer__section-divider"></div>';
      }
      for (const [subId, info] of this._activeSubagents) {
        html += `<div class="task-drawer__item task-drawer__item--subagent">
          <div class="task-drawer__subagent-row">
            <span class="task-drawer__msg">${escHtml(info.description)}</span>
            <span class="task-drawer__badge">${escHtml(info.agent_type)}</span>
          </div>
          <button class="task-drawer__stop-btn btn-icon"
                  aria-label="Stop subagent"
                  data-sub-id="${escHtml(subId)}">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
              <rect x="1" y="1" width="10" height="10" rx="2"/>
            </svg>
          </button>
        </div>`;
      }
    }

    // First-time hint.
    if (!lsGet('task_strip_hint_shown')) {
      lsSet('task_strip_hint_shown', '1');
      html += '<div class="task-drawer__hint">I\'ll show what I\'m working on here.</div>';
    }

    list.innerHTML = html;

    // Wire stop buttons for each subagent item.
    list.querySelectorAll('.task-drawer__stop-btn').forEach((btn) => {
      btn.addEventListener('click', () => this._stopSubagent(btn.dataset.subId));
    });
  }

  /**
   * Request cancellation of a subagent via the stop endpoint.
   * @param {string} subId
   */
  async _stopSubagent(subId) {
    try {
      await this._api._post(`/chat/subagent/${subId}/stop`, {});
    } catch (err) {
      console.warn('[TaskStrip] Subagent stop request failed:', err);
    }
  }
}

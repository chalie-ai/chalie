import { escHtml, relativeTime, lsGet, lsSet } from './utils.js';

/**
 * Task drawer — displays active tasks and pending reminders in a slideout
 * panel that slides in from the right edge of the viewport.
 */
export class TaskStrip {
  constructor({ api }) {
    this._api = api;
    this._taskStripInterval = null;
    this._onAuthFailureCb = null;
  }

  /**
   * Bind trigger / close controls and start 60s polling interval.
   */
  init() {
    const triggerBtn = document.getElementById('taskDrawerBtn');
    const scrim      = document.getElementById('taskDrawerScrim');
    if (!triggerBtn || !scrim) return;

    triggerBtn.addEventListener('click', () => this._openDrawer());
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
   * Fetch and render active tasks and pending reminders.
   * Public — called on init and from the event router.
   */
  async loadActiveTasks() {
    try {
      const schedData = await this._api._get('/scheduler?status=pending').catch((e) => { if (e?.message === 'AUTH') throw e; return {}; });
      const reminders = (schedData.items || []).filter(
        r => r.status === 'pending' && r.due_at
      );
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

  _render(reminders = []) {
    const drawer   = document.getElementById('taskDrawer');
    const list     = document.getElementById('taskDrawerList');
    const trigger  = document.getElementById('taskDrawerBtn');
    if (!drawer || !list) return;

    const totalCount = reminders.length;

    // Update trigger button visibility and badge
    if (trigger) {
      if (totalCount === 0) {
        trigger.classList.add('hidden');
      } else {
        trigger.classList.remove('hidden');
      }
    }

    // Auto-close the drawer when all items are gone
    if (totalCount === 0) {
      this._closeDrawer();
      list.innerHTML = '';
      return;
    }

    let html = '';

    // Render pending reminders
    for (const r of reminders) {
      const msg = (r.message || '').slice(0, 80);
      const due = r.due_at ? relativeTime(r.due_at) : '';
      const id  = r.id;

      html += `<div class="task-drawer__item task-drawer__item--reminder">
        <span class="task-drawer__kind-dot task-drawer__kind-dot--reminder"></span>
        <span class="task-drawer__msg">${escHtml(msg)}</span>
        ${due ? `<span class="task-drawer__due">${escHtml(due)}</span>` : ''}
        <button class="task-drawer__dismiss" data-dismiss-reminder="${escHtml(id)}" aria-label="Dismiss reminder">&times;</button>
      </div>`;
    }

    // First-time hint
    if (!lsGet('task_strip_hint_shown')) {
      lsSet('task_strip_hint_shown', '1');
      html += '<div class="task-drawer__hint">I\'ll show what I\'m working on here.</div>';
    }

    list.innerHTML = html;

    // Wire dismiss buttons — reminders
    list.querySelectorAll('[data-dismiss-reminder]').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const remId = btn.dataset.dismissReminder;
        try {
          await this._api._delete(`/scheduler/${remId}`);
        } catch { /* ignore */ }
        this.loadActiveTasks();
      });
    });

  }
}

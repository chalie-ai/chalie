import { escHtml, relativeTime, lsGet, lsSet } from './utils.js';

/**
 * Persistent task strip — displays active tasks and pending reminders.
 */
export class TaskStrip {
  constructor({ api }) {
    this._api = api;
    this._taskStripInterval = null;
    this._onAuthFailureCb = null;
  }

  /**
   * Bind toggle button and start 60s polling interval.
   */
  init() {
    const toggle = document.getElementById('taskStripToggle');
    const strip = document.getElementById('taskStrip');
    if (!toggle || !strip) return;

    toggle.addEventListener('click', () => {
      strip.classList.toggle('--expanded');
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
      const [taskData, schedData] = await Promise.all([
        this._api._get('/system/observability/tasks').catch((e) => { if (e?.message === 'AUTH') throw e; return {}; }),
        this._api._get('/scheduler?status=pending').catch((e) => { if (e?.message === 'AUTH') throw e; return {}; }),
      ]);
      const tasks = (taskData.persistent_tasks || []).filter(
        t => t.status === 'accepted' || t.status === 'in_progress' || t.status === 'paused'
      );
      const reminders = (schedData.items || []).filter(
        r => r.status === 'pending' && r.due_at
      );
      this._renderTaskStrip(tasks, reminders);
    } catch (e) {
      if (e?.message === 'AUTH') {
        this._onAuthFailureCb?.();
      }
      // Other errors: silently fail — task strip is supplementary
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

  _renderTaskStrip(tasks, reminders = []) {
    const strip = document.getElementById('taskStrip');
    const list = document.getElementById('taskStripList');
    const countEl = document.getElementById('taskStripCount');
    if (!strip || !list) return;

    const totalCount = tasks.length + reminders.length;
    const wasEmpty = strip.classList.contains('hidden');

    if (totalCount === 0) {
      strip.classList.add('hidden');
      strip.classList.remove('--expanded');
      document.body.classList.remove('has-task-strip');
      return;
    }

    strip.classList.remove('hidden');
    document.body.classList.add('has-task-strip');
    countEl.textContent = totalCount;

    // Auto-expand when items appear for the first time (was hidden → now visible)
    if (wasEmpty) {
      strip.classList.add('--expanded');
    }

    let html = '';

    // Render persistent tasks
    for (const t of tasks) {
      const goal = (t.goal || 'Working…').slice(0, 60);
      const progress = t.progress || {};
      const coverage = Math.round((progress.coverage_estimate || 0) * 100);
      const summary = progress.last_summary || '';
      const pausedClass = t.status === 'paused' ? ' --paused' : '';

      // Step-level progress from plan DAG
      const plan = progress.plan;
      let stepsHtml = '';
      if (plan && plan.steps && plan.steps.length > 0) {
        const done = plan.steps.filter(s => s.status === 'completed' || s.status === 'skipped').length;
        const total = plan.steps.length;
        const current = plan.steps.find(s => s.status === 'in_progress');
        stepsHtml = `<div class="task-strip__steps">${done}/${total} steps</div>`;
        if (current) {
          stepsHtml += `<div class="task-strip__current-step">${escHtml(current.description)}</div>`;
        }
        if (plan.blocked_on) {
          stepsHtml += `<div class="task-strip__blocked">Blocked: ${escHtml(plan.blocked_reason || 'dependency failed')}</div>`;
        }
      }

      html += `<div class="task-strip__item${pausedClass}">
        <span class="task-strip__kind-dot task-strip__kind-dot--task"></span>
        <div class="task-strip__goal">${escHtml(goal)}</div>
        <div class="task-strip__progress-bar">
          <div class="task-strip__progress-fill" style="width:${coverage}%"></div>
        </div>
        ${stepsHtml}
        ${summary ? `<div class="task-strip__summary">${escHtml(summary)}</div>` : ''}
        <button class="task-strip__dismiss" data-dismiss-task="${t.id}" aria-label="Dismiss task">&times;</button>
      </div>`;
    }

    // Render pending reminders
    for (const r of reminders) {
      const msg = (r.message || '').slice(0, 80);
      const due = r.due_at ? relativeTime(r.due_at) : '';
      const id = r.id;

      html += `<div class="task-strip__item task-strip__item--reminder">
        <span class="task-strip__kind-dot task-strip__kind-dot--reminder"></span>
        <span class="task-strip__msg">${escHtml(msg)}</span>
        ${due ? `<span class="task-strip__due">${escHtml(due)}</span>` : ''}
        <button class="task-strip__dismiss" data-dismiss-reminder="${escHtml(id)}" aria-label="Dismiss reminder">&times;</button>
      </div>`;
    }

    // First-time hint
    if (!lsGet('task_strip_hint_shown')) {
      lsSet('task_strip_hint_shown', '1');
      html += '<div class="task-strip__hint">I\'ll show what I\'m working on here.</div>';
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

    // Wire dismiss buttons — persistent tasks
    list.querySelectorAll('[data-dismiss-task]').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const taskId = btn.dataset.dismissTask;
        try {
          await this._api._delete(`/system/observability/tasks/${taskId}`);
        } catch { /* ignore */ }
        this.loadActiveTasks();
      });
    });
  }
}

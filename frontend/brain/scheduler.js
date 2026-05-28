// Scheduler panel — schedule CRUD with filters.
const PanelScheduler = (() => {
  let _root = null;
  let _sub = 'all';
  let _items = [];
  let _loaded = false;

  function mount(root, sub) {
    _root = root;
    _sub = sub || 'all';
    _render();
  }

  function unmount() { _root = null; }

  function _render() {
    if (!_root) return;
    _root.innerHTML = `<div class="panel-header">
      <h2>Scheduler</h2>
      <button class="btn btn-primary" id="newSchedBtn">${Icons.Plus(14)} New Schedule</button>
    </div>
    <div id="schedContent"><div class="loading">Loading…</div></div>`;

    document.getElementById('newSchedBtn').addEventListener('click', () => _openModal(null));
    if (!_loaded) _load();
    else _renderList();
  }

  async function _load() {
    try {
      const res = await BrainApp.apiFetch('/scheduler');
      if (!res.ok) throw new Error();
      const data = await res.json();
      _items = data.schedules || data.items || [];
      _loaded = true;
    } catch { BrainApp.showToast('Failed to load schedules', 'error'); }
    _renderList();
  }

  function _renderList() {
    const el = document.getElementById('schedContent');
    if (!el) return;

    const filtered = _sub === 'all' ? _items : _items.filter(s => s.status === _sub);

    if (filtered.length === 0) {
      el.innerHTML = `<div class="empty-state"><div class="empty-icon">${Icons.Calendar(40)}</div><h3>No schedules</h3><p>Create your first scheduled task.</p></div>`;
      return;
    }

    el.innerHTML = `<table class="records-table">
      <thead><tr><th>Message</th><th>Status</th><th>Due</th><th>Recurrence</th><th>Type</th><th></th></tr></thead>
      <tbody>${filtered.map(s => {
        const statusCls = { pending: 'badge-warning', fired: 'badge-success', failed: 'badge-danger', cancelled: 'badge-muted' }[s.status] || 'badge-muted';
        return `<tr>
          <td class="key-cell">${BrainApp.escapeHtml(s.message || s.prompt || '')}</td>
          <td><span class="badge ${statusCls}">${BrainApp.escapeHtml(s.status || '')}</span></td>
          <td>${BrainApp.formatDate(s.due_at || s.due)}</td>
          <td>${BrainApp.escapeHtml(s.recurrence || '—')}</td>
          <td><span class="badge badge-muted">${BrainApp.escapeHtml(s.type || 'notification')}</span></td>
          <td class="row-actions">
            <button class="btn btn-sm btn-secondary" data-edit-sched="${s.id}">Edit</button>
            ${s.status === 'pending' ? `<button class="btn btn-sm btn-danger" data-cancel-sched="${s.id}">Cancel</button>` : ''}
          </td>
        </tr>`;
      }).join('')}</tbody>
    </table>`;

    el.querySelectorAll('[data-edit-sched]').forEach(b => {
      b.addEventListener('click', () => _openModal(Number(b.dataset.editSched) || b.dataset.editSched));
    });
    el.querySelectorAll('[data-cancel-sched]').forEach(b => {
      b.addEventListener('click', () => _cancelSchedule(Number(b.dataset.cancelSched) || b.dataset.cancelSched));
    });
  }

  function _openModal(id) {
    const item = id ? _items.find(s => s.id === id || s.id === String(id)) : null;
    const el = document.getElementById('schedContent');
    if (!el) return;

    el.innerHTML = `<div class="provider-form-page">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm back-btn" id="backToSched">${Icons.Chevron(14)} Back</button>
        <h3>${id ? 'Edit Schedule' : 'New Schedule'}</h3>
      </div>
      <form id="schedForm">
        <div class="form-group">
          <label for="schedMsg">Prompt / Message</label>
          <textarea id="schedMsg" rows="3" maxlength="1000" placeholder="What Chalie should do when this fires" required>${BrainApp.escapeHtml(item?.message || item?.prompt || '')}</textarea>
        </div>
        <div class="form-group">
          <label for="schedDue">Due Date & Time</label>
          <input type="datetime-local" id="schedDue" value="${item?.due_at ? item.due_at.slice(0, 16) : ''}" required>
        </div>
        <div class="form-group">
          <label for="schedType">Type</label>
          <select id="schedType">
            <option value="notification" ${item?.type === 'notification' ? 'selected' : ''}>Notification</option>
            <option value="prompt" ${item?.type === 'prompt' || item?.item_type === 'prompt' ? 'selected' : ''}>Prompt</option>
          </select>
        </div>
        <div class="form-group">
          <label for="schedRecur">Recurrence</label>
          <select id="schedRecur">
            <option value="">None (one-time)</option>
            <option value="interval" ${item?.recurrence === 'interval' ? 'selected' : ''}>Every X minutes</option>
            <option value="hourly" ${item?.recurrence === 'hourly' ? 'selected' : ''}>Hourly</option>
            <option value="daily" ${item?.recurrence === 'daily' ? 'selected' : ''}>Daily</option>
            <option value="weekdays" ${item?.recurrence === 'weekdays' ? 'selected' : ''}>Weekdays</option>
            <option value="weekly" ${item?.recurrence === 'weekly' ? 'selected' : ''}>Weekly</option>
            <option value="monthly" ${item?.recurrence === 'monthly' ? 'selected' : ''}>Monthly</option>
          </select>
        </div>
        <div class="form-actions">
          <button type="button" class="btn btn-secondary" id="cancelSchedBtn">Cancel</button>
          <button type="submit" class="btn btn-primary">Save</button>
        </div>
      </form>
    </div>`;

    document.getElementById('backToSched').addEventListener('click', _renderList);
    document.getElementById('cancelSchedBtn').addEventListener('click', _renderList);

    document.getElementById('schedForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const body = {
        message: document.getElementById('schedMsg').value.trim(),
        due_at: document.getElementById('schedDue').value,
        type: document.getElementById('schedType').value,
        recurrence: document.getElementById('schedRecur').value || null,
      };
      try {
        const method = id ? 'PUT' : 'POST';
        const path = id ? `/scheduler/${id}` : '/scheduler';
        const res = await BrainApp.apiFetch(path, { method, body: JSON.stringify(body) });
        if (res.ok) { BrainApp.showToast(id ? 'Schedule updated' : 'Schedule created', 'success'); _loaded = false; _load(); }
        else { const d = await res.json().catch(() => ({})); BrainApp.showToast(d.error || 'Save failed', 'error'); }
      } catch { BrainApp.showToast('Network error', 'error'); }
    });
  }

  function _showConfirm({ title, desc, confirmLabel, confirmClass, onConfirm }) {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    const esc = BrainApp.escapeHtml;
    overlay.innerHTML = `<div class="modal modal-sm">
      <div class="modal-header"><h3>${esc(title)}</h3></div>
      <p class="modal-desc">${desc}</p>
      <div class="modal-actions"><button class="btn btn-secondary" data-cancel>Cancel</button><button class="btn ${confirmClass}" data-confirm>${esc(confirmLabel)}</button></div>
    </div>`;
    document.body.appendChild(overlay);
    overlay.querySelector('[data-cancel]').addEventListener('click', () => overlay.remove());
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    overlay.querySelector('[data-confirm]').addEventListener('click', () => { overlay.remove(); onConfirm(); });
  }

  function _cancelSchedule(id) {
    _showConfirm({
      title: 'Cancel Schedule',
      desc: 'Cancel this scheduled item?',
      confirmLabel: 'Cancel Item',
      confirmClass: 'btn-danger',
      onConfirm: async () => {
        try {
          const res = await BrainApp.apiFetch(`/scheduler/${id}`, { method: 'DELETE' });
          if (res.ok) { BrainApp.showToast('Schedule cancelled', 'success'); _loaded = false; _load(); }
          else BrainApp.showToast('Cancel failed', 'error');
        } catch { BrainApp.showToast('Network error', 'error'); }
      }
    });
  }

  return { mount, unmount };
})();

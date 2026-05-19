// Capabilities panel — connected services + routing groups.
const PanelCapabilities = (() => {
  let _root = null;
  let _caps = [];
  let _loaded = false;

  function mount(root) {
    _root = root;
    _render();
  }

  function unmount() { _root = null; }

  function _render() {
    if (!_root) return;
    _root.innerHTML = `<div class="panel-header"><h2>Capabilities</h2></div>
    <div id="capsContent"><div class="loading">Loading…</div></div>`;
    if (!_loaded) _load();
    else _renderCaps();
  }

  async function _load() {
    try {
      const res = await BrainApp.apiFetch('/api/capabilities');
      if (!res.ok) throw new Error();
      const data = await res.json();
      _caps = data.capabilities || [];
      _loaded = true;
    } catch { BrainApp.showToast('Failed to load capabilities', 'error'); }
    _renderCaps();
  }

  function _renderCaps() {
    const el = document.getElementById('capsContent');
    if (!el) return;
    if (_caps.length === 0) {
      el.innerHTML = `<div class="empty-state"><div class="empty-icon">${Icons.Capability(40)}</div><h3>No capabilities</h3><p>No services are configured yet.</p></div>`;
      return;
    }

    el.innerHTML = `<div class="caps-grid">${_caps.map(c => {
      const connected = c.connected || c.status === 'connected';
      return `<div class="cap-card${connected ? ' connected' : ''}">
        <div class="cap-header">
          <div class="cap-name">${BrainApp.escapeHtml(c.name || '')}</div>
          <span class="status-dot ${connected ? 'active' : ''}"></span>
        </div>
        <div class="cap-meta">${BrainApp.escapeHtml(c.platform || '')}</div>
        ${c.version ? `<div class="cap-version">${BrainApp.escapeHtml(c.version)}</div>` : ''}
        <div class="cap-actions">
          ${connected
            ? `<button class="btn btn-sm btn-secondary" data-edit-cap="${c.id}">Edit</button>
               <button class="btn btn-sm btn-danger" data-disconnect="${c.id}">Disconnect</button>`
            : `<button class="btn btn-sm btn-primary" data-setup="${c.id}">Setup</button>`}
        </div>
      </div>`;
    }).join('')}</div>`;

    el.querySelectorAll('[data-setup]').forEach(b => {
      b.addEventListener('click', () => _openSetupForm(b.dataset.setup));
    });
    el.querySelectorAll('[data-edit-cap]').forEach(b => {
      b.addEventListener('click', () => _openSetupForm(b.dataset.editCap));
    });
    el.querySelectorAll('[data-disconnect]').forEach(b => {
      b.addEventListener('click', () => _disconnect(b.dataset.disconnect));
    });
  }

  async function _openSetupForm(id) {
    const cap = _caps.find(c => c.id === id);
    if (!cap) return;
    const el = document.getElementById('capsContent');
    if (!el) return;

    const connected = cap.connected || cap.status === 'connected';
    let config = {};
    if (connected) {
      try {
        const res = await BrainApp.apiFetch(`/api/capabilities/${id}`);
        if (res.ok) { const d = await res.json(); config = d.config || d.settings || d || {}; }
      } catch { /* use empty config */ }
    }

    const fields = cap.fields || [];
    el.innerHTML = `<div class="provider-form-page">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm back-btn" id="backToCaps">${Icons.Chevron(14)} Back</button>
        <h3>${BrainApp.escapeHtml(cap.name)} ${connected ? 'Settings' : 'Setup'}</h3>
      </div>
      <form id="capSetupForm">
        ${fields.map(f => {
          const val = config[f.name] ?? f.default ?? '';
          if (f.type === 'checkbox') {
            const checked = config[f.name] !== undefined ? config[f.name] : (f.default || false);
            return `<div class="form-group">
              <label class="switch-label">
                <label class="switch">
                  <input type="checkbox" id="cap_${f.name}" ${checked ? 'checked' : ''}>
                  <span class="switch-track"></span>
                </label>
                <span>${BrainApp.escapeHtml(f.label)}</span>
              </label>
            </div>`;
          }
          return `<div class="form-group">
            <label for="cap_${f.name}">${BrainApp.escapeHtml(f.label)}</label>
            <input type="${f.type === 'password' ? 'password' : 'text'}" id="cap_${f.name}"
              placeholder="${BrainApp.escapeHtml(f.placeholder || '')}"
              value="${f.type !== 'password' ? BrainApp.escapeHtml(String(val)) : ''}"
              ${f.required && !connected ? 'required' : ''}>
          </div>`;
        }).join('')}
        <div class="form-actions">
          <button type="button" class="btn btn-secondary" id="cancelCapBtn">Cancel</button>
          <button type="submit" class="btn btn-primary">${connected ? 'Save' : 'Connect'}</button>
        </div>
      </form>
    </div>`;

    document.getElementById('backToCaps').addEventListener('click', _renderCaps);
    document.getElementById('cancelCapBtn').addEventListener('click', _renderCaps);
    document.getElementById('capSetupForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const body = {};
      for (const f of fields) {
        const input = document.getElementById(`cap_${f.name}`);
        if (!input) continue;
        body[f.name] = f.type === 'checkbox' ? input.checked : input.value.trim();
      }
      try {
        const res = await BrainApp.apiFetch(`/api/capabilities/${id}/setup`, { method: 'POST', body: JSON.stringify(body) });
        if (res.ok) { BrainApp.showToast(connected ? 'Settings saved' : 'Capability connected', 'success'); _loaded = false; _load(); }
        else { const d = await res.json().catch(() => ({})); BrainApp.showToast(d.error || 'Setup failed', 'error'); }
      } catch { BrainApp.showToast('Network error', 'error'); }
    });
  }

  async function _disconnect(id) {
    try {
      const res = await BrainApp.apiFetch(`/api/capabilities/${id}/disconnect`, { method: 'POST' });
      if (res.ok) { BrainApp.showToast('Capability disconnected', 'success'); _loaded = false; _load(); }
      else BrainApp.showToast('Failed to disconnect', 'error');
    } catch { BrainApp.showToast('Network error', 'error'); }
  }

  return { mount, unmount };
})();

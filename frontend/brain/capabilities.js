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
            ? `<button class="btn btn-sm btn-danger" data-disconnect="${c.id}">Disconnect</button>`
            : `<button class="btn btn-sm btn-primary" data-setup="${c.id}">Setup</button>`}
        </div>
      </div>`;
    }).join('')}</div>`;

    el.querySelectorAll('[data-setup]').forEach(b => {
      b.addEventListener('click', () => _setup(b.dataset.setup));
    });
    el.querySelectorAll('[data-disconnect]').forEach(b => {
      b.addEventListener('click', () => _disconnect(b.dataset.disconnect));
    });
  }

  async function _setup(id) {
    try {
      const res = await BrainApp.apiFetch(`/api/capabilities/${id}/setup`, { method: 'POST', body: JSON.stringify({}) });
      if (res.ok) { BrainApp.showToast('Capability connected', 'success'); _loaded = false; _load(); }
      else { const d = await res.json().catch(() => ({})); BrainApp.showToast(d.error || 'Setup failed', 'error'); }
    } catch { BrainApp.showToast('Network error', 'error'); }
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

// MCP panel — Inbound server config + Outbound client connections.
//
// Inbound: the existing MCP-server config (token, port, enable) for
//   external agents (Claude Code, Codex) connecting INTO Chalie.
// Outbound: the new MCP-client manager for Chalie connecting OUT to
//   remote MCP servers (wired to /api/mcp-clients).
//
// Dylan's decision #4 (2026-05-31): both sections live in this file,
// in the existing 'mcp' tab — no new tab, no mcp_clients.js.
const PanelMcp = (() => {
  // ── State ──
  let _root = null;
  let _inConfig = {};
  let _inLoaded = false;
  let _outServers = [];
  let _outLoaded = false;
  let _editingId = null;

  // ── Mount / unmount ────────────────────────────────────────────────────────

  function mount(root) {
    _root = root;
    _render();
  }

  function unmount() { _root = null; _editingId = null; }

  // ── Top-level render ───────────────────────────────────────────────────────

  function _render() {
    if (!_root) return;
    _root.innerHTML = `
      <div class="panel-header"><h2>MCP</h2></div>

      <section class="mcp-section">
        <h3 class="mcp-section-title">Inbound</h3>
        <p class="panel-desc">
          External agents (Claude Code, Codex, CI bots) connect to Chalie via MCP.
          Manage access and view the connection token below.
        </p>
        <div id="mcpInboundContent"><div class="loading">Loading…</div></div>
      </section>

      <section class="mcp-section" style="margin-top:28px">
        <h3 class="mcp-section-title">Outbound</h3>
        <p class="panel-desc">
          Connect Chalie to remote MCP servers so their tools become available.
          Chalie uses the streamable-HTTP transport.
        </p>
        <div id="mcpOutboundContent"><div class="loading">Loading…</div></div>
      </section>
    `;
    if (!_inLoaded) _loadInbound();
    else _renderInbound();
    if (!_outLoaded) _loadOutbound();
    else _renderOutbound();
  }

  // ── ── Inbound section ─────────────────────────────────────────────────────

  async function _loadInbound() {
    try {
      const res = await BrainApp.apiFetch('/api/mcp-server');
      if (!res.ok) throw new Error();
      _inConfig = await res.json();
      _inLoaded = true;
    } catch {
      BrainApp.showToast('Failed to load MCP inbound settings', 'error');
    }
    _renderInbound();
  }

  function _renderInbound() {
    const el = document.getElementById('mcpInboundContent');
    if (!el) return;
    el.innerHTML = `<div class="mcp-settings">
      <div class="mcp-row">
        <label class="mcp-label">Server Enabled</label>
        <label class="switch">
          <input type="checkbox" id="mcpEnabled" ${_inConfig.enabled !== false ? 'checked' : ''}>
          <span class="switch-track"></span>
        </label>
      </div>
      <div class="form-group">
        <label for="mcpPort">Port</label>
        <div class="input-group">
          <span class="input-prefix">TCP</span>
          <input type="number" id="mcpPort" value="${_inConfig.port || 8462}" min="1024" max="65535">
        </div>
      </div>
      <div class="mcp-token-section">
        <h4>Connection Token</h4>
        <p class="mcp-hint">Give this token to external agents so they can authenticate.</p>
        <div class="input-group">
          <input type="text" id="mcpTokenInput" value="${BrainApp.escapeHtml(_inConfig.token || '')}" readonly class="monospace">
          <button class="input-suffix-btn" id="mcpCopy" title="Copy">${Icons.Copy(14)}</button>
        </div>
        <div style="margin-top:10px">
          <button class="btn btn-sm btn-danger" id="mcpRegen">Regenerate Token</button>
        </div>
      </div>
    </div>`;

    document.getElementById('mcpEnabled').addEventListener('change', (e) =>
      _saveInbound({ enabled: e.target.checked }));
    document.getElementById('mcpPort').addEventListener('blur', () => {
      const val = Number(document.getElementById('mcpPort').value);
      if (val >= 1024 && val <= 65535 && val !== _inConfig.port)
        _saveInbound({ port: val });
    });
    document.getElementById('mcpCopy').addEventListener('click', () => {
      navigator.clipboard.writeText(_inConfig.token || '').then(
        () => BrainApp.showToast('Token copied', 'success'),
        () => BrainApp.showToast('Copy failed', 'error'),
      );
    });
    document.getElementById('mcpRegen').addEventListener('click', _regenToken);
  }

  async function _saveInbound(updates) {
    try {
      const res = await BrainApp.apiFetch('/api/mcp-server', {
        method: 'PUT', body: JSON.stringify(updates),
      });
      if (res.ok) {
        Object.assign(_inConfig, updates);
        BrainApp.showToast('Settings saved', 'success');
      } else {
        BrainApp.showToast('Save failed', 'error');
      }
    } catch { BrainApp.showToast('Network error', 'error'); }
  }

  async function _regenToken() {
    try {
      const res = await BrainApp.apiFetch('/api/mcp-server/regenerate-token', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        _inConfig.token = data.token;
        _renderInbound();
        BrainApp.showToast('Token regenerated', 'success');
      } else {
        BrainApp.showToast('Regenerate failed', 'error');
      }
    } catch { BrainApp.showToast('Network error', 'error'); }
  }

  // ── ── Outbound section ────────────────────────────────────────────────────

  async function _loadOutbound() {
    try {
      const res = await BrainApp.apiFetch('/api/mcp-clients/');
      if (!res.ok) throw new Error();
      _outServers = await res.json();
      _outLoaded = true;
    } catch {
      BrainApp.showToast('Failed to load MCP outbound servers', 'error');
      _outServers = [];
    }
    _renderOutbound();
  }

  function _renderOutbound() {
    const el = document.getElementById('mcpOutboundContent');
    if (!el) return;

    const rows = _outServers.map((s) =>
      String(s.id) === String(_editingId) ? _editCardHtml(s) : _displayCardHtml(s),
    ).join('');

    el.innerHTML = `
      ${rows}
      <div class="mcp-out-add-form" id="mcpAddForm">
        <h4 class="mcp-out-add-title">Add Remote MCP Server</h4>
        <div class="mcp-out-form-row">
          <label class="mcp-label">Enabled</label>
          <label class="switch">
            <input type="checkbox" id="mcpOutEnabled" checked>
            <span class="switch-track"></span>
          </label>
        </div>
        <div class="form-group">
          <label for="mcpOutName">Name</label>
          <input type="text" id="mcpOutName" placeholder="e.g. taskie" class="form-input">
        </div>
        <div class="form-group">
          <label for="mcpOutHost">Host (incl. port)</label>
          <input type="text" id="mcpOutHost" placeholder="http://grck.lan:5100/mcp" class="form-input">
        </div>
        <div class="form-group">
          <label>Additional Headers</label>
          <div id="mcpOutHeaders" class="mcp-headers-list"></div>
          <button class="btn btn-xs btn-secondary" id="mcpAddHeader" type="button">+ Header</button>
        </div>
        <button class="btn btn-primary" id="mcpOutAddBtn" type="button">Add Server</button>
      </div>
    `;

    // Wire card-level action buttons.
    el.querySelectorAll('[data-action]').forEach((btn) => {
      btn.addEventListener('click', _handleCardAction);
    });

    const addHeaders = document.getElementById('mcpOutHeaders');
    document.getElementById('mcpAddHeader').addEventListener('click', () => _addHeaderRow(addHeaders));
    document.getElementById('mcpOutAddBtn').addEventListener('click', _addServer);

    if (_editingId != null) _wireEditForm();
  }

  function _displayCardHtml(s) {
    return `
      <div class="mcp-out-card" data-id="${BrainApp.escapeHtml(s.id)}">
        <div class="mcp-out-card-header">
          <div>
            <span class="mcp-out-name">${BrainApp.escapeHtml(s.name)}</span>
            <span class="mcp-out-host"> — ${BrainApp.escapeHtml(s.host)}</span>
          </div>
          <div class="mcp-out-badges">
            <span class="badge badge-status badge-${s.status}">${BrainApp.escapeHtml(s.status)}</span>
            <span class="badge ${s.enabled ? 'badge-enabled' : 'badge-disabled'}">${s.enabled ? 'enabled' : 'disabled'}</span>
          </div>
        </div>
        <div class="mcp-out-card-actions">
          <button class="btn btn-xs btn-secondary" data-action="edit" data-id="${BrainApp.escapeHtml(s.id)}">Edit</button>
          <button class="btn btn-xs btn-secondary" data-action="test" data-id="${BrainApp.escapeHtml(s.id)}">Test</button>
          <button class="btn btn-xs ${s.enabled ? 'btn-secondary' : 'btn-primary'}" data-action="toggle" data-id="${BrainApp.escapeHtml(s.id)}" data-enabled="${s.enabled}">
            ${s.enabled ? 'Disable' : 'Enable'}
          </button>
          <button class="btn btn-xs btn-danger" data-action="delete" data-id="${BrainApp.escapeHtml(s.id)}">Delete</button>
        </div>
      </div>
    `;
  }

  function _editCardHtml(s) {
    const headersHtml = Object.entries(s.headers || {})
      .map(([k, v]) => `<div class="mcp-header-row">${_headerRowHtml(k, v)}</div>`)
      .join('');
    return `
      <div class="mcp-out-card mcp-out-card-editing" data-id="${BrainApp.escapeHtml(s.id)}">
        <h4 class="mcp-out-add-title">Edit MCP Server</h4>
        <div class="mcp-out-form-row">
          <label class="mcp-label">Enabled</label>
          <label class="switch">
            <input type="checkbox" id="mcpEditEnabled" ${s.enabled ? 'checked' : ''}>
            <span class="switch-track"></span>
          </label>
        </div>
        <div class="form-group">
          <label for="mcpEditName">Name</label>
          <input type="text" id="mcpEditName" class="form-input" value="${BrainApp.escapeHtml(s.name)}">
        </div>
        <div class="form-group">
          <label for="mcpEditHost">Host (incl. port)</label>
          <input type="text" id="mcpEditHost" class="form-input" value="${BrainApp.escapeHtml(s.host)}">
        </div>
        <div class="form-group">
          <label>Additional Headers</label>
          <div id="mcpEditHeaders" class="mcp-headers-list">${headersHtml}</div>
          <button class="btn btn-xs btn-secondary" id="mcpEditAddHeader" type="button">+ Header</button>
        </div>
        <div class="mcp-out-card-actions">
          <button class="btn btn-primary" id="mcpEditSaveBtn" type="button">Save</button>
          <button class="btn btn-secondary" id="mcpEditCancelBtn" type="button">Cancel</button>
        </div>
      </div>
    `;
  }

  function _wireEditForm() {
    const editHeaders = document.getElementById('mcpEditHeaders');
    if (!editHeaders) { _editingId = null; return; }
    _wireHeaderRemoves(editHeaders);
    document.getElementById('mcpEditAddHeader').addEventListener('click', () => _addHeaderRow(editHeaders));
    document.getElementById('mcpEditSaveBtn').addEventListener('click', () => _saveEdit(_editingId));
    document.getElementById('mcpEditCancelBtn').addEventListener('click', _cancelEdit);
  }

  function _handleCardAction(e) {
    const btn = e.currentTarget;
    const action = btn.dataset.action;
    const id = btn.dataset.id;
    if (action === 'edit') _editServer(id);
    else if (action === 'test') _testServer(id);
    else if (action === 'toggle') _toggleServer(id, btn.dataset.enabled === 'true');
    else if (action === 'delete') _deleteServer(id);
  }

  function _headerRowHtml(key, val) {
    return `
      <input type="text" placeholder="Header name" class="form-input mcp-header-key" value="${BrainApp.escapeHtml(key || '')}">
      <span class="mcp-header-sep">:</span>
      <input type="text" placeholder="Value" class="form-input mcp-header-val" value="${BrainApp.escapeHtml(val || '')}">
      <button class="btn btn-xs btn-danger mcp-header-remove" type="button">✕</button>
    `;
  }

  function _addHeaderRow(container) {
    const row = document.createElement('div');
    row.className = 'mcp-header-row';
    row.innerHTML = _headerRowHtml('', '');
    row.querySelector('.mcp-header-remove').addEventListener('click', () => row.remove());
    container.appendChild(row);
  }

  function _wireHeaderRemoves(container) {
    container.querySelectorAll('.mcp-header-row .mcp-header-remove').forEach((btn) => {
      btn.addEventListener('click', () => btn.closest('.mcp-header-row').remove());
    });
  }

  function _collectHeaders(container) {
    const headers = {};
    container.querySelectorAll('.mcp-header-row').forEach((row) => {
      const key = row.querySelector('.mcp-header-key').value.trim();
      const val = row.querySelector('.mcp-header-val').value.trim();
      if (key) headers[key] = val;
    });
    return headers;
  }

  async function _addServer() {
    const name = (document.getElementById('mcpOutName').value || '').trim();
    const host = (document.getElementById('mcpOutHost').value || '').trim();
    const enabled = document.getElementById('mcpOutEnabled').checked;
    const headers = _collectHeaders(document.getElementById('mcpOutHeaders'));

    if (!name) { BrainApp.showToast('Name is required', 'error'); return; }
    if (!host) { BrainApp.showToast('Host is required', 'error'); return; }

    try {
      const res = await BrainApp.apiFetch('/api/mcp-clients/', {
        method: 'POST',
        body: JSON.stringify({ name, host, headers, enabled }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        BrainApp.showToast(err.error || 'Failed to add server', 'error');
        return;
      }
      BrainApp.showToast(`Server "${name}" added`, 'success');
      _outLoaded = false;
      _loadOutbound();
    } catch { BrainApp.showToast('Network error', 'error'); }
  }

  function _editServer(id) {
    _editingId = id;
    _renderOutbound();
  }

  function _cancelEdit() {
    _editingId = null;
    _renderOutbound();
  }

  async function _saveEdit(id) {
    const name = (document.getElementById('mcpEditName').value || '').trim();
    const host = (document.getElementById('mcpEditHost').value || '').trim();
    const enabled = document.getElementById('mcpEditEnabled').checked;
    const headers = _collectHeaders(document.getElementById('mcpEditHeaders'));

    if (!name) { BrainApp.showToast('Name is required', 'error'); return; }
    if (!host) { BrainApp.showToast('Host is required', 'error'); return; }

    try {
      const res = await BrainApp.apiFetch(`/api/mcp-clients/${id}`, {
        method: 'PUT',
        body: JSON.stringify({ name, host, headers, enabled }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        BrainApp.showToast(err.error || 'Failed to update server', 'error');
        return;
      }
      BrainApp.showToast(`Server "${name}" updated`, 'success');
      _editingId = null;
      // Re-sync the remote tool set against the edited host/headers.
      _testServer(id, { silent: true });
    } catch { BrainApp.showToast('Network error', 'error'); }
  }

  async function _testServer(id, { silent = false } = {}) {
    try {
      const res = await BrainApp.apiFetch(`/api/mcp-clients/${id}/test`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        if (!silent) {
          const msg = data.reachable
            ? `Online — ${data.tool_count} tool(s) synced`
            : `Offline (${data.status})`;
          BrainApp.showToast(msg, data.reachable ? 'success' : 'error');
        }
      } else if (!silent) {
        BrainApp.showToast('Test request failed', 'error');
      }
    } catch {
      if (!silent) BrainApp.showToast('Network error', 'error');
    } finally {
      // Always refresh the list so a post-save resync reflects the edit
      // even if the test request itself failed.
      _outLoaded = false;
      _loadOutbound();
    }
  }

  async function _toggleServer(id, currentlyEnabled) {
    const enabled = !currentlyEnabled;
    try {
      const res = await BrainApp.apiFetch(`/api/mcp-clients/${id}`, {
        method: 'PUT',
        body: JSON.stringify({ enabled }),
      });
      if (!res.ok) { BrainApp.showToast('Update failed', 'error'); return; }
      BrainApp.showToast(enabled ? 'Server enabled' : 'Server disabled', 'success');
      _outLoaded = false;
      _loadOutbound();
    } catch { BrainApp.showToast('Network error', 'error'); }
  }

  async function _deleteServer(id) {
    if (!confirm('Delete this MCP server connection? This will remove its tools and policy rows.')) return;
    try {
      const res = await BrainApp.apiFetch(`/api/mcp-clients/${id}`, { method: 'DELETE' });
      if (!res.ok) { BrainApp.showToast('Delete failed', 'error'); return; }
      BrainApp.showToast('Server deleted', 'success');
      _outLoaded = false;
      _loadOutbound();
    } catch { BrainApp.showToast('Network error', 'error'); }
  }

  return { mount, unmount };
})();

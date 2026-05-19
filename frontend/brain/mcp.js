// MCP Server panel — enable/disable, port, token.
const PanelMcp = (() => {
  let _root = null;
  let _config = {};
  let _loaded = false;

  function mount(root) {
    _root = root;
    _render();
  }

  function unmount() { _root = null; }

  function _render() {
    if (!_root) return;
    _root.innerHTML = `<div class="panel-header"><h2>MCP Server</h2></div>
    <p class="panel-desc">External agents (Claude Code, Codex, CI bots) connect to Chalie via MCP. Manage access and view the connection token below.</p>
    <div id="mcpContent"><div class="loading">Loading…</div></div>`;
    if (!_loaded) _load();
    else _renderConfig();
  }

  async function _load() {
    try {
      const res = await BrainApp.apiFetch('/api/mcp-server');
      if (!res.ok) throw new Error();
      _config = await res.json();
      _loaded = true;
    } catch { BrainApp.showToast('Failed to load MCP settings', 'error'); }
    _renderConfig();
  }

  function _renderConfig() {
    const el = document.getElementById('mcpContent');
    if (!el) return;

    el.innerHTML = `<div class="mcp-settings">
      <div class="mcp-row">
        <label class="mcp-label">Server Enabled</label>
        <label class="switch">
          <input type="checkbox" id="mcpEnabled" ${_config.enabled !== false ? 'checked' : ''}>
          <span class="switch-track"></span>
        </label>
      </div>
      <div class="form-group">
        <label for="mcpPort">Port</label>
        <div class="input-group">
          <span class="input-prefix">TCP</span>
          <input type="number" id="mcpPort" value="${_config.port || 8462}" min="1024" max="65535">
        </div>
      </div>
      <div class="mcp-token-section">
        <h4>Connection Token</h4>
        <p class="mcp-hint">Give this token to external agents so they can authenticate.</p>
        <div class="input-group">
          <input type="text" id="mcpTokenInput" value="${BrainApp.escapeHtml(_config.token || '')}" readonly class="monospace">
          <button class="input-suffix-btn" id="mcpCopy" title="Copy">${Icons.Copy(14)}</button>
        </div>
        <div style="margin-top:10px">
          <button class="btn btn-sm btn-danger" id="mcpRegen">Regenerate Token</button>
        </div>
      </div>
    </div>`;

    document.getElementById('mcpEnabled').addEventListener('change', (e) => _save({ enabled: e.target.checked }));
    document.getElementById('mcpPort').addEventListener('blur', () => {
      const val = Number(document.getElementById('mcpPort').value);
      if (val >= 1024 && val <= 65535 && val !== _config.port) _save({ port: val });
    });
    document.getElementById('mcpCopy').addEventListener('click', () => {
      navigator.clipboard.writeText(_config.token || '').then(
        () => BrainApp.showToast('Token copied', 'success'),
        () => BrainApp.showToast('Copy failed', 'error'),
      );
    });
    document.getElementById('mcpRegen').addEventListener('click', _regen);
  }

  async function _save(updates) {
    try {
      const res = await BrainApp.apiFetch('/api/mcp-server', { method: 'PUT', body: JSON.stringify(updates) });
      if (res.ok) { Object.assign(_config, updates); BrainApp.showToast('Settings saved', 'success'); }
      else BrainApp.showToast('Save failed', 'error');
    } catch { BrainApp.showToast('Network error', 'error'); }
  }

  async function _regen() {
    try {
      const res = await BrainApp.apiFetch('/api/mcp-server/regenerate-token', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        _config.token = data.token;
        _renderConfig();
        BrainApp.showToast('Token regenerated', 'success');
      } else BrainApp.showToast('Regenerate failed', 'error');
    } catch { BrainApp.showToast('Network error', 'error'); }
  }

  return { mount, unmount };
})();

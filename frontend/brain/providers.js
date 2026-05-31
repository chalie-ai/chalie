// Providers panel — LLM provider CRUD + selection.
const PanelProviders = (() => {
  let _root = null;
  let _providers = [];
  let _selectedId = null;
  let _editPlatform = 'ollama';
  let _editingId = null;
  let _editModel = '';
  let _modelsFetchInFlight = false;
  let _modelFetchTimer = null;
  const _MODEL_FETCH_DEBOUNCE_MS = 600;

  const PLATFORM_CONFIG = {
    ollama: { desc: 'Run locally — no API key needed.', hasHost: true, hasApiKey: false, placeholder: 'e.g. gemma4:31b', models: [] },
    anthropic: { desc: 'API key from <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener noreferrer">console.anthropic.com</a>', hasHost: false, hasApiKey: true, placeholder: 'e.g. claude-sonnet-4-6', models: [] },
    openai: { desc: 'API key from <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer">platform.openai.com</a>', hasHost: false, hasApiKey: true, placeholder: 'e.g. gpt-4o', models: ['gpt-4o', 'gpt-4.1', 'o3', 'o4-mini'] },
    gemini: { desc: 'API key from <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener noreferrer">aistudio.google.com/apikey</a>', hasHost: false, hasApiKey: true, placeholder: 'e.g. gemini-2.5-flash', models: ['gemini-3.1-pro-preview', 'gemini-3-flash-preview', 'gemini-2.5-pro', 'gemini-2.5-flash'] },
    openai_compatible: { desc: 'Any OpenAI-compatible API (Groq, DeepSeek, Together, etc.)', hasHost: true, hasApiKey: true, placeholder: 'e.g. MiniMax-M2', models: [] },
  };

  async function mount(root) {
    _root = root;
    root.innerHTML = `<div class="panel-header">
      <h2>LLM Providers</h2>
      <button class="btn btn-primary" id="addProviderBtn">${Icons.Plus(14)} Add Provider</button>
    </div>
    <div id="providersList" class="providers-list"><div class="loading">Loading providers…</div></div>`;
    document.getElementById('addProviderBtn').addEventListener('click', () => _openModal(null));
    await _load();
  }

  function unmount() { _root = null; }

  async function _load() {
    try {
      const [provRes, selRes] = await Promise.all([
        BrainApp.apiFetch('/providers'),
        BrainApp.apiFetch('/providers/selected'),
      ]);
      if (provRes.ok) { const d = await provRes.json(); _providers = d.providers || []; }
      if (selRes.ok) { const d = await selRes.json(); _selectedId = d.provider ? d.provider.id : null; }
    } catch (e) { BrainApp.showToast('Failed to connect to backend', 'error'); }
    _render();
  }

  function _render() {
    const el = document.getElementById('providersList');
    if (!el) return;
    if (_providers.length === 0) {
      el.innerHTML = `<div class="empty-state"><div class="empty-icon">${Icons.Providers(40)}</div><h3>No providers</h3><p>Add your first LLM provider to get started.</p></div>`;
      return;
    }
    el.innerHTML = _providers.map(p => {
      const isSelected = p.id === _selectedId;
      const warn = p.decrypt_failed ? `<span class="provider-warn" title="Misconfigured — re-enter credentials">⚠</span>` : '';
      return `<div class="provider-card${isSelected ? ' selected' : ''}" data-id="${p.id}">
        <label class="provider-radio">
          <input type="radio" name="sel_prov" value="${p.id}" ${isSelected ? 'checked' : ''}>
          <span class="radio-dot"></span>
        </label>
        <div class="provider-info">
          <div class="provider-name">${BrainApp.escapeHtml(p.name)}${warn}</div>
          <div class="provider-meta">
            <span class="badge badge-${p.platform}">${BrainApp.escapeHtml(p.platform)}</span>
            ${p.supports_vision ? `<span class="badge badge-success" title="Verified image understanding">Vision</span>` : ''}
            <span>${BrainApp.escapeHtml(p.model || 'no model')}</span>
            ${p.host ? `<span>· ${BrainApp.escapeHtml(p.host)}</span>` : ''}
          </div>
        </div>
        <div class="provider-actions">
          <button class="btn btn-sm btn-secondary" data-edit="${p.id}">Edit</button>
          <button class="btn btn-sm btn-danger" data-del="${p.id}">Delete</button>
        </div>
      </div>`;
    }).join('');

    el.querySelectorAll('input[name="sel_prov"]').forEach(r => {
      r.addEventListener('change', () => _selectProvider(Number(r.value)));
    });
    el.querySelectorAll('[data-edit]').forEach(b => {
      b.addEventListener('click', () => _openModal(Number(b.dataset.edit)));
    });
    el.querySelectorAll('[data-del]').forEach(b => {
      b.addEventListener('click', () => _confirmDelete(Number(b.dataset.del)));
    });
  }

  async function _selectProvider(id) {
    try {
      const res = await BrainApp.apiFetch('/providers/selected', { method: 'PUT', body: JSON.stringify({ provider_id: id }) });
      if (res.ok) { _selectedId = id; _render(); BrainApp.showToast('Provider selected', 'success'); }
      else { const e = await res.json(); BrainApp.showToast(e.error || 'Failed', 'error'); }
    } catch { BrainApp.showToast('Failed to select provider', 'error'); }
  }

  function _openModal(id) {
    _editingId = id;
    _editPlatform = 'ollama';
    _editModel = '';
    const p = id ? _providers.find(x => x.id === id) : null;
    if (p) { _editPlatform = p.platform; _editModel = p.model || ''; }

    const cfg = PLATFORM_CONFIG[_editPlatform];
    const platforms = Object.keys(PLATFORM_CONFIG);

    const root = document.getElementById('providersList');
    if (!root) return;
    root.innerHTML = `<div class="provider-form-page">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm back-btn" id="backToProviders">${Icons.Chevron(14)} Back</button>
        <h3>${id ? 'Edit Provider' : 'Add Provider'}</h3>
      </div>
      <div class="platform-tabs" id="platTabs">
        ${platforms.map(k => `<button class="platform-tab${k === _editPlatform ? ' active' : ''}" data-plat="${k}">${k === 'openai_compatible' ? 'OpenAI-Compat' : k.charAt(0).toUpperCase() + k.slice(1)}</button>`).join('')}
      </div>
      <p class="platform-desc" id="platDesc">${cfg.desc}</p>
      <form id="providerForm">
        <div class="form-group">
          <label for="pName">Provider Name</label>
          <input type="text" id="pName" placeholder="e.g. My Ollama" required value="${BrainApp.escapeHtml(p?.name || '')}">
        </div>
        <div class="form-group" id="hostGroup" style="display:${cfg.hasHost ? '' : 'none'}">
          <label for="pHost">Host</label>
          <input type="text" id="pHost" value="${BrainApp.escapeHtml(p?.host || 'http://localhost:11434')}">
        </div>
        <div class="form-group" id="keyGroup" style="display:${cfg.hasApiKey ? '' : 'none'}">
          <label for="pKey">API Key</label>
          <input type="password" id="pKey" placeholder="${id ? 'Leave blank to keep existing' : 'Enter API key'}">
        </div>
        <div class="form-group">
          <label for="pModel">Model</label>
          <select id="pModel"><option value="">Select model…</option></select>
          <span class="model-status" id="modelStatus"></span>
        </div>
        <div class="form-actions">
          <button type="button" class="btn btn-secondary" id="cancelProvBtn">Cancel</button>
          <button type="button" class="btn btn-secondary" id="testProvBtn">Test</button>
          <button type="submit" class="btn btn-primary">Save</button>
        </div>
      </form>
    </div>`;

    _populateModels();

    document.getElementById('backToProviders').addEventListener('click', _render);
    document.getElementById('cancelProvBtn').addEventListener('click', _render);

    document.getElementById('platTabs').addEventListener('click', (e) => {
      const tab = e.target.closest('[data-plat]');
      if (!tab) return;
      _editPlatform = tab.dataset.plat;
      document.querySelectorAll('#platTabs .platform-tab').forEach(t => t.classList.toggle('active', t === tab));
      const c = PLATFORM_CONFIG[_editPlatform];
      document.getElementById('platDesc').innerHTML = c.desc;
      document.getElementById('hostGroup').style.display = c.hasHost ? '' : 'none';
      document.getElementById('keyGroup').style.display = c.hasApiKey ? '' : 'none';
      _populateModels();
      if (_canFetchModels()) _fetchModels();
    });

    document.getElementById('pHost')?.addEventListener('input', _debouncedFetchModels);
    document.getElementById('pKey')?.addEventListener('input', _debouncedFetchModels);

    document.getElementById('testProvBtn').addEventListener('click', _testConnection);
    document.getElementById('providerForm').addEventListener('submit', (e) => { e.preventDefault(); _saveProvider(); });

    if (_canFetchModels()) _fetchModels();
  }

  function _populateModels() {
    const sel = document.getElementById('pModel');
    if (!sel) return;
    sel.innerHTML = '<option value="">Select model…</option>';
    const models = PLATFORM_CONFIG[_editPlatform].models;
    for (const m of models) {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      if (m === _editModel) opt.selected = true;
      sel.appendChild(opt);
    }
    if (_editModel && !models.includes(_editModel)) {
      const opt = document.createElement('option');
      opt.value = _editModel; opt.textContent = _editModel; opt.selected = true;
      sel.appendChild(opt);
    }
  }

  function _canFetchModels() {
    const cfg = PLATFORM_CONFIG[_editPlatform];
    if (cfg.hasApiKey && !document.getElementById('pKey')?.value?.trim()) return false;
    return true;
  }

  function _debouncedFetchModels() {
    clearTimeout(_modelFetchTimer);
    _modelFetchTimer = setTimeout(() => { if (_canFetchModels()) _fetchModels(); }, _MODEL_FETCH_DEBOUNCE_MS);
  }

  async function _fetchModels() {
    if (_modelsFetchInFlight) return;
    _modelsFetchInFlight = true;
    const status = document.getElementById('modelStatus');
    if (status) { status.textContent = 'Loading models…'; status.className = 'model-status loading'; }
    try {
      const creds = {};
      if (PLATFORM_CONFIG[_editPlatform].hasHost) creds.host = document.getElementById('pHost')?.value?.trim();
      if (PLATFORM_CONFIG[_editPlatform].hasApiKey) creds.api_key = document.getElementById('pKey')?.value?.trim();
      const res = await BrainApp.apiFetch('/providers/list-models', { method: 'POST', body: JSON.stringify({ platform: _editPlatform, ...creds }) });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.models) {
        PLATFORM_CONFIG[_editPlatform].models = data.models.map(m => typeof m === 'string' ? m : m?.id || '').filter(Boolean);
        _populateModels();
        if (status) { status.textContent = `${PLATFORM_CONFIG[_editPlatform].models.length} models`; status.className = 'model-status ok'; }
      } else {
        if (status) { status.textContent = data.error || 'Failed'; status.className = 'model-status err'; }
      }
    } catch (e) {
      if (status) { status.textContent = 'Network error'; status.className = 'model-status err'; }
    }
    _modelsFetchInFlight = false;
  }

  async function _testConnection() {
    const btn = document.getElementById('testProvBtn');
    btn.disabled = true; btn.textContent = 'Testing…';
    try {
      const body = { platform: _editPlatform, name: document.getElementById('pName').value.trim(), model: document.getElementById('pModel').value };
      if (PLATFORM_CONFIG[_editPlatform].hasHost) body.host = document.getElementById('pHost').value.trim();
      if (PLATFORM_CONFIG[_editPlatform].hasApiKey) body.api_key = document.getElementById('pKey').value.trim();
      const res = await BrainApp.apiFetch('/providers/test', { method: 'POST', body: JSON.stringify(body) });
      const data = await res.json().catch(() => ({}));
      if (res.ok) BrainApp.showToast('Connection successful', 'success');
      else BrainApp.showToast(data.error || 'Test failed', 'error');
    } catch { BrainApp.showToast('Network error', 'error'); }
    btn.disabled = false; btn.textContent = 'Test';
  }

  async function _saveProvider() {
    const body = { platform: _editPlatform, name: document.getElementById('pName').value.trim(), model: document.getElementById('pModel').value };
    if (PLATFORM_CONFIG[_editPlatform].hasHost) body.host = document.getElementById('pHost').value.trim();
    if (PLATFORM_CONFIG[_editPlatform].hasApiKey) { const k = document.getElementById('pKey').value.trim(); if (k) body.api_key = k; }

    try {
      const method = _editingId ? 'PUT' : 'POST';
      const path = _editingId ? `/providers/${_editingId}` : '/providers';
      const res = await BrainApp.apiFetch(path, { method, body: JSON.stringify(body) });
      if (res.ok) {
        BrainApp.showToast(_editingId ? 'Provider updated' : 'Provider added', 'success');
        await _load();
        if (BrainApp.isProvidersOnly()) {
          try {
            const sr = await BrainApp.apiFetch('/auth/status');
            if (sr.ok) {
              const sd = await sr.json();
              if (sd.has_providers) BrainApp.liftProvidersOnly();
            }
          } catch { /* keep locked */ }
        }
      } else {
        const e = await res.json().catch(() => ({}));
        BrainApp.showToast(e.error || 'Save failed', 'error');
      }
    } catch { BrainApp.showToast('Network error', 'error'); }
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

  function _confirmDelete(id) {
    const p = _providers.find(x => x.id === id);
    _showConfirm({
      title: 'Delete Provider',
      desc: `Delete "${BrainApp.escapeHtml(p?.name || 'this provider')}"? This cannot be undone.`,
      confirmLabel: 'Delete',
      confirmClass: 'btn-danger',
      onConfirm: async () => {
        try {
          const res = await BrainApp.apiFetch(`/providers/${id}`, { method: 'DELETE' });
          if (res.ok) { BrainApp.showToast('Provider deleted', 'success'); await _load(); }
          else { BrainApp.showToast('Delete failed', 'error'); }
        } catch { BrainApp.showToast('Network error', 'error'); }
      }
    });
  }

  return { mount, unmount };
})();

// Providers panel — LLM provider CRUD with a reactive setup wizard.
//
// Add flow is progressive: pick a provider from the curated catalog → the host
// is pre-filled (or skipped for native APIs) → the API key field appears → once
// credentials are present, models are fetched live via POST /providers/list-models
// and the model picker is revealed. Fields a provider doesn't need are skipped.
const PanelProviders = (() => {
  let _root = null;
  let _providers = [];
  let _selectedId = null;
  let _catalog = [];
  let _catalogLoaded = false;

  // Vision / delegate provider selection state.
  let _visionId = null;       // explicitly-pinned vision provider id, or null
  let _visionSource = 'none'; // 'explicit' | 'auto' | 'none'
  let _delegateId = null;     // explicitly-pinned delegate provider id, or null
  let _delegateSource = 'none';

  // Wizard working state.
  let _preset = null;          // the chosen catalog preset (or a synthetic one)
  let _editingId = null;
  let _editModel = '';
  let _models = [];            // live-fetched model list for the current preset
  let _modelsFetchInFlight = false;
  let _modelFetchTimer = null;
  let _lastFetchKey = '';      // dedupe identical credential fetches
  const _MODEL_FETCH_DEBOUNCE_MS = 600;

  // A synthetic preset so users can still add any provider not in the catalog.
  const _CUSTOM_PRESET = {
    id: 'custom', name: 'Custom (OpenAI-compatible)',
    platform: 'openai_compatible', host: '', needs_key: true,
  };

  // ── Mount / list view ───────────────────────────────────────────────

  async function mount(root) {
    _root = root;
    root.innerHTML = `<div class="panel-header">
      <h2>LLM Providers</h2>
      <button class="btn btn-primary" id="addProviderBtn">${Icons.Plus(14)} Add Provider</button>
    </div>
    <div id="providersList" class="providers-list"><div class="loading">Loading providers…</div></div>
    <div id="roleSelectors"></div>`;
    document.getElementById('addProviderBtn').addEventListener('click', () => _openWizard(null));
    await _load();
  }

  function unmount() { _root = null; }

  async function _load() {
    try {
      const [provRes, selRes, visRes, delRes] = await Promise.all([
        BrainApp.apiFetch('/providers'),
        BrainApp.apiFetch('/providers/selected'),
        BrainApp.apiFetch('/providers/vision'),
        BrainApp.apiFetch('/providers/delegate'),
      ]);
      if (provRes.ok) { const d = await provRes.json(); _providers = d.providers || []; }
      if (selRes.ok) { const d = await selRes.json(); _selectedId = d.provider ? d.provider.id : null; }
      if (visRes.ok) { const d = await visRes.json(); _visionId = d.provider ? d.provider.id : null; _visionSource = d.source || 'none'; }
      if (delRes.ok) { const d = await delRes.json(); _delegateId = d.provider ? d.provider.id : null; _delegateSource = d.source || 'none'; }
    } catch (e) { BrainApp.showToast('Failed to connect to backend', 'error'); }
    _render();
  }

  function _render() {
    const el = document.getElementById('providersList');
    if (!el) return;
    _renderRoleSelectors();
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
      b.addEventListener('click', () => _openWizard(Number(b.dataset.edit)));
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

  // ── Vision / Delegate role selectors ────────────────────────────────

  function _clearRoleSelectors() {
    const host = document.getElementById('roleSelectors');
    if (host) host.innerHTML = '';
  }

  function _renderRoleSelectors() {
    const host = document.getElementById('roleSelectors');
    if (!host) return;
    if (_providers.length === 0) { host.innerHTML = ''; return; }

    host.innerHTML = `
      <h4 class="section-head" style="margin-top:32px;">Vision</h4>
      <p class="panel-desc">Choose which provider Chalie uses to understand images. Only providers that passed the vision probe can be selected.</p>
      <div class="form-group">
        <select id="visionSelect">${_visionOptions()}</select>
      </div>
      <h4 class="section-head" style="margin-top:32px;">Delegate</h4>
      <p class="panel-desc">Subagent work (web_search, web_browse, …) runs on this provider. Defaults to your main provider.</p>
      <div class="form-group">
        <select id="delegateSelect">${_delegateOptions()}</select>
      </div>`;

    const visionSel = document.getElementById('visionSelect');
    if (visionSel && !visionSel.disabled) {
      visionSel.addEventListener('change', () => _setVision(visionSel.value));
    }
    const delegateSel = document.getElementById('delegateSelect');
    if (delegateSel) {
      delegateSel.addEventListener('change', () => _setDelegate(delegateSel.value));
    }
  }

  function _visionOptions() {
    const visionCapable = _providers.filter(p => p.supports_vision);
    if (visionCapable.length === 0) {
      return `<option value="" selected disabled>Disabled — no vision-capable providers</option>`;
    }
    // "Use main provider" only when the main provider itself supports vision —
    // picking it clears the explicit pin (PUT provider_id: null → source 'auto').
    const mainSupportsVision = _selectedId != null &&
      visionCapable.some(p => p.id === _selectedId);
    const useMainSelected = _visionSource === 'auto' || _visionSource === 'none';
    let html = '';
    if (mainSupportsVision) {
      html += `<option value=""${useMainSelected ? ' selected' : ''}>Use main provider</option>`;
    }
    html += visionCapable.map(p => {
      const sel = _visionSource === 'explicit' && p.id === _visionId;
      return `<option value="${p.id}"${sel ? ' selected' : ''}>${BrainApp.escapeHtml(p.name)}</option>`;
    }).join('');
    return html;
  }

  function _delegateOptions() {
    const useMainSelected = _delegateSource !== 'explicit';
    let html = `<option value=""${useMainSelected ? ' selected' : ''}>Use main provider</option>`;
    html += _providers.map(p => {
      const sel = _delegateSource === 'explicit' && p.id === _delegateId;
      return `<option value="${p.id}"${sel ? ' selected' : ''}>${BrainApp.escapeHtml(p.name)}</option>`;
    }).join('');
    return html;
  }

  async function _setVision(value) {
    const pid = value === '' ? null : Number(value);
    try {
      const res = await BrainApp.apiFetch('/providers/vision', { method: 'PUT', body: JSON.stringify({ provider_id: pid }) });
      if (res.ok) {
        const d = await res.json().catch(() => ({}));
        _visionId = d.provider ? d.provider.id : null;
        _visionSource = d.source || (pid == null ? 'auto' : 'explicit');
        _renderRoleSelectors();
        BrainApp.showToast('Vision provider updated', 'success');
      } else {
        const e = await res.json().catch(() => ({}));
        BrainApp.showToast(e.error || 'Failed', 'error');
        _renderRoleSelectors();
      }
    } catch {
      BrainApp.showToast('Failed to set vision provider', 'error');
      _renderRoleSelectors();
    }
  }

  async function _setDelegate(value) {
    const pid = value === '' ? null : Number(value);
    try {
      const res = await BrainApp.apiFetch('/providers/delegate', { method: 'PUT', body: JSON.stringify({ provider_id: pid }) });
      if (res.ok) {
        const d = await res.json().catch(() => ({}));
        _delegateId = d.provider ? d.provider.id : null;
        _delegateSource = d.source || (pid == null ? 'auto' : 'explicit');
        _renderRoleSelectors();
        BrainApp.showToast('Delegate provider updated', 'success');
      } else {
        const e = await res.json().catch(() => ({}));
        BrainApp.showToast(e.error || 'Failed', 'error');
        _renderRoleSelectors();
      }
    } catch {
      BrainApp.showToast('Failed to set delegate provider', 'error');
      _renderRoleSelectors();
    }
  }

  // ── Catalog ─────────────────────────────────────────────────────────

  async function _ensureCatalog() {
    if (_catalogLoaded) return;
    try {
      const res = await BrainApp.apiFetch('/providers/catalog');
      if (res.ok) { const d = await res.json(); _catalog = Array.isArray(d.catalog) ? d.catalog : []; }
    } catch { _catalog = []; }
    _catalogLoaded = true;
  }

  function _presetFor(provider) {
    // Reconstruct a wizard preset for an existing provider being edited: prefer
    // the catalog entry matching its platform + host, else synthesise one.
    const match = _catalog.find(c => c.platform === provider.platform &&
      (c.host || '') === (provider.host || ''));
    if (match) return match;
    return {
      id: 'edit', name: provider.name, platform: provider.platform,
      host: provider.host || '', needs_key: provider.platform !== 'ollama',
    };
  }

  // ── Wizard ──────────────────────────────────────────────────────────

  async function _openWizard(id) {
    _editingId = id;
    _editModel = '';
    _models = [];
    _lastFetchKey = '';
    await _ensureCatalog();

    if (id) {
      const p = _providers.find(x => x.id === id);
      if (!p) { _render(); return; }
      _editModel = p.model || '';
      _models = _editModel ? [_editModel] : [];
      _preset = _presetFor(p);
      _renderForm(p);
    } else {
      _preset = null;
      _renderPicker();
    }
  }

  function _renderPicker() {
    const root = document.getElementById('providersList');
    if (!root) return;
    _clearRoleSelectors();
    const tiles = [..._catalog, _CUSTOM_PRESET];
    root.innerHTML = `<div class="provider-wizard">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm back-btn" id="backToProviders">${Icons.Chevron(14)} Back</button>
        <h3>Choose a provider</h3>
      </div>
      <p class="wizard-hint">Pick your AI provider. We'll pre-fill the connection details and fetch its models for you.</p>
      <div class="provider-grid" id="providerGrid">
        ${tiles.map(t => `<button type="button" class="provider-tile" data-pid="${BrainApp.escapeHtml(t.id)}">
          <span class="provider-tile-avatar">${BrainApp.escapeHtml(_avatar(t.name))}</span>
          <span class="provider-tile-name">${BrainApp.escapeHtml(t.name)}</span>
          <span class="provider-tile-platform">${BrainApp.escapeHtml(t.platform === 'openai_compatible' ? 'OpenAI-compatible' : t.platform)}</span>
        </button>`).join('')}
      </div>
    </div>`;

    document.getElementById('backToProviders').addEventListener('click', _render);
    document.getElementById('providerGrid').addEventListener('click', (e) => {
      const tile = e.target.closest('[data-pid]');
      if (!tile) return;
      const pid = tile.dataset.pid;
      const preset = pid === 'custom' ? { ..._CUSTOM_PRESET } : _catalog.find(c => c.id === pid);
      if (!preset) return;
      _preset = preset;
      _editModel = '';
      _models = [];
      _lastFetchKey = '';
      _renderForm(null);
    });
  }

  function _avatar(name) {
    const s = (name || '').trim();
    return s ? s[0].toUpperCase() : '?';
  }

  function _needsHost() { return _preset.platform === 'ollama' || _preset.platform === 'openai_compatible'; }
  function _needsKey() { return !!_preset.needs_key; }

  function _renderForm(provider) {
    const root = document.getElementById('providersList');
    if (!root) return;
    _clearRoleSelectors();
    const editing = !!provider;
    const hostVal = provider ? (provider.host || '') : (_preset.host || '');
    const nameVal = provider ? provider.name : _preset.name;

    root.innerHTML = `<div class="provider-wizard">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm back-btn" id="backStep">${Icons.Chevron(14)} ${editing ? 'Back' : 'Providers'}</button>
        <h3>${editing ? 'Edit Provider' : `Set up ${BrainApp.escapeHtml(_preset.name)}`}</h3>
      </div>
      <form id="providerForm" class="wizard-form" autocomplete="off">
        <div class="form-group">
          <label for="pName">Name</label>
          <input type="text" id="pName" required value="${BrainApp.escapeHtml(nameVal)}">
        </div>
        <div class="form-group wizard-step" id="hostGroup" hidden>
          <label for="pHost">Host / Base URL</label>
          <input type="text" id="pHost" value="${BrainApp.escapeHtml(hostVal)}" placeholder="https://…">
        </div>
        <div class="form-group wizard-step" id="keyGroup" hidden>
          <label for="pKey">API Key</label>
          <input type="password" id="pKey" autocomplete="new-password" placeholder="${editing ? 'Leave blank to keep existing' : 'Paste your API key'}">
        </div>
        <div class="form-group wizard-step" id="modelGroup" hidden>
          <label for="pModel">Model</label>
          <select id="pModel"><option value="">Select model…</option></select>
          <span class="model-status" id="modelStatus"></span>
        </div>
        <div class="form-actions">
          <button type="button" class="btn btn-secondary" id="cancelProvBtn">Cancel</button>
          <button type="button" class="btn btn-secondary" id="testProvBtn">Test</button>
          <button type="submit" class="btn btn-primary" id="saveProvBtn" disabled>Save</button>
        </div>
      </form>
    </div>`;

    document.getElementById('backStep').addEventListener('click', () => editing ? _render() : _renderPicker());
    document.getElementById('cancelProvBtn').addEventListener('click', _render);
    document.getElementById('testProvBtn').addEventListener('click', _testConnection);
    document.getElementById('pHost')?.addEventListener('input', _onCredsInput);
    document.getElementById('pKey')?.addEventListener('input', _onCredsInput);
    document.getElementById('pModel').addEventListener('change', _refreshSaveState);
    document.getElementById('providerForm').addEventListener('submit', (e) => { e.preventDefault(); _saveProvider(); });

    _populateModels();
    _refreshVisibility();
    // Fetch immediately when credentials are already complete: a no-key local
    // provider (Ollama), or an edit where the host is set and no key is needed.
    if (_canFetch()) _fetchModels();
  }

  // ── Progressive reveal ──────────────────────────────────────────────

  function _hostValue() { return document.getElementById('pHost')?.value?.trim() || ''; }
  function _keyValue() { return document.getElementById('pKey')?.value?.trim() || ''; }
  function _hostReady() { return !_needsHost() || _hostValue() !== ''; }
  function _keyReady() { return !_needsKey() || _keyValue() !== ''; }
  // Live fetch needs real credentials. Editing keeps the existing model visible
  // without a fetch (blank key = keep current), so we never fire a doomed call.
  function _canFetch() { return _hostReady() && _keyReady(); }

  function _refreshVisibility() {
    const hostG = document.getElementById('hostGroup');
    const keyG = document.getElementById('keyGroup');
    const modelG = document.getElementById('modelGroup');
    if (!hostG) return;
    hostG.hidden = !_needsHost();
    keyG.hidden = !(_needsKey() && _hostReady());
    modelG.hidden = !(_canFetch() || _models.length > 0);
    _refreshSaveState();
  }

  function _refreshSaveState() {
    const btn = document.getElementById('saveProvBtn');
    if (btn) btn.disabled = !document.getElementById('pModel')?.value;
  }

  function _onCredsInput() {
    _refreshVisibility();
    _debouncedFetchModels();
  }

  function _populateModels() {
    const sel = document.getElementById('pModel');
    if (!sel) return;
    sel.innerHTML = '<option value="">Select model…</option>';
    for (const m of _models) {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      if (m === _editModel) opt.selected = true;
      sel.appendChild(opt);
    }
    if (_editModel && !_models.includes(_editModel)) {
      const opt = document.createElement('option');
      opt.value = _editModel; opt.textContent = _editModel; opt.selected = true;
      sel.appendChild(opt);
    }
    _refreshSaveState();
  }

  function _debouncedFetchModels() {
    clearTimeout(_modelFetchTimer);
    _modelFetchTimer = setTimeout(() => { if (_canFetch()) _fetchModels(); }, _MODEL_FETCH_DEBOUNCE_MS);
  }

  async function _fetchModels() {
    if (_modelsFetchInFlight) return;
    const creds = {};
    if (_needsHost()) creds.host = _hostValue();
    if (_needsKey()) creds.api_key = _keyValue();
    const fetchKey = `${_preset.platform}|${creds.host || ''}|${creds.api_key ? 'k' : ''}`;
    if (fetchKey === _lastFetchKey && _models.length) return;
    _lastFetchKey = fetchKey;

    _modelsFetchInFlight = true;
    const status = document.getElementById('modelStatus');
    if (status) { status.textContent = 'Loading models…'; status.className = 'model-status loading'; }
    try {
      const res = await BrainApp.apiFetch('/providers/list-models', {
        method: 'POST', body: JSON.stringify({ platform: _preset.platform, ...creds }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.models && !data.error) {
        _models = data.models.map(m => typeof m === 'string' ? m : m?.id || '').filter(Boolean);
        _populateModels();
        if (status) { status.textContent = `${_models.length} model${_models.length === 1 ? '' : 's'}`; status.className = 'model-status ok'; }
      } else {
        if (status) { status.textContent = data.error || 'Failed to fetch models'; status.className = 'model-status err'; }
        _lastFetchKey = '';  // allow retry after the user fixes credentials
      }
    } catch (e) {
      if (status) { status.textContent = 'Network error'; status.className = 'model-status err'; }
      _lastFetchKey = '';
    }
    _modelsFetchInFlight = false;
  }

  // ── Test / save ─────────────────────────────────────────────────────

  async function _testConnection() {
    const btn = document.getElementById('testProvBtn');
    btn.disabled = true; btn.textContent = 'Testing…';
    try {
      const body = { platform: _preset.platform, name: document.getElementById('pName').value.trim(), model: document.getElementById('pModel').value };
      if (_needsHost()) body.host = _hostValue();
      if (_needsKey()) body.api_key = _keyValue();
      if (_editingId) body.provider_id = _editingId;
      const res = await BrainApp.apiFetch('/providers/test', { method: 'POST', body: JSON.stringify(body) });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.success) BrainApp.showToast(data.message || 'Connection successful', 'success');
      else BrainApp.showToast(data.error || 'Test failed', 'error');
    } catch { BrainApp.showToast('Network error', 'error'); }
    btn.disabled = false; btn.textContent = 'Test';
  }

  async function _saveProvider() {
    const model = document.getElementById('pModel').value;
    if (!model) { BrainApp.showToast('Select a model first', 'error'); return; }
    const body = { platform: _preset.platform, name: document.getElementById('pName').value.trim(), model };
    if (_needsHost()) body.host = _hostValue();
    if (_needsKey()) { const k = _keyValue(); if (k) body.api_key = k; }

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

  // ── Delete ──────────────────────────────────────────────────────────

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

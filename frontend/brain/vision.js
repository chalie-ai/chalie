// Vision panel — choose which vision-capable provider understands images.
const PanelVision = (() => {
  let _root = null;
  let _providers = [];
  let _visionId = null;
  let _source = 'none';

  async function mount(root) {
    _root = root;
    root.innerHTML = `<div class="panel-header"><h2>Vision Provider</h2></div>
      <p class="panel-desc">Pick the provider Chalie uses to understand images. Only providers that passed the vision probe can be selected.</p>
      <div id="visionList" class="providers-list"><div class="loading">Loading…</div></div>`;
    await _load();
  }

  function unmount() { _root = null; }

  async function _load() {
    try {
      const [pRes, vRes] = await Promise.all([
        BrainApp.apiFetch('/providers'),
        BrainApp.apiFetch('/providers/vision'),
      ]);
      if (pRes.ok) { const d = await pRes.json(); _providers = d.providers || []; }
      if (vRes.ok) { const d = await vRes.json(); _visionId = d.provider ? d.provider.id : null; _source = d.source || 'none'; }
    } catch { BrainApp.showToast('Failed to connect to backend', 'error'); }
    _render();
  }

  function _render() {
    const el = document.getElementById('visionList');
    if (!el) return;
    if (_providers.length === 0) {
      el.innerHTML = `<div class="empty-state"><div class="empty-icon">${Icons.Eye(40)}</div><h3>No providers</h3><p>Add an LLM provider first.</p></div>`;
      return;
    }
    const autoNote = (_source === 'auto' && _visionId)
      ? `<p class="panel-desc">Auto-selected the active provider because it supports vision. Pick one explicitly to lock it in.</p>` : '';
    el.innerHTML = autoNote + _providers.map(p => {
      const supports = !!p.supports_vision;
      const isSel = supports && p.id === _visionId;
      return `<div class="provider-card${isSel ? ' selected' : ''}${supports ? '' : ' disabled'}" data-id="${p.id}">
        <label class="provider-radio">
          <input type="radio" name="vis_prov" value="${p.id}" ${isSel ? 'checked' : ''} ${supports ? '' : 'disabled'}>
          <span class="radio-dot"></span>
        </label>
        <div class="provider-info">
          <div class="provider-name">${BrainApp.escapeHtml(p.name)}</div>
          <div class="provider-meta">
            <span class="badge badge-${p.platform}">${BrainApp.escapeHtml(p.platform)}</span>
            <span>${BrainApp.escapeHtml(p.model || 'no model')}</span>
            ${supports ? `<span class="badge badge-success">Vision</span>` : `<span class="badge badge-muted" title="Did not pass the vision probe">No vision</span>`}
          </div>
        </div>
      </div>`;
    }).join('');

    el.querySelectorAll('input[name="vis_prov"]').forEach(r => {
      r.addEventListener('change', () => _setVision(Number(r.value)));
    });
  }

  async function _setVision(id) {
    try {
      const res = await BrainApp.apiFetch('/providers/vision', { method: 'PUT', body: JSON.stringify({ provider_id: id }) });
      if (res.ok) { _visionId = id; _source = 'explicit'; _render(); BrainApp.showToast('Vision provider set', 'success'); }
      else { const e = await res.json().catch(() => ({})); BrainApp.showToast(e.error || 'Failed', 'error'); }
    } catch { BrainApp.showToast('Failed to set vision provider', 'error'); }
  }

  return { mount, unmount };
})();

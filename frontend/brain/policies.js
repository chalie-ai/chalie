// Policies panel — per-context permission grid with presets.
// Labels and categories are derived from the AbilityRegistry via GET /api/policies.
const PanelPolicies = (() => {
  let _root = null;
  let _sub = 'chat';
  let _policies = {};
  let _blocked = [];
  let _loaded = false;
  let _blockedOpen = false;
  let _labels = {};
  let _categories = {};

  const CONTEXT_MAP = { chat: 'chat', subagent: 'subagent', background: 'subconscious', external: 'external_agent' };

  function mount(root, sub) {
    _root = root;
    _sub = sub || 'chat';
    _render();
  }

  function unmount() { _root = null; }

  function _render() {
    if (!_root) return;
    _root.innerHTML = `<div class="panel-header"><h2>Policies</h2></div>
    <div id="policiesContent"><div class="loading">Loading…</div></div>`;
    if (!_loaded) _load();
    else _renderPolicies();
  }

  async function _load() {
    try {
      const [polRes, blockRes] = await Promise.all([
        BrainApp.apiFetch('/api/policies'),
        BrainApp.apiFetch('/api/policies/blocked'),
      ]);
      if (polRes.ok) {
        const raw = await polRes.json();
        const actionKeyed = raw.policies || {};
        _policies = {};
        for (const [actionId, contexts] of Object.entries(actionKeyed)) {
          for (const [ctx, state] of Object.entries(contexts)) {
            if (!_policies[ctx]) _policies[ctx] = {};
            _policies[ctx][actionId] = state;
          }
        }
        _labels = {};
        _categories = {};
        for (const entry of raw.meta || []) {
          _labels[entry.action_id] = entry.label;
          if (!_categories[entry.category]) _categories[entry.category] = [];
          _categories[entry.category].push(entry.action_id);
        }
      }
      if (blockRes.ok) { const d = await blockRes.json(); _blocked = d.entries || []; }
      _loaded = true;
    } catch { BrainApp.showToast('Failed to load policies', 'error'); }
    _renderPolicies();
  }

  function _renderPolicies() {
    const el = document.getElementById('policiesContent');
    if (!el) return;
    const ctx = CONTEXT_MAP[_sub] || 'chat';
    const rules = _policies[ctx] || {};

    let html = `<div class="policies-grid">`;
    for (const [catName, actionIds] of Object.entries(_categories)) {
      if (actionIds.length === 0) continue;
      html += `<div class="policy-category">
        <h4 class="section-head">${BrainApp.escapeHtml(catName)}</h4>
        ${actionIds.map(actionId => {
          const label = _labels[actionId] || actionId;
          const currentVal = rules[actionId] || 'ask';
          return `<div class="policy-rule">
            <span class="policy-label">${label}</span>
            <div class="segmented" data-action="${actionId}">
              ${['allow', 'ask', 'deny'].map(v => `<button class="seg-btn${v === currentVal ? ' active' : ''}" data-val="${v}">${v.charAt(0).toUpperCase() + v.slice(1)}</button>`).join('')}
            </div>
          </div>`;
        }).join('')}
      </div>`;
    }
    html += `</div>`;

    if (_blocked.length > 0) {
      html += `<div class="blocked-section">
        <button class="blocked-toggle" id="blockedToggle">Blocked Actions Log ${_blockedOpen ? '▴' : '▾'}</button>
        ${_blockedOpen ? `<div class="blocked-list">${_blocked.map(b => `<div class="blocked-item">
          <span class="badge badge-danger">${BrainApp.escapeHtml(b.action_id || '')}</span>
          <span>${BrainApp.escapeHtml(b.context || '')}</span>
          <span class="blocked-time">${BrainApp.formatDate(b.created_at)}</span>
        </div>`).join('')}</div>` : ''}
      </div>`;
    }

    el.innerHTML = html;

    el.querySelectorAll('.segmented').forEach(seg => {
      seg.addEventListener('click', (e) => {
        const btn = e.target.closest('.seg-btn');
        if (!btn) return;
        const actionId = seg.dataset.action;
        const value = btn.dataset.val;
        seg.querySelectorAll('.seg-btn').forEach(b => b.classList.toggle('active', b === btn));
        _updatePolicy(ctx, actionId, value);
      });
    });

    document.getElementById('blockedToggle')?.addEventListener('click', () => {
      _blockedOpen = !_blockedOpen;
      _renderPolicies();
    });
  }

  async function _updatePolicy(context, actionId, value) {
    try {
      const res = await BrainApp.apiFetch('/api/policies', {
        method: 'PUT',
        body: JSON.stringify({ rules: [{ action_id: actionId, context, state: value }] }),
      });
      if (res.ok) {
        if (!_policies[context]) _policies[context] = {};
        _policies[context][actionId] = value;
        BrainApp.showToast(`${actionId} → ${value}`, 'success');
      } else BrainApp.showToast('Failed to update policy', 'error');
    } catch { BrainApp.showToast('Network error', 'error'); }
  }

  return { mount, unmount };
})();

// Policies panel — flat (channel, permission, setting) rows from GET /api/policies.
// No client-side pivot, no meta matrix (D4). Category = permission prefix.
const PanelPolicies = (() => {
  let _root = null;
  let _sub = 'chat';
  let _rows = [];          // [{channel, permission, setting}]
  let _blocked = [];
  let _loaded = false;
  let _blockedOpen = false;

  const CONTEXT_MAP = { chat: 'chat', subagent: 'subagent', background: 'subconscious', external: 'external_agent' };

  function mount(root, sub) { _root = root; _sub = sub || 'chat'; _render(); }
  function unmount() { _root = null; }

  function _render() {
    if (!_root) return;
    _root.innerHTML = `<div class="panel-header"><h2>Policies</h2></div>
    <div id="policiesContent"><div class="loading">Loading…</div></div>`;
    if (!_loaded) _load(); else _renderPolicies();
  }

  async function _load() {
    try {
      const [polRes, blockRes] = await Promise.all([
        BrainApp.apiFetch('/api/policies'),
        BrainApp.apiFetch('/api/policies/blocked'),
      ]);
      if (polRes.ok) { const raw = await polRes.json(); _rows = raw.policies || []; }
      if (blockRes.ok) { const d = await blockRes.json(); _blocked = d.entries || []; }
      _loaded = true;
    } catch { BrainApp.showToast('Failed to load policies', 'error'); }
    _renderPolicies();
  }

  function _renderPolicies() {
    const el = document.getElementById('policiesContent');
    if (!el) return;
    const channel = CONTEXT_MAP[_sub] || 'chat';

    // Group this channel's rows by category. MCP rows carry a server-title
    // `group` (which also flags the group as MCP); native rows group by the
    // permission prefix. Both carry a humanized `label` tagged backend-side.
    const byCat = {};
    for (const r of _rows) {
      if (r.channel !== channel) continue;
      const isMcp = !!r.group;
      const cat = r.group || r.permission.split('.')[0];
      const label = r.label || r.permission;
      const bucket = byCat[cat] || (byCat[cat] = { isMcp, rows: [] });
      bucket.rows.push({ r, label });
    }

    let html = `<div class="policies-grid">`;
    for (const cat of Object.keys(byCat).sort((a, b) => a.localeCompare(b))) {
      const { isMcp, rows } = byCat[cat];
      const pill = isMcp ? `<span class="badge badge-cyan">MCP</span> ` : '';
      html += `<div class="policy-category">
        <h4 class="section-head">${pill}${BrainApp.escapeHtml(cat)}</h4>
        ${rows.map(({ r, label }) => `<div class="policy-rule">
          <span class="policy-label">${BrainApp.escapeHtml(label)}</span>
          <div class="segmented" data-permission="${BrainApp.escapeHtml(r.permission)}">
            ${['allow', 'ask', 'deny'].map(v => `<button class="seg-btn${v === r.setting ? ' active' : ''}" data-val="${v}">${v.charAt(0).toUpperCase() + v.slice(1)}</button>`).join('')}
          </div>
        </div>`).join('')}
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
        const permission = seg.dataset.permission;
        const value = btn.dataset.val;
        seg.querySelectorAll('.seg-btn').forEach(b => b.classList.toggle('active', b === btn));
        _updatePolicy(channel, permission, value);
      });
    });

    document.getElementById('blockedToggle')?.addEventListener('click', () => {
      _blockedOpen = !_blockedOpen;
      _renderPolicies();
    });
  }

  async function _updatePolicy(channel, permission, setting) {
    try {
      const res = await BrainApp.apiFetch('/api/policies', {
        method: 'PUT',
        body: JSON.stringify({ channel, permission, setting }),
      });
      if (res.ok) {
        const row = _rows.find(r => r.channel === channel && r.permission === permission);
        if (row) row.setting = setting;
        BrainApp.showToast(`${permission} → ${setting}`, 'success');
      } else BrainApp.showToast('Failed to update policy', 'error');
    } catch { BrainApp.showToast('Network error', 'error'); }
  }

  return { mount, unmount };
})();

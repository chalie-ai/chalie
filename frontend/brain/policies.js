// Policies panel — per-context permission grid with presets.
const PanelPolicies = (() => {
  let _root = null;
  let _sub = 'chat';
  let _policies = {};
  let _blocked = [];
  let _loaded = false;
  let _blockedOpen = false;

  const _POLICY_ACTION_LABELS = {
    'browser.search': 'Web search',
    'browser.navigate': 'Navigate to URL',
    'browser.interact': 'Interact with page',
    'browser.read': 'Read page content',
    'calendar.read': 'Read calendar',
    'calendar.create': 'Create events',
    'calendar.update': 'Update events',
    'calendar.delete': 'Delete events',
    'code_eval.evaluate': 'Run sandboxed code',
    'contacts.search': 'Search contacts',
    'contacts.add': 'Add contacts',
    'contacts.update': 'Update contacts',
    'document.read': 'Read documents',
    'document.search': 'Search documents',
    'email.search': 'Search email',
    'email.read': 'Read email',
    'email.send': 'Send email',
    'email.reply': 'Reply to email',
    'home.get_state': 'Get device state',
    'home.control': 'Control devices',
    'home.subscribe_events': 'Subscribe events',
    'list.read': 'Read lists',
    'list.create': 'Create lists',
    'list.update': 'Update lists',
    'list.delete': 'Delete lists',
    'memory.recall': 'Recall memory',
    'memory.store': 'Store memory',
    'memory.forget': 'Forget memory',
    'news.search': 'Search news',
    'schedule.read': 'Read schedules',
    'schedule.create': 'Create schedules',
    'schedule.update': 'Update schedules',
    'schedule.delete': 'Delete schedules',
    'search.web': 'Web search (tool)',
    'subagent.spawn': 'Spawn subagent',
    'weather.current': 'Current weather',
    'weather.forecast': 'Weather forecast',
  };

  const _POLICY_CATEGORIES = {
    'Browser': ['browser.search', 'browser.navigate', 'browser.interact', 'browser.read'],
    'Calendar & Email': ['calendar.read', 'calendar.create', 'calendar.update', 'calendar.delete', 'email.search', 'email.read', 'email.send', 'email.reply'],
    'Code execution': ['code_eval.evaluate'],
    'Contacts': ['contacts.search', 'contacts.add', 'contacts.update'],
    'Documents & Lists': ['document.read', 'document.search', 'list.read', 'list.create', 'list.update', 'list.delete'],
    'Home': ['home.get_state', 'home.control', 'home.subscribe_events'],
    'Memory': ['memory.recall', 'memory.store', 'memory.forget'],
    'News & Weather': ['news.search', 'weather.current', 'weather.forecast'],
    'Scheduling': ['schedule.read', 'schedule.create', 'schedule.update', 'schedule.delete'],
    'Search': ['search.web'],
    'Subagent': ['subagent.spawn'],
  };

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
      if (polRes.ok) _policies = await polRes.json();
      if (blockRes.ok) { const d = await blockRes.json(); _blocked = d.blocked || d.actions || []; }
      _loaded = true;
    } catch { BrainApp.showToast('Failed to load policies', 'error'); }
    _renderPolicies();
  }

  function _renderPolicies() {
    const el = document.getElementById('policiesContent');
    if (!el) return;
    const ctx = CONTEXT_MAP[_sub] || 'chat';
    const rules = _policies[ctx] || _policies.rules?.[ctx] || {};

    let html = `<div class="policies-grid">`;
    for (const [catName, actionIds] of Object.entries(_POLICY_CATEGORIES)) {
      const catRules = actionIds.filter(a => _POLICY_ACTION_LABELS[a]);
      if (catRules.length === 0) continue;
      html += `<div class="policy-category">
        <h4 class="section-head">${BrainApp.escapeHtml(catName)}</h4>
        ${catRules.map(actionId => {
          const label = _POLICY_ACTION_LABELS[actionId] || actionId;
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
          <span class="badge badge-danger">${BrainApp.escapeHtml(b.action || '')}</span>
          <span>${BrainApp.escapeHtml(b.context || '')}</span>
          <span class="blocked-time">${BrainApp.escapeHtml(b.time || '')}</span>
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
        body: JSON.stringify({ context, action_id: actionId, value }),
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

// Policies panel — per-context permission grid with presets.
const PanelPolicies = (() => {
  let _root = null;
  let _sub = 'chat';
  let _policies = {};
  let _blocked = [];
  let _loaded = false;
  let _blockedOpen = false;

  const _POLICY_ACTION_LABELS = {
    'browser.interact': 'Interact with page',
    'browser.monitor': 'Monitor page',
    'browser.render': 'Render page',
    'browser.screenshot': 'Take screenshot',
    'calendar.get_event': 'Get event',
    'calendar.list_events': 'List events',
    'calendar.update_event': 'Update event',
    'code_eval': 'Run sandboxed code',
    'contacts.get': 'Get contact',
    'contacts.list': 'List contacts',
    'document.create': 'Create document',
    'document.delete': 'Delete document',
    'document.list': 'List documents',
    'document.restore': 'Restore document',
    'document.search': 'Search documents',
    'document.upload': 'Upload document',
    'document.view': 'View document',
    'email.draft': 'Draft email',
    'email.forward': 'Forward email',
    'email.manage': 'Manage email',
    'email.read': 'Read email',
    'email.reply': 'Reply to email',
    'email.search': 'Search email',
    'email.send': 'Send email',
    'find_tools': 'Find tools',
    'home.control': 'Control devices',
    'home.get_state': 'Get device state',
    'home.list_automations': 'List automations',
    'home.list_devices': 'List devices',
    'home.subscribe_events': 'Subscribe events',
    'home.trigger_automation': 'Trigger automation',
    'list.add': 'Add list items',
    'list.check': 'Check list items',
    'list.clear': 'Clear list',
    'list.create': 'Create list',
    'list.delete': 'Delete list',
    'list.list_all': 'List all lists',
    'list.remove': 'Remove list items',
    'list.rename': 'Rename list',
    'list.view': 'View list',
    'memory.forget': 'Forget memory',
    'memory.recall': 'Recall memory',
    'memory.reflect': 'Reflect on memory',
    'memory.store': 'Store memory',
    'news': 'Search news',
    'read': 'Read content',
    'review_tool_calls': 'Review tool calls',
    'review_transcript': 'Review transcript',
    'save_graph': 'Save to knowledge graph',
    'save_pattern': 'Save pattern',
    'schedule.cancel': 'Cancel schedule',
    'schedule.create': 'Create schedule',
    'schedule.list': 'List schedules',
    'schedule.search': 'Search schedules',
    'search': 'Web search',
    'subagent': 'Spawn subagent',
    'timer': 'Set timer',
    'weather': 'Weather lookup',
    'programming_docs_search': 'Search programming docs',
  };

  const _POLICY_CATEGORIES = {
    'Browser': ['browser.interact', 'browser.monitor', 'browser.render', 'browser.screenshot'],
    'Calendar': ['calendar.get_event', 'calendar.list_events', 'calendar.update_event'],
    'Email': ['email.search', 'email.read', 'email.send', 'email.reply', 'email.draft', 'email.forward', 'email.manage'],
    'Code': ['code_eval'],
    'Contacts': ['contacts.get', 'contacts.list'],
    'Documents': ['document.list', 'document.view', 'document.search', 'document.create', 'document.upload', 'document.delete', 'document.restore'],
    'Lists': ['list.list_all', 'list.view', 'list.create', 'list.add', 'list.check', 'list.remove', 'list.clear', 'list.rename', 'list.delete'],
    'Home': ['home.get_state', 'home.list_devices', 'home.control', 'home.list_automations', 'home.trigger_automation', 'home.subscribe_events'],
    'Memory': ['memory.recall', 'memory.store', 'memory.forget', 'memory.reflect'],
    'News & Weather': ['news', 'weather'],
    'Scheduling': ['schedule.list', 'schedule.search', 'schedule.create', 'schedule.cancel'],
    'Search & Tools': ['search', 'find_tools', 'read', 'programming_docs_search'],
    'Subagent': ['subagent'],
    'System': ['timer', 'review_tool_calls', 'review_transcript', 'save_graph', 'save_pattern'],
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
          <span class="blocked-time">${BrainApp.formatDate(b.time)}</span>
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

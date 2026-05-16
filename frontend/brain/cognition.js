// Cognition panel — 7 sub-views: memory, tools, working, world, personality, errors, usage.
const PanelCognition = (() => {
  let _root = null;
  let _sub = 'memory';
  let _loaded = {};
  let _data = {};

  function mount(root, sub) {
    _root = root;
    _sub = sub || 'memory';
    _render();
  }

  function unmount() { _root = null; }

  function _render() {
    if (!_root) return;
    _root.innerHTML = `<div class="panel-header">
      <h2>Cognition</h2>
      <div class="panel-header-actions">
        <span class="obs-timestamp" id="obsTimestamp"></span>
        <button class="btn btn-sm btn-secondary" id="obsRefreshBtn">${Icons.Refresh(14)} Refresh</button>
      </div>
    </div>
    <div id="cognitionContent"></div>`;

    document.getElementById('obsRefreshBtn').addEventListener('click', () => { _loaded[_sub] = false; _loadSub(); });
    _loadSub();
  }

  async function _loadSub() {
    const el = document.getElementById('cognitionContent');
    if (!el) return;

    if (!_loaded[_sub]) {
      el.innerHTML = '<div class="loading">Loading…</div>';
      try {
        switch (_sub) {
          case 'memory': await _fetchMemory(); break;
          case 'tools': await _fetchTools(); break;
          case 'working': await _fetchWorking(); break;
          case 'world': await _fetchWorld(); break;
          case 'personality': await _fetchPersonality(); break;
          case 'errors': await _fetchErrors(); break;
          case 'usage': await _fetchUsage(); break;
        }
        _loaded[_sub] = true;
      } catch (e) { el.innerHTML = `<div class="empty-state"><p>Failed to load data.</p></div>`; return; }
    }
    _renderSub(el);
  }

  // ── Memory ──
  let _memorySource = 'episodes';
  let _memorySearch = '';
  let _memoryRecords = [];
  let _memoryOffset = 0;
  let _memoryHasMore = false;

  async function _fetchMemory() {
    const params = new URLSearchParams({ source: _memorySource, limit: '50', offset: String(_memoryOffset) });
    if (_memorySearch) params.set('q', _memorySearch);
    const res = await BrainApp.apiFetch(`/system/observability/records?${params}`);
    if (!res.ok) throw new Error('fetch failed');
    const data = await res.json();
    _memoryRecords = data.records || [];
    _memoryHasMore = data.has_more || false;
    _data.memoryTs = data.timestamp;
  }

  // ── Tools ──
  let _tools = [];
  async function _fetchTools() {
    const res = await BrainApp.apiFetch('/system/observability/tools');
    if (!res.ok) throw new Error('fetch failed');
    const data = await res.json();
    _tools = data.tools || [];
  }

  // ── Working On ──
  let _tasks = [];
  async function _fetchWorking() {
    const res = await BrainApp.apiFetch('/system/observability/records?source=tasks');
    if (!res.ok) throw new Error('fetch failed');
    const data = await res.json();
    _tasks = data.records || [];
  }

  // ── World State ──
  let _worldState = '';
  async function _fetchWorld() {
    const res = await BrainApp.apiFetch('/system/observability/world-state');
    if (!res.ok) throw new Error('fetch failed');
    const data = await res.json();
    _worldState = typeof data.state === 'string' ? data.state : JSON.stringify(data.state, null, 2);
  }

  // ── Personality ──
  let _personality = { warmth: 0, mood: 0, expressiveness: 0, curiosity: 0, humor: 0 };
  async function _fetchPersonality() {
    const res = await BrainApp.apiFetch('/settings/personality');
    if (!res.ok) throw new Error('fetch failed');
    _personality = await res.json();
  }

  // ── Errors ──
  let _errors = [];
  async function _fetchErrors() {
    const res = await BrainApp.apiFetch('/system/observability/errors');
    if (!res.ok) throw new Error('fetch failed');
    const data = await res.json();
    _errors = data.errors || [];
  }

  // ── Usage ──
  let _usage = {};
  let _usageWindow = 'day';
  async function _fetchUsage() {
    const res = await BrainApp.apiFetch(`/system/observability/token-usage?window=${_usageWindow}`);
    if (!res.ok) throw new Error('fetch failed');
    _usage = await res.json();
  }

  // ── Sub Renderers ──
  function _renderSub(el) {
    switch (_sub) {
      case 'memory': _renderMemory(el); break;
      case 'tools': _renderTools(el); break;
      case 'working': _renderWorking(el); break;
      case 'world': _renderWorld(el); break;
      case 'personality': _renderPersonality(el); break;
      case 'errors': _renderErrors(el); break;
      case 'usage': _renderUsage(el); break;
    }
    const ts = document.getElementById('obsTimestamp');
    if (ts && _data.memoryTs) ts.textContent = `Updated ${_data.memoryTs}`;
  }

  function _renderMemory(el) {
    el.innerHTML = `<div class="records-controls">
      <div class="filter-tabs" id="memSourceTabs">
        ${['episodes', 'user', 'system'].map(s => `<button class="filter-tab${s === _memorySource ? ' active' : ''}" data-src="${s}">${s.charAt(0).toUpperCase() + s.slice(1)}</button>`).join('')}
      </div>
      <input type="text" class="search-input" id="memSearch" placeholder="Search…" value="${BrainApp.escapeHtml(_memorySearch)}">
    </div>
    ${_memoryRecords.length === 0 ? '<div class="empty-state"><p>No records found.</p></div>' : `
    <table class="records-table">
      <thead><tr><th>Created</th><th>Last Accessed</th><th>Key</th><th>Value</th></tr></thead>
      <tbody>${_memoryRecords.map(r => `<tr>
        <td>${BrainApp.escapeHtml(r.created || '')}</td>
        <td>${BrainApp.escapeHtml(r.last_accessed || '')}</td>
        <td class="key-cell">${BrainApp.escapeHtml(r.key || '')}</td>
        <td class="val-cell">${BrainApp.escapeHtml(r.value || '')}</td>
      </tr>`).join('')}</tbody>
    </table>
    ${_memoryHasMore ? '<div class="records-footer"><button class="btn btn-secondary" id="memLoadMore">Load more</button></div>' : ''}`}`;

    el.querySelector('#memSourceTabs')?.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-src]');
      if (!btn) return;
      _memorySource = btn.dataset.src;
      _memoryOffset = 0; _loaded.memory = false; _loadSub();
    });
    el.querySelector('#memSearch')?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { _memorySearch = e.target.value; _memoryOffset = 0; _loaded.memory = false; _loadSub(); }
    });
    el.querySelector('#memLoadMore')?.addEventListener('click', () => {
      _memoryOffset += 50; _loaded.memory = false; _loadSub();
    });
  }

  function _renderTools(el) {
    if (_tools.length === 0) { el.innerHTML = '<div class="empty-state"><p>No tools loaded.</p></div>'; return; }
    el.innerHTML = `<table class="records-table">
      <thead><tr><th>Name</th><th>Type</th><th>Calls</th><th>Last Used</th></tr></thead>
      <tbody>${_tools.map(t => `<tr>
        <td class="key-cell">${BrainApp.escapeHtml(t.name || '')}</td>
        <td><span class="badge badge-muted">${BrainApp.escapeHtml(t.type || 'tool')}</span></td>
        <td>${t.call_count ?? '—'}</td>
        <td>${BrainApp.escapeHtml(t.last_used || '—')}</td>
      </tr>`).join('')}</tbody>
    </table>`;
  }

  function _renderWorking(el) {
    if (_tasks.length === 0) { el.innerHTML = '<div class="empty-state"><p>No active tasks.</p></div>'; return; }
    el.innerHTML = `<div class="task-cards">${_tasks.map(t => `<div class="task-card">
      <div class="task-title">${BrainApp.escapeHtml(t.key || t.title || '')}</div>
      <div class="task-meta">${BrainApp.escapeHtml(t.value || t.meta || '')}</div>
    </div>`).join('')}</div>`;
  }

  function _renderWorld(el) {
    el.innerHTML = `<div class="code-block"><pre><code>${BrainApp.escapeHtml(_worldState)}</code></pre></div>`;
  }

  function _renderPersonality(el) {
    const sliders = [
      { key: 'warmth', left: 'Cool', right: 'Warm' },
      { key: 'mood', left: 'Level-headed', right: 'Moody' },
      { key: 'expressiveness', left: 'Reserved', right: 'Vocal' },
      { key: 'curiosity', left: 'Matter-of-fact', right: 'Inquisitive' },
      { key: 'humor', left: 'Dry', right: 'Playful' },
    ];
    el.innerHTML = `<p class="panel-desc">Adjust how Chalie communicates. Changes take effect on the next message.</p>
    <div class="personality-grid">${sliders.map(s => `<div class="personality-row">
      <span class="pole pole-left">${s.left}</span>
      <div class="personality-track">
        <label class="personality-label">${s.key.charAt(0).toUpperCase() + s.key.slice(1)}</label>
        <input type="range" min="-2" max="2" step="1" value="${_personality[s.key] || 0}" data-key="${s.key}" class="personality-range">
      </div>
      <span class="pole pole-right">${s.right}</span>
    </div>`).join('')}</div>
    <div class="personality-actions">
      <button class="btn btn-primary" id="savePersonalityBtn">Save</button>
    </div>`;

    document.getElementById('savePersonalityBtn')?.addEventListener('click', async () => {
      const body = {};
      el.querySelectorAll('.personality-range').forEach(r => { body[r.dataset.key] = Number(r.value); });
      try {
        const res = await BrainApp.apiFetch('/settings/personality', { method: 'PUT', body: JSON.stringify(body) });
        if (res.ok) { _personality = body; BrainApp.showToast('Personality saved', 'success'); }
        else BrainApp.showToast('Failed to save', 'error');
      } catch { BrainApp.showToast('Network error', 'error'); }
    });
  }

  function _renderErrors(el) {
    if (_errors.length === 0) { el.innerHTML = '<div class="empty-state"><p>No recent errors.</p></div>'; return; }
    el.innerHTML = `<div class="error-list">${_errors.map(e => `<div class="error-item">
      <span class="error-time">${BrainApp.escapeHtml(e.time || e.timestamp || '')}</span>
      <span class="error-msg">${BrainApp.escapeHtml(e.message || '')}</span>
    </div>`).join('')}</div>`;
  }

  function _renderUsage(el) {
    const windows = ['hour', 'day', 'week', 'month', 'lifetime'];
    const summary = _usage.summary || [];
    const chart = _usage.chart || [];

    el.innerHTML = `<div class="filter-tabs" id="usageWindowTabs">
      ${windows.map(w => `<button class="filter-tab${w === _usageWindow ? ' active' : ''}" data-win="${w}">${w.charAt(0).toUpperCase() + w.slice(1)}</button>`).join('')}
    </div>
    <div class="stat-grid">${summary.map(s => `<div class="stat-card">
      <div class="stat-value">${BrainApp.escapeHtml(String(s.value ?? '—'))}</div>
      <div class="stat-label">${BrainApp.escapeHtml(s.label || '')}</div>
      ${s.sub ? `<div class="stat-sub">${BrainApp.escapeHtml(s.sub)}</div>` : ''}
    </div>`).join('')}</div>
    ${chart.length > 0 ? _renderChart(chart) : ''}`;

    el.querySelector('#usageWindowTabs')?.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-win]');
      if (!btn) return;
      _usageWindow = btn.dataset.win;
      _loaded.usage = false; _loadSub();
    });
  }

  function _renderChart(chart) {
    const maxVal = Math.max(...chart.map(d => (d.local || 0) + (d.cloud || 0)), 1);
    const barW = Math.max(20, Math.floor(600 / chart.length) - 4);
    const h = 160;
    const bars = chart.map((d, i) => {
      const localH = ((d.local || 0) / maxVal) * h;
      const cloudH = ((d.cloud || 0) / maxVal) * h;
      const x = i * (barW + 4);
      return `<g>
        <rect x="${x}" y="${h - localH - cloudH}" width="${barW}" height="${cloudH}" class="bar-cloud"/>
        <rect x="${x}" y="${h - localH}" width="${barW}" height="${localH}" class="bar-local"/>
        <text x="${x + barW / 2}" y="${h + 14}" class="bar-label">${BrainApp.escapeHtml(String(d.hour || d.label || ''))}</text>
      </g>`;
    }).join('');
    const svgW = chart.length * (barW + 4);
    return `<div class="chart-wrap"><svg class="usage-chart" viewBox="0 0 ${svgW} ${h + 20}" preserveAspectRatio="none">${bars}</svg></div>
    <div class="chart-legend"><span class="legend-local">Local</span><span class="legend-cloud">Cloud</span></div>`;
  }

  return { mount, unmount };
})();

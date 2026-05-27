// Cognition panel — 6 sub-views: memory, tools, world, personality, errors, usage.
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
    _root.innerHTML = `<div class="panel-header"><h2>Cognition</h2></div>
    <div id="cognitionContent"></div>`;
    _loadSub();
  }

  async function _loadSub() {
    const el = document.getElementById('cognitionContent');
    if (!el) return;
    const targetSub = _sub;

    if (!_loaded[targetSub]) {
      el.innerHTML = '<div class="loading">Loading…</div>';
      try {
        switch (targetSub) {
          case 'memory': await _fetchMemory(); break;
          case 'tools': await _fetchTools(); break;
          case 'world': await _fetchWorld(); break;
          case 'personality': await _fetchPersonality(); break;
          case 'errors': await _fetchErrors(); break;
          case 'usage': await _fetchUsage(); break;
        }
        if (_sub !== targetSub) return;
        _loaded[targetSub] = true;
      } catch (e) {
        if (_sub !== targetSub) return;
        el.innerHTML = `<div class="empty-state"><p>Failed to load data.</p></div>`; return;
      }
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
    _memoryRecords = data.rows || [];
    _memoryHasMore = data.has_more || false;
    _data.memoryTs = data.generated_at;
  }

  // ── Tools ──
  let _tools = [];
  async function _fetchTools() {
    const res = await BrainApp.apiFetch('/system/observability/tools');
    if (!res.ok) throw new Error('fetch failed');
    const data = await res.json();
    _tools = data.tools || [];
  }


  // ── World State ──
  let _worldState = {};
  let _worldViewMode = 'formatted';
  async function _fetchWorld() {
    const res = await BrainApp.apiFetch('/system/observability/world-state');
    if (!res.ok) throw new Error('fetch failed');
    _worldState = await res.json();
  }

  function _mdToHtml(md) {
    return BrainApp.escapeHtml(md)
      .replace(/^### (.+)$/gm, '<h3>$1</h3>')
      .replace(/^## (.+)$/gm, '<h3>$1</h3>')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/^\* (.+)$/gm, '<li>$1</li>')
      .replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`)
      .replace(/\[([^\]]+)\]/g, '<span class="world-tag">$1</span>')
      .replace(/\n{2,}/g, '<br>')
      .replace(/\n/g, '\n');
  }

  // ── Personality ──
  let _personality = { warmth: 0, mood: 0, expressiveness: 0, curiosity: 0, humor: 0 };
  let _personalityVoice = '';
  async function _fetchPersonality() {
    const res = await BrainApp.apiFetch('/settings/personality');
    if (!res.ok) throw new Error('fetch failed');
    const data = await res.json();
    const t = data.tuple || [0, 0, 0, 0, 0];
    _personality = { warmth: t[0], mood: t[1], expressiveness: t[2], curiosity: t[3], humor: t[4] };
    _personalityVoice = data.voice || '';
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
      case 'world': _renderWorld(el); break;
      case 'personality': _renderPersonality(el); break;
      case 'errors': _renderErrors(el); break;
      case 'usage': _renderUsage(el); break;
    }
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
      <thead><tr><th>Created</th><th>Last Accessed</th><th>${_memorySource === 'episodes' ? 'Location' : 'Key'}</th><th>Value</th></tr></thead>
      <tbody>${_memoryRecords.map(r => `<tr>
        <td>${BrainApp.formatDate(r.created)}</td>
        <td>${BrainApp.formatDate(r.last_accessed)}</td>
        <td class="key-cell">${BrainApp.escapeHtml(_memorySource === 'episodes' ? (r.location || '') : (r.key || ''))}</td>
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
      <thead><tr><th>Name</th><th>Calls</th><th>Last Used</th></tr></thead>
      <tbody>${_tools.map(t => `<tr>
        <td class="key-cell">${BrainApp.escapeHtml(t.tool_name || '')}</td>
        <td>${t.count ?? '—'}</td>
        <td>${BrainApp.formatDate(t.last_used_at) || '—'}</td>
      </tr>`).join('')}</tbody>
    </table>`;
  }


  function _renderWorld(el) {
    const inputs = _worldState.inputs || {};
    const telemetry = inputs.telemetry || {};
    const schedule = inputs.schedule || [];
    const signals = inputs.signals || {};
    const bgProcs = inputs.bg_processes || [];

    const deviceInfo = telemetry.device || {};
    const battery = telemetry.battery || {};
    const location = telemetry.location_name || '';
    const localTime = telemetry.local_time || '';
    const timezone = telemetry.timezone || '';
    const prefs = telemetry.preferences || {};

    const signalEntries = Object.entries(signals);
    const pendingSched = schedule.filter(s => s.status === 'pending');

    el.innerHTML = `<div class="world-state-grid">
      <div class="world-section">
        <h4>Device & Environment</h4>
        <table class="records-table">
          <tbody>
            <tr><td class="key-cell">Location</td><td>${BrainApp.escapeHtml(location || '—')}</td></tr>
            <tr><td class="key-cell">Local Time</td><td>${BrainApp.escapeHtml(localTime)} (${BrainApp.escapeHtml(timezone)})</td></tr>
            <tr><td class="key-cell">Device</td><td>${BrainApp.escapeHtml(deviceInfo.class || '—')} · ${BrainApp.escapeHtml(deviceInfo.platform || '')} · ${deviceInfo.screen_w || '?'}×${deviceInfo.screen_h || '?'}</td></tr>
            <tr><td class="key-cell">Battery</td><td>${Math.round((battery.level || 0) * 100)}%${battery.charging ? ' ⚡ charging' : ''}</td></tr>
            <tr><td class="key-cell">Network</td><td>${BrainApp.escapeHtml(telemetry.connection || '—')}</td></tr>
            <tr><td class="key-cell">Theme</td><td>${BrainApp.escapeHtml(prefs.color_scheme || '—')}</td></tr>
          </tbody>
        </table>
      </div>
      ${pendingSched.length > 0 ? `<div class="world-section">
        <h4>Pending Schedules</h4>
        <table class="records-table">
          <thead><tr><th>Message</th><th>Due</th><th>Recurrence</th></tr></thead>
          <tbody>${pendingSched.map(s => `<tr>
            <td class="key-cell">${BrainApp.escapeHtml(s.message || '')}</td>
            <td>${BrainApp.formatDate(s.due_at)}</td>
            <td>${BrainApp.escapeHtml(s.recurrence || '—')}</td>
          </tr>`).join('')}</tbody>
        </table>
      </div>` : ''}
      ${signalEntries.length > 0 ? `<div class="world-section">
        <h4>Active Signals</h4>
        <table class="records-table">
          <thead><tr><th>Signal</th><th>Label</th></tr></thead>
          <tbody>${signalEntries.map(([k, v]) => `<tr>
            <td class="key-cell">${BrainApp.escapeHtml(k)}</td>
            <td>${BrainApp.escapeHtml(v.label || JSON.stringify(v))}</td>
          </tr>`).join('')}</tbody>
        </table>
      </div>` : ''}
      ${bgProcs.length > 0 ? `<div class="world-section">
        <h4>Background Processes</h4>
        <table class="records-table">
          <tbody>${bgProcs.map(p => `<tr><td>${BrainApp.escapeHtml(typeof p === 'string' ? p : JSON.stringify(p))}</td></tr>`).join('')}</tbody>
        </table>
      </div>` : '<div class="world-section"><h4>Background Processes</h4><p class="panel-desc">None running.</p></div>'}
      ${_worldState.rendered ? `<div class="world-section">
        <div class="world-sees-header">
          <h4>What Chalie Sees</h4>
          <div class="segmented" id="worldViewToggle">
            <button class="seg-btn${_worldViewMode === 'formatted' ? ' active' : ''}" data-mode="formatted">Formatted</button>
            <button class="seg-btn${_worldViewMode === 'raw' ? ' active' : ''}" data-mode="raw">Raw</button>
          </div>
        </div>
        ${_worldViewMode === 'raw'
          ? `<div class="code-block"><pre><code>${BrainApp.escapeHtml(_worldState.rendered)}</code></pre></div>`
          : `<div class="doc-content world-formatted">${_mdToHtml(_worldState.rendered)}</div>`}
      </div>` : ''}
    </div>`;

    el.querySelector('#worldViewToggle')?.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-mode]');
      if (!btn || btn.dataset.mode === _worldViewMode) return;
      _worldViewMode = btn.dataset.mode;
      _renderWorld(el);
    });
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
      const vals = {};
      el.querySelectorAll('.personality-range').forEach(r => { vals[r.dataset.key] = Number(r.value); });
      const tuple = [vals.warmth || 0, vals.mood || 0, vals.expressiveness || 0, vals.curiosity || 0, vals.humor || 0];
      try {
        const res = await BrainApp.apiFetch('/settings/personality', { method: 'PUT', body: JSON.stringify({ tuple }) });
        if (res.ok) { _personality = vals; BrainApp.showToast('Personality saved', 'success'); }
        else BrainApp.showToast('Failed to save', 'error');
      } catch { BrainApp.showToast('Network error', 'error'); }
    });
  }

  function _renderErrors(el) {
    if (_errors.length === 0) { el.innerHTML = '<div class="empty-state"><p>No recent errors.</p></div>'; return; }
    el.innerHTML = `<div class="error-list">${_errors.map(e => `<div class="error-item">
      <span class="error-time">${BrainApp.formatDate(e.time || e.timestamp)}</span>
      <span class="error-msg">${BrainApp.escapeHtml(e.message || '')}</span>
    </div>`).join('')}</div>`;
  }

  function _renderUsage(el) {
    const windows = ['hour', 'day', 'week', 'month', 'lifetime'];
    const rawSummary = _usage.summary || {};
    const entries = _usage.entries || [];

    const windowLabels = { hour: 'Last Hour', day: 'Last 24h', week: 'Last 7 Days', month: 'Last 30 Days', lifetime: 'All Time' };
    const summary = [
      { value: _fmtTokens(rawSummary.total_tokens || 0), label: windowLabels[_usageWindow] || 'Total Tokens' },
      { value: rawSummary.cache_hit_pct != null ? `${rawSummary.cache_hit_pct}%` : 'N/A', label: 'Cache Hit Rate' },
      { value: _fmtTokens(rawSummary.tokens_today || 0), label: 'Today (UTC)' },
      { value: rawSummary.most_active_model || '—', label: 'Top Model' },
    ];

    const bucketMap = {};
    for (const e of entries) {
      const b = e.bucket || '';
      if (!bucketMap[b]) bucketMap[b] = { input: 0, output: 0 };
      bucketMap[b].input += (e.tokens_input || 0) + (e.tokens_cache_read || 0);
      bucketMap[b].output += (e.tokens_output || 0) + (e.tokens_thinking || 0);
    }
    const chart = Object.entries(bucketMap)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([bucket, v]) => {
        let label;
        if (_usageWindow === 'hour' || _usageWindow === 'day') label = bucket.slice(11, 16);
        else label = bucket.slice(5, 10);
        return { label, input: v.input, output: v.output };
      });

    el.innerHTML = `<div class="filter-tabs" id="usageWindowTabs">
      ${windows.map(w => `<button class="filter-tab${w === _usageWindow ? ' active' : ''}" data-win="${w}">${w.charAt(0).toUpperCase() + w.slice(1)}</button>`).join('')}
    </div>
    <div class="stat-grid">${summary.map(s => `<div class="stat-card">
      <div class="stat-value">${BrainApp.escapeHtml(String(s.value ?? '—'))}</div>
      <div class="stat-label">${BrainApp.escapeHtml(s.label || '')}</div>
    </div>`).join('')}</div>
    ${chart.length > 0 ? _renderChart(chart) : ''}
    ${chart.length > 0 ? _renderUsageTable(chart) : ''}`;

    el.querySelector('#usageWindowTabs')?.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-win]');
      if (!btn) return;
      _usageWindow = btn.dataset.win;
      _loaded.usage = false; _loadSub();
    });
  }

  function _fmtTokens(n) {
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
    return String(n);
  }

  function _renderUsageTable(chart) {
    const dateHeader = (_usageWindow === 'hour' || _usageWindow === 'day') ? 'Hour' : 'Date';
    const rows = chart.map(d => `<tr>
      <td>${BrainApp.escapeHtml(String(d.label || ''))}</td>
      <td class="num">${(d.input || 0).toLocaleString()}</td>
      <td class="num">${(d.output || 0).toLocaleString()}</td>
    </tr>`).join('');
    return `<table class="usage-table">
      <thead><tr><th>${dateHeader}</th><th>Input Tokens</th><th>Output Tokens</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  }

  function _renderChart(chart) {
    const maxVal = Math.max(...chart.map(d => (d.input || 0) + (d.output || 0)), 1);
    const barW = Math.max(16, Math.floor(600 / chart.length) - 4);
    const h = 160;
    const labelH = 40;
    const showEvery = chart.length > 20 ? Math.ceil(chart.length / 15) : 1;
    const bars = chart.map((d, i) => {
      const inputH = ((d.input || 0) / maxVal) * h;
      const outputH = ((d.output || 0) / maxVal) * h;
      const x = i * (barW + 4);
      const showLabel = i % showEvery === 0;
      return `<g>
        <rect x="${x}" y="${h - inputH - outputH}" width="${barW}" height="${outputH}" class="bar-cloud"/>
        <rect x="${x}" y="${h - inputH}" width="${barW}" height="${inputH}" class="bar-local"/>
        ${showLabel ? `<text x="${x + barW / 2}" y="${h + 6}" class="bar-label" transform="rotate(45 ${x + barW / 2} ${h + 6})">${BrainApp.escapeHtml(String(d.label || ''))}</text>` : ''}
      </g>`;
    }).join('');
    const svgW = chart.length * (barW + 4);
    return `<div class="chart-wrap"><svg class="usage-chart" viewBox="0 0 ${svgW} ${h + labelH}" preserveAspectRatio="none">${bars}</svg></div>
    <div class="chart-legend"><span class="legend-local">Chat</span><span class="legend-cloud">Subconscious</span></div>`;
  }

  return { mount, unmount };
})();

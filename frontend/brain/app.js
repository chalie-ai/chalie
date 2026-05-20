// Brain Dashboard — main controller.
// Routing, init, command palette, toast, theme, session heartbeat.
const BrainApp = (() => {
  // ── State ──
  let _route = { section: 'providers', sub: null };
  let _providersOnly = false;
  let _sidebarCollapsed = false;
  let _mobileOpen = false;
  let _theme = localStorage.getItem('chalie_theme') || 'dark';
  let _heartbeatInterval = null;
  let _heartbeatFired = false;

  const SUB_ROUTES = {
    providers: null,
    cognition: ['memory', 'tools', 'world', 'personality', 'errors', 'usage'],
    scheduler: ['all', 'pending', 'fired', 'failed', 'cancelled'],
    lists: null,
    documents: ['active', 'processing', 'uploads', 'deleted'],
    capabilities: null,
    policies: ['chat', 'subagent', 'background', 'external', 'discord'],
    mcp: null,
  };

  const LABELS = {
    providers: 'Providers', cognition: 'Cognition', scheduler: 'Scheduler',
    lists: 'Lists', documents: 'Documents', capabilities: 'Capabilities',
    policies: 'Policies', mcp: 'MCP Server',
    memory: 'Memory', tools: 'Tools', world: 'World state',
    personality: 'Personality', errors: 'Errors', usage: 'Usage',
    all: 'All', pending: 'Pending', fired: 'Fired', failed: 'Failed', cancelled: 'Cancelled',
    active: 'Active', processing: 'Processing', uploads: 'Uploads', deleted: 'Deleted',
    chat: 'Chat', subagent: 'Subagent', background: 'Background', external: 'External agent', discord: 'Discord',
  };

  const PANELS = {
    providers: typeof PanelProviders !== 'undefined' ? PanelProviders : null,
    cognition: typeof PanelCognition !== 'undefined' ? PanelCognition : null,
    scheduler: typeof PanelScheduler !== 'undefined' ? PanelScheduler : null,
    lists: typeof PanelLists !== 'undefined' ? PanelLists : null,
    documents: typeof PanelDocuments !== 'undefined' ? PanelDocuments : null,
    capabilities: typeof PanelCapabilities !== 'undefined' ? PanelCapabilities : null,
    policies: typeof PanelPolicies !== 'undefined' ? PanelPolicies : null,
    mcp: typeof PanelMcp !== 'undefined' ? PanelMcp : null,
  };

  // ── API ──
  let API_BASE = (() => {
    if (location.protocol === 'http:' || location.protocol === 'https:') return '';
    return localStorage.getItem('chalie_backend_host') || 'http://localhost:8080';
  })();

  async function apiFetch(path, options = {}, isMultipart = false) {
    const url = API_BASE ? `${API_BASE.replace(/\/$/, '')}${path}` : path;
    const headers = {
      ...(isMultipart ? {} : { 'Content-Type': 'application/json' }),
      ...(options.headers || {}),
    };
    const creds = API_BASE ? 'include' : 'same-origin';
    const res = await fetch(url, { ...options, headers, credentials: creds });
    if (res.status === 401) { location.replace('/login/?next=/brain/'); return res; }
    return res;
  }

  // ── Toast ──
  function showToast(message, type = 'info', opts = {}) {
    const { action, onAction, duration = 3200 } = opts;
    const host = document.getElementById('toastHost');
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = message;
    if (action && onAction) {
      const btn = document.createElement('button');
      btn.className = 'toast-action';
      btn.textContent = action;
      btn.addEventListener('click', () => { onAction(); el.remove(); });
      el.appendChild(btn);
    }
    host.appendChild(el);
    setTimeout(() => {
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 300);
    }, duration);
  }

  // ── Routing ──
  function _readHash() {
    const h = (location.hash || '').replace(/^#\/?/, '');
    const [section, sub] = h.split('/');
    if (section && SUB_ROUTES.hasOwnProperty(section)) {
      const allowed = SUB_ROUTES[section];
      const validSub = allowed ? (allowed.includes(sub) ? sub : allowed[0]) : null;
      return { section, sub: validSub };
    }
    return { section: 'providers', sub: null };
  }

  function navigate(section, sub) {
    if (_providersOnly && section !== 'providers') {
      section = 'providers';
      sub = null;
    }
    _route = { section, sub };
    const h = sub ? `#/${section}/${sub}` : `#/${section}`;
    history.replaceState(null, '', h);
    _render();
  }

  function getRoute() { return _route; }

  // ── Theme ──
  function toggleTheme() {
    _theme = _theme === 'dark' ? 'light' : 'dark';
    localStorage.setItem('chalie_theme', _theme);
    document.documentElement.setAttribute('data-theme', _theme);
    BrainSidebar.updateThemeIcon();
  }

  // ── Sidebar ──
  function isSidebarCollapsed() { return _sidebarCollapsed; }

  function toggleSidebar() {
    _sidebarCollapsed = !_sidebarCollapsed;
    document.getElementById('appShell').dataset.collapsed = _sidebarCollapsed;
    BrainSidebar.update();
  }

  function closeMobileSidebar() {
    _mobileOpen = false;
    document.getElementById('appShell').dataset.mobileOpen = 'false';
  }

  function openMobileSidebar() {
    _mobileOpen = true;
    document.getElementById('appShell').dataset.mobileOpen = 'true';
  }

  // ── Command Palette ──
  function openCommandPalette() {
    const overlay = document.getElementById('cpOverlay');
    overlay.classList.remove('hidden');
    _renderCommandPalette('');
    setTimeout(() => {
      const input = overlay.querySelector('.cp-input');
      if (input) input.focus();
    }, 50);
  }

  function _closeCommandPalette() {
    document.getElementById('cpOverlay').classList.add('hidden');
  }

  function _renderCommandPalette(query) {
    const overlay = document.getElementById('cpOverlay');
    const allItems = [
      { kind: 'Jump', label: 'Providers', icon: 'Providers', section: 'providers' },
      { kind: 'Jump', label: 'Cognition · Memory', icon: 'Memory', section: 'cognition', sub: 'memory' },
      { kind: 'Jump', label: 'Cognition · Tools', icon: 'Tool', section: 'cognition', sub: 'tools' },
      { kind: 'Jump', label: 'Cognition · Personality', icon: 'Sparkles', section: 'cognition', sub: 'personality' },
      { kind: 'Jump', label: 'Cognition · Usage', icon: 'Chart', section: 'cognition', sub: 'usage' },
      { kind: 'Jump', label: 'Scheduler · Pending', icon: 'Calendar', section: 'scheduler', sub: 'pending' },
      { kind: 'Jump', label: 'Lists', icon: 'List', section: 'lists' },
      { kind: 'Jump', label: 'Documents · Active', icon: 'Document', section: 'documents', sub: 'active' },
      { kind: 'Jump', label: 'Capabilities', icon: 'Capability', section: 'capabilities' },
      { kind: 'Jump', label: 'Policies · Chat', icon: 'Policy', section: 'policies', sub: 'chat' },
      { kind: 'Jump', label: 'MCP Server', icon: 'Server', section: 'mcp' },
    ];

    const q = query.toLowerCase();
    const filtered = allItems.filter(a => !q || a.label.toLowerCase().includes(q));
    let sel = 0;

    let html = `<div class="cp" id="cpDialog">
      <input class="cp-input" id="cpInput" placeholder="Search the brain… (try ‘memory’, ‘policy’)" value="${_escapeAttr(query)}">
      <div class="cp-results">`;

    const groups = {};
    for (const item of filtered) {
      (groups[item.kind] = groups[item.kind] || []).push(item);
    }
    let idx = 0;
    for (const [gname, items] of Object.entries(groups)) {
      html += `<div class="cp-section">${gname}</div>`;
      for (const it of items) {
        html += `<div class="cp-row${idx === sel ? ' selected' : ''}" data-idx="${idx}" data-section="${it.section}" data-sub="${it.sub || ''}">
          <span class="cp-icon">${Icons[it.icon](14)}</span>
          <span>${it.label}</span>
          ${idx === sel ? '<span class="cp-hint">↵</span>' : ''}
        </div>`;
        idx++;
      }
    }
    if (filtered.length === 0) {
      html += `<div class="cp-empty">No matches. Try fewer words.</div>`;
    }
    html += `</div>
      <div class="cp-footer">
        <span><kbd>↑</kbd><kbd>↓</kbd> navigate · <kbd>↵</kbd> open · <kbd>esc</kbd> close</span>
        <span>${filtered.length} result${filtered.length === 1 ? '' : 's'}</span>
      </div>
    </div>`;
    overlay.innerHTML = html;

    const input = document.getElementById('cpInput');
    input.addEventListener('input', () => _renderCommandPalette(input.value));
    input.addEventListener('keydown', (e) => {
      const rows = overlay.querySelectorAll('.cp-row');
      if (e.key === 'ArrowDown') { e.preventDefault(); sel = Math.min(filtered.length - 1, sel + 1); _highlightCpRow(rows, sel); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); sel = Math.max(0, sel - 1); _highlightCpRow(rows, sel); }
      else if (e.key === 'Enter') { e.preventDefault(); _selectCpRow(rows[sel]); }
      else if (e.key === 'Escape') { _closeCommandPalette(); }
    });

    overlay.onclick = (e) => {
      if (e.target === overlay) { _closeCommandPalette(); return; }
      const row = e.target.closest('.cp-row');
      if (row) _selectCpRow(row);
    };
  }

  function _highlightCpRow(rows, idx) {
    rows.forEach((r, i) => {
      r.classList.toggle('selected', i === idx);
      if (i === idx) {
        const hint = r.querySelector('.cp-hint');
        if (!hint) r.insertAdjacentHTML('beforeend', '<span class="cp-hint">↵</span>');
      } else {
        const hint = r.querySelector('.cp-hint');
        if (hint) hint.remove();
      }
    });
  }

  function _selectCpRow(row) {
    if (!row) return;
    const section = row.dataset.section;
    const sub = row.dataset.sub || null;
    navigate(section, sub);
    _closeCommandPalette();
  }

  // ── Topbar ──
  function _renderTopbar() {
    const topbar = document.getElementById('topbar');
    const topLabel = LABELS[_route.section] || _route.section;
    const subLabel = _route.sub ? LABELS[_route.sub] : null;

    topbar.innerHTML = `
      <button class="icon-btn hamburger" id="hamburgerBtn" aria-label="Open menu">${Icons.Menu()}</button>
      <button class="icon-btn desktop-collapser" id="collapserBtn" aria-label="Toggle sidebar" title="Toggle sidebar">${Icons.Sidebar()}</button>
      <div class="crumb">
        <span>Brain</span>
        <span class="sep">/</span>
        <span class="now">${topLabel}</span>
        ${subLabel ? `<span class="sep">/</span><span class="now">${subLabel}</span>` : ''}
      </div>
      <div class="topbar-center">
        <div class="topbar-search" id="topbarSearch" role="button" tabindex="0">
          ${Icons.Search(14)}
          <span class="topbar-search-text">Search…</span>
          <kbd>⌘K</kbd>
        </div>
      </div>
      <div class="topbar-actions"></div>`;

    document.getElementById('hamburgerBtn').addEventListener('click', openMobileSidebar);
    document.getElementById('collapserBtn').addEventListener('click', toggleSidebar);
    const searchBtn = document.getElementById('topbarSearch');
    searchBtn.addEventListener('click', openCommandPalette);
    searchBtn.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openCommandPalette(); }
    });
  }

  // ── Panel Rendering ──
  let _activePanel = null;

  function _renderPanel() {
    const root = document.getElementById('panelRoot');
    _activePanel?.unmount?.(root);

    const panel = PANELS[_route.section];
    if (panel?.mount) {
      panel.mount(root, _route.sub);
      _activePanel = panel;
    } else {
      root.innerHTML = `<div class="panel-placeholder"><h2>${LABELS[_route.section] || _route.section}</h2><p>Panel loading…</p></div>`;
      _activePanel = null;
    }
  }

  // ── Main Render ──
  function _render() {
    _renderTopbar();
    BrainSidebar.update();
    _renderPanel();
  }

  // ── Session Heartbeat ──
  async function _checkSession() {
    if (_heartbeatFired) return;
    try {
      const url = API_BASE ? `${API_BASE.replace(/\/$/, '')}/auth/status` : '/auth/status';
      const creds = API_BASE ? 'include' : 'same-origin';
      const res = await fetch(url, { credentials: creds });
      if (!res.ok) return;
      const data = await res.json();
      if (data.has_master_account && !data.has_session) {
        _heartbeatFired = true;
        if (_heartbeatInterval) { clearInterval(_heartbeatInterval); _heartbeatInterval = null; }
        location.replace('/login/?next=/brain/');
      }
    } catch (e) { /* transient */ }
  }

  function _startHeartbeat() {
    if (_heartbeatInterval) return;
    _heartbeatInterval = setInterval(_checkSession, 5 * 60 * 1000);
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') _checkSession();
    });
  }

  // ── Providers-Only Mode ──
  function _applyProvidersOnly() {
    _providersOnly = true;
    _route = { section: 'providers', sub: null };
    document.getElementById('appShell').dataset.providersOnly = 'true';
    history.replaceState(null, '', '#/providers');
  }

  function liftProvidersOnly() {
    _providersOnly = false;
    document.getElementById('appShell').dataset.providersOnly = 'false';
    BrainSidebar.update();
  }

  function isProvidersOnly() { return _providersOnly; }

  // ── Init ──
  async function init() {
    document.documentElement.setAttribute('data-theme', _theme);

    const gate = await globalThis.chalieGateReady;
    if (!gate.stay) return;

    if (gate.providersOnly) _applyProvidersOnly();
    else _route = _readHash();

    BrainSidebar.render(document.getElementById('sidebar'));
    _render();

    document.getElementById('mobileScrim').addEventListener('click', closeMobileSidebar);
    globalThis.addEventListener('hashchange', () => {
      const parsed = _readHash();
      navigate(parsed.section, parsed.sub);
    });
    globalThis.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); openCommandPalette(); }
      else if (e.key === 'Escape') _closeCommandPalette();
    });

    _startHeartbeat();
    document.getElementById('appShell').style.opacity = '1';
  }

  // ── Helpers ──
  function _escapeAttr(s) { return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;'); }
  function escapeHtml(s) {
    if (!s) return '';
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function formatDate(raw) {
    if (!raw) return '';
    try {
      const d = new Date(raw.includes('T') || raw.includes('+') || raw.includes('Z') ? raw : raw.replace(' ', 'T') + 'Z');
      if (isNaN(d.getTime())) return raw;
      const pad = (n) => String(n).padStart(2, '0');
      return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
    } catch { return raw; }
  }

  // ── Public API ──
  return {
    init,
    navigate,
    getRoute,
    toggleTheme,
    toggleSidebar,
    isSidebarCollapsed,
    isProvidersOnly,
    liftProvidersOnly,
    openMobileSidebar,
    closeMobileSidebar,
    openCommandPalette,
    apiFetch,
    showToast,
    escapeHtml,
    formatDate,
  };
})();

document.addEventListener('DOMContentLoaded', BrainApp.init);

// Sidebar navigation — collapsible groups with sub-menus.
const BrainSidebar = (() => {
  const NAV = [
    { id: 'providers', label: 'Providers', icon: 'Providers', group: 'cognition' },
    { id: 'cognition', label: 'Cognition', icon: 'Brain', group: 'cognition',
      sub: [
        { id: 'memory', label: 'Memory' },
        { id: 'tools', label: 'Tools' },
        { id: 'personality', label: 'Personality' },
        { id: 'errors', label: 'Errors' },
        { id: 'usage', label: 'Usage' },
      ]},
    { id: 'scheduler', label: 'Scheduler', icon: 'Calendar', group: 'cognition',
      sub: [
        { id: 'all', label: 'All' },
        { id: 'pending', label: 'Pending' },
        { id: 'fired', label: 'Fired' },
        { id: 'failed', label: 'Failed' },
        { id: 'cancelled', label: 'Cancelled' },
      ]},
    { id: 'lists', label: 'Lists', icon: 'List', group: 'cognition' },
    { id: 'documents', label: 'Documents', icon: 'Document', group: 'cognition',
      sub: [
        { id: 'active', label: 'Active' },
        { id: 'processing', label: 'Processing' },
        { id: 'uploads', label: 'Uploads' },
        { id: 'deleted', label: 'Deleted' },
      ]},
    { id: 'capabilities', label: 'Capabilities', icon: 'Capability', group: 'system' },
    { id: 'policies', label: 'Policies', icon: 'Policy', group: 'system',
      sub: [
        { id: 'chat', label: 'Chat' },
        { id: 'subagent', label: 'Subagent' },
        { id: 'background', label: 'Background' },
        { id: 'external', label: 'External agent' },
      ]},
    { id: 'mcp', label: 'MCP Server', icon: 'Server', group: 'system' },
  ];

  let _el = null;

  function render(container) {
    _el = container;
    container.innerHTML = `
      <div class="sidebar-brand">
        <img src="/icons/icon.png" alt="Chalie" width="28" height="28">
        <div class="sidebar-brand-text">
          <div class="wordmark">Chalie</div>
          <div class="wordmark-sub">Brain</div>
        </div>
      </div>
      <nav class="sidebar-scroll" id="sidebarNav"></nav>
      <div class="sidebar-footer">
        <button class="icon-btn theme-toggle" id="themeToggleBtn" aria-label="Toggle theme">
          ${Icons.Sun()}
        </button>
      </div>`;

    document.getElementById('themeToggleBtn').addEventListener('click', BrainApp.toggleTheme);
    _renderNav();
  }

  function _renderNav() {
    const nav = document.getElementById('sidebarNav');
    if (!nav) return;
    const route = BrainApp.getRoute();
    const collapsed = BrainApp.isSidebarCollapsed();

    const cognitionItems = NAV.filter(n => n.group === 'cognition');
    const systemItems = NAV.filter(n => n.group === 'system');

    nav.innerHTML =
      _renderGroup('Cognition', cognitionItems, route, collapsed) +
      _renderGroup('System', systemItems, route, collapsed);

    nav.querySelectorAll('[data-nav]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const section = btn.dataset.nav;
        const item = NAV.find(n => n.id === section);
        if (item.sub) {
          if (route.section !== section) {
            BrainApp.navigate(section, item.sub[0].id);
          }
        } else {
          BrainApp.navigate(section, null);
        }
        if (window.innerWidth <= 900) BrainApp.closeMobileSidebar();
      });
    });

    nav.querySelectorAll('[data-sub]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const sectionEl = btn.closest('[data-section]');
        if (!sectionEl) return;
        BrainApp.navigate(sectionEl.dataset.section, btn.dataset.sub);
        if (window.innerWidth <= 900) BrainApp.closeMobileSidebar();
      });
    });
  }

  function _renderGroup(title, items, route, collapsed) {
    let html = `<div class="nav-group">`;
    if (!collapsed) html += `<div class="nav-group-title">${title}</div>`;
    for (const item of items) {
      const isActive = route.section === item.id;
      const expanded = isActive && !!item.sub && !collapsed;
      html += `<div data-section="${item.id}">`;
      html += `<button class="nav-item${isActive ? ' active' : ''}" data-nav="${item.id}" data-expanded="${expanded}">
        <span class="nav-icon">${Icons[item.icon]()}</span>
        <span class="nav-label">${item.label}</span>
        ${item.sub ? `<span class="nav-chev">${Icons.Chevron(14)}</span>` : ''}
      </button>`;
      if (item.sub) {
        html += `<div class="nav-sublist" data-open="${expanded}"><div>`;
        for (const s of item.sub) {
          html += `<button class="nav-sub${route.sub === s.id && isActive ? ' active' : ''}" data-sub="${s.id}">
            <span>${s.label}</span>
          </button>`;
        }
        html += `</div></div>`;
      }
      html += `</div>`;
    }
    html += `</div>`;
    return html;
  }

  function update() { if (_el) _renderNav(); }

  function updateThemeIcon() {
    const btn = document.getElementById('themeToggleBtn');
    if (btn) {
      const theme = document.documentElement.getAttribute('data-theme');
      btn.innerHTML = theme === 'dark' ? Icons.Sun() : Icons.Moon();
    }
  }

  return { NAV, render, update, updateThemeIcon };
})();

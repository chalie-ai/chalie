import { escHtml } from './utils.js';

/**
 * Interface daemon management — apps panel, scope approval, app overlay.
 */
export class AppsPanel {
  constructor({ getHost }) {
    this._getHost = getHost;
    this._apps = [];
    this._appsOpen = false;
    this._currentScopeApp = null;
    this._appsInterval = null;
  }

  init() {
    const appsBtn = document.getElementById('appsBtn');
    const closePanel = document.getElementById('closeAppsPanel');
    const closeOverlay = document.getElementById('closeAppOverlay');

    // Show apps button (always visible when dashboard serves the UI)
    appsBtn?.classList.remove('hidden');

    appsBtn?.addEventListener('click', () => this._toggleAppsPanel());
    closePanel?.addEventListener('click', () => this._toggleAppsPanel(false));

    closeOverlay?.addEventListener('click', () => this._closeAppOverlay());

    // Scope dialog
    const scopeDialog = document.getElementById('scopeDialog');
    document.getElementById('scopeDialogClose')?.addEventListener('click', () => scopeDialog?.close());
    document.getElementById('scopeDenyBtn')?.addEventListener('click', () => scopeDialog?.close());
    document.getElementById('scopeApproveBtn')?.addEventListener('click', () => this._submitScopeApproval());

    // Detail dialog
    const detailDialog = document.getElementById('appDetailDialog');
    document.getElementById('appDetailClose')?.addEventListener('click', () => detailDialog?.close());
    document.getElementById('appDetailRemoveBtn')?.addEventListener('click', () => this._removeCurrentApp());
    document.getElementById('appDetailSaveBtn')?.addEventListener('click', () => this._saveAppScopes());

    // Initial load + periodic refresh
    this.loadApps();
    this._appsInterval = setInterval(() => this.loadApps(), 30000);
  }

  async loadApps() {
    try {
      const host = this._getHost();
      const base = host ? host.replace(/\/$/, '') : '';
      const resp = await fetch(`${base}/api/apps`, { credentials: 'same-origin' });
      if (!resp.ok) return;
      this._apps = await resp.json();
      this._renderAppsList();
    } catch (_) { /* dashboard may not be running */ }
  }

  destroy() {
    clearInterval(this._appsInterval);
    this._appsInterval = null;
  }

  _renderAppsList() {
    const list = document.getElementById('appsPanelList');
    if (!list) return;

    if (!this._apps.length) {
      list.innerHTML = '<p class="apps-panel__empty">No apps installed</p>';
      return;
    }

    // Sort: pending first, then online, then offline
    const order = { pending: 0, online: 1, offline: 2 };
    const sorted = [...this._apps].sort((a, b) =>
      (order[a.status] ?? 3) - (order[b.status] ?? 3)
    );

    list.innerHTML = sorted.map(app => `
      <div class="app-card" data-app-id="${app.id}" data-status="${app.status}">
        <span class="app-card__dot app-card__dot--${app.status}"></span>
        <div class="app-card__info">
          <div class="app-card__name">${escHtml(app.name)}</div>
          ${app.description ? `<div class="app-card__desc">${escHtml(app.description)}</div>` : ''}
        </div>
        ${app.status === 'pending'
          ? '<span class="app-card__badge app-card__badge--pending">Needs approval</span>'
          : `<button class="btn-icon app-card__gear" data-gear="${app.id}" aria-label="Settings">
               <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                 <circle cx="12" cy="12" r="3"></circle>
                 <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
               </svg>
             </button>`
        }
      </div>
    `).join('');

    // Bind click handlers
    list.querySelectorAll('.app-card').forEach(card => {
      card.addEventListener('click', (e) => {
        // If gear icon clicked, open detail instead
        if (e.target.closest('[data-gear]')) {
          e.stopPropagation();
          this._openAppDetail(card.dataset.appId);
          return;
        }
        const app = this._apps.find(a => a.id === card.dataset.appId);
        if (!app) return;

        if (app.status === 'pending') {
          this._openScopeApproval(app);
        } else if (app.status === 'online') {
          this._openAppOverlay(app);
        }
      });
    });
  }

  _toggleAppsPanel(forceState) {
    const panel = document.getElementById('appsPanel');
    if (!panel) return;

    const shouldOpen = forceState !== undefined ? forceState : !this._appsOpen;
    this._appsOpen = shouldOpen;

    if (shouldOpen) {
      panel.classList.remove('hidden');
      requestAnimationFrame(() => panel.classList.add('open'));
      this.loadApps();
    } else {
      panel.classList.remove('open');
      panel.addEventListener('transitionend', () => {
        if (!this._appsOpen) panel.classList.add('hidden');
      }, { once: true });
    }
  }

  _openScopeApproval(app) {
    this._currentScopeApp = app;
    const dialog = document.getElementById('scopeDialog');
    const title = document.getElementById('scopeDialogTitle');
    const desc = document.getElementById('scopeDialogDesc');
    const scopesEl = document.getElementById('scopeDialogScopes');

    title.textContent = app.name;
    desc.textContent = app.description || 'This app wants to connect to Chalie.';

    const scopes = app.requested_scopes || {};
    scopesEl.innerHTML = this._renderScopeToggles(scopes, {});

    dialog?.showModal();
  }

  _renderScopeToggles(requested, approved) {
    const labels = {
      context: 'Context Access',
      signals: 'Signal Types',
      messages: 'Message Topics',
    };

    let html = '';
    for (const [cat, items] of Object.entries(requested)) {
      if (!items || typeof items !== 'object') continue;
      const entries = Object.entries(items);
      if (!entries.length) continue;

      const approvedCat = approved[cat] || {};
      html += `<div class="scope-category">
        <div class="scope-category__title">${labels[cat] || cat}</div>`;

      for (const [key, desc] of entries) {
        const isOn = key in (typeof approvedCat === 'object' ? approvedCat : {});
        html += `<div class="scope-item">
          <button class="scope-item__toggle ${isOn ? 'on' : ''}" data-scope-cat="${cat}" data-scope-key="${key}"></button>
          <div>
            <div class="scope-item__label">${escHtml(key)}</div>
            ${desc ? `<div class="scope-item__desc">${escHtml(String(desc))}</div>` : ''}
          </div>
        </div>`;
      }
      html += '</div>';
    }

    // Bind toggle after render
    setTimeout(() => {
      document.querySelectorAll('.scope-item__toggle').forEach(btn => {
        btn.addEventListener('click', () => btn.classList.toggle('on'));
      });
    }, 0);

    return html;
  }

  async _submitScopeApproval() {
    const app = this._currentScopeApp;
    if (!app) return;

    const approved = this._collectScopeState();
    const host = this._getHost();
    const base = host ? host.replace(/\/$/, '') : '';

    try {
      const resp = await fetch(`${base}/api/apps/${app.id}/scopes`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ approved_scopes: approved }),
      });
      if (resp.ok) {
        document.getElementById('scopeDialog')?.close();
        this.loadApps();
      }
    } catch (e) {
      console.error('Failed to approve scopes:', e);
    }
  }

  _collectScopeState() {
    const approved = {};
    document.querySelectorAll('.scope-item__toggle.on').forEach(btn => {
      const cat = btn.dataset.scopeCat;
      const key = btn.dataset.scopeKey;
      if (!approved[cat]) approved[cat] = {};
      approved[cat][key] = true;
    });
    return approved;
  }

  async _openAppOverlay(app) {
    const overlay = document.getElementById('appOverlay');
    const title = document.getElementById('appOverlayTitle');
    const content = document.getElementById('appOverlayContent');

    title.textContent = app.name;
    content.innerHTML = '<p style="color:var(--text-tertiary)">Loading...</p>';
    overlay?.classList.remove('hidden');

    // Close apps panel
    this._toggleAppsPanel(false);

    try {
      const resp = await fetch(`/gw/${app.id}/render`);
      if (resp.ok) {
        const html = await resp.text();
        // innerHTML doesn't execute <script> tags — parse and re-insert them
        content.innerHTML = html;
        content.querySelectorAll('script').forEach(old => {
          const s = document.createElement('script');
          if (old.src) s.src = old.src;
          else s.textContent = old.textContent;
          old.replaceWith(s);
        });
        if (window.lucide) lucide.createIcons({ node: content });
      } else {
        content.innerHTML = '<p style="color:var(--text-secondary)">Could not load app interface.</p>';
      }
    } catch (_) {
      content.innerHTML = '<p style="color:var(--text-secondary)">App is not reachable.</p>';
    }
  }

  _closeAppOverlay() {
    document.getElementById('appOverlay')?.classList.add('hidden');
    document.getElementById('appOverlayContent').innerHTML = '';
  }

  async _openAppDetail(appId) {
    const host = this._getHost();
    const base = host ? host.replace(/\/$/, '') : '';
    try {
      const resp = await fetch(`${base}/api/apps/${appId}`, { credentials: 'same-origin' });
      if (!resp.ok) return;
      const app = await resp.json();
      this._currentScopeApp = app;

      document.getElementById('appDetailTitle').textContent = app.name;
      document.getElementById('appDetailMeta').textContent =
        [app.version && `v${app.version}`, app.author, `${app.host}:${app.port}`, app.status].filter(Boolean).join(' · ');

      // Scopes
      document.getElementById('appDetailScopes').innerHTML =
        this._renderScopeToggles(app.requested_scopes || {}, app.approved_scopes || {});

      // Capabilities
      const capEl = document.getElementById('appDetailCapabilities');
      const caps = app.capabilities || [];
      if (caps.length) {
        capEl.innerHTML = `<div class="app-detail__cap-title">Capabilities (${caps.length})</div>` +
          caps.map(c => `<div class="app-detail__cap-item">${escHtml(c.name || c.id || JSON.stringify(c))}</div>`).join('');
      } else {
        capEl.innerHTML = '';
      }

      document.getElementById('appDetailDialog')?.showModal();
    } catch (e) {
      console.error('Failed to load app detail:', e);
    }
  }

  async _saveAppScopes() {
    const app = this._currentScopeApp;
    if (!app) return;
    const approved = this._collectScopeState();
    const host = this._getHost();
    const base = host ? host.replace(/\/$/, '') : '';

    try {
      await fetch(`${base}/api/apps/${app.id}/scopes`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ approved_scopes: approved }),
      });
      document.getElementById('appDetailDialog')?.close();
      this.loadApps();
    } catch (e) {
      console.error('Failed to save scopes:', e);
    }
  }

  async _removeCurrentApp() {
    const app = this._currentScopeApp;
    if (!app) return;
    const host = this._getHost();
    const base = host ? host.replace(/\/$/, '') : '';

    try {
      await fetch(`${base}/api/apps/${app.id}`, {
        method: 'DELETE',
        credentials: 'same-origin',
      });
      document.getElementById('appDetailDialog')?.close();
      this.loadApps();
    } catch (e) {
      console.error('Failed to remove app:', e);
    }
  }
}

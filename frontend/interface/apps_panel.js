import { escHtml } from './utils.js';
import { renderMarkupTo } from './markup_renderer.js';

/**
 * Apps (interface daemons) — side panel, scope approval, overlay, detail dialog.
 */
export class AppsPanel {
  /**
   * @param {{ getHost: () => string }} opts
   */
  constructor({ getHost }) {
    this._getHost = getHost;
    this._apps = [];
    this._appsOpen = false;
    this._currentScopeApp = null;
    this._overlayPollTimers = [];
    this._overlayGateway = null;
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
    this._loadApps();
    this._appsInterval = setInterval(() => this._loadApps(), 30000);
  }

  destroy() {
    clearInterval(this._appsInterval);
    this._appsInterval = null;
    for (const id of this._overlayPollTimers) clearInterval(id);
    this._overlayPollTimers = [];
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  async _loadApps() {
    try {
      const base = this._getHost() ? this._getHost().replace(/\/$/, '') : '';
      const resp = await fetch(`${base}/api/apps`, { credentials: 'same-origin' });
      if (!resp.ok) return;
      this._apps = await resp.json();
      this._renderAppsList();
    } catch (e) { console.warn('[apps_panel] failed to load apps:', e); }
  }

  _renderAppsList() {
    const list = document.getElementById('appsPanelList');
    if (!list) return;

    if (!this._apps.length) {
      list.textContent = '';
      const emptyP = document.createElement('p');
      emptyP.className = 'apps-panel__empty';
      emptyP.textContent = 'No apps installed';
      list.appendChild(emptyP);
      return;
    }

    // Sort: pending first, then online, then offline
    const order = { pending: 0, online: 1, offline: 2 };
    const sorted = [...this._apps].sort((a, b) =>
      (order[a.status] ?? 3) - (order[b.status] ?? 3)
    );

    const frag = document.createDocumentFragment();

    for (const app of sorted) {
      const card = document.createElement('div');
      card.className = 'app-card';
      card.dataset.appId = app.id;
      card.dataset.status = app.status;

      const dot = document.createElement('span');
      dot.className = `app-card__dot app-card__dot--${app.status}`;
      card.appendChild(dot);

      const info = document.createElement('div');
      info.className = 'app-card__info';
      const nameEl = document.createElement('div');
      nameEl.className = 'app-card__name';
      nameEl.textContent = app.name;
      info.appendChild(nameEl);
      if (app.description) {
        const descEl = document.createElement('div');
        descEl.className = 'app-card__desc';
        descEl.textContent = app.description;
        info.appendChild(descEl);
      }
      card.appendChild(info);

      if (app.status === 'pending') {
        const badge = document.createElement('span');
        badge.className = 'app-card__badge app-card__badge--pending';
        badge.textContent = 'Needs approval';
        card.appendChild(badge);
      } else {
        const gearBtn = document.createElement('button');
        gearBtn.className = 'btn-icon app-card__gear';
        gearBtn.dataset.gear = app.id;
        gearBtn.setAttribute('aria-label', 'Settings');
        gearBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>';
        card.appendChild(gearBtn);
      }

      // Bind click handler inline
      card.addEventListener('click', (e) => {
        if (e.target.closest('[data-gear]')) {
          e.stopPropagation();
          this._openAppDetail(app.id);
          return;
        }
        if (app.status === 'pending') {
          this._openScopeApproval(app);
        } else if (app.status === 'online') {
          this._openAppOverlay(app);
        }
      });

      frag.appendChild(card);
    }

    list.textContent = '';
    list.appendChild(frag);
  }

  _toggleAppsPanel(forceState) {
    const panel = document.getElementById('appsPanel');
    if (!panel) return;

    const shouldOpen = forceState !== undefined ? forceState : !this._appsOpen;
    this._appsOpen = shouldOpen;

    if (shouldOpen) {
      panel.classList.remove('hidden');
      requestAnimationFrame(() => panel.classList.add('open'));
      this._loadApps();
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
    const base = this._getHost() ? this._getHost().replace(/\/$/, '') : '';

    try {
      const resp = await fetch(`${base}/api/apps/${app.id}/scopes`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ approved_scopes: approved }),
      });
      if (resp.ok) {
        document.getElementById('scopeDialog')?.close();
        this._loadApps();
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
    content.textContent = '';
    const loadingP = document.createElement('p');
    loadingP.style.color = 'var(--text-tertiary)';
    loadingP.textContent = 'Loading...';
    content.appendChild(loadingP);
    overlay?.classList.remove('hidden');

    // Close apps panel
    this._toggleAppsPanel(false);

    try {
      const resp = await fetch(`/gw/${app.id}/render`);
      if (!resp.ok) {
        content.textContent = '';
        const errP = document.createElement('p');
        errP.style.color = 'var(--text-secondary)';
        errP.textContent = 'Could not load app interface.';
        content.appendChild(errP);
        return;
      }

      const ct = resp.headers.get('Content-Type') || '';

      if (ct.includes('application/json')) {
        // SDK v2: XML content response
        const data = await resp.json();
        this._overlayGateway = data.gateway || `/gw/${app.id}`;
        content.textContent = '';
        renderMarkupTo(content, data.content || '');
        this._wireOverlayActions(content);
        this._startOverlayPolling(content);
      } else {
        // SDK v1 legacy: raw HTML
        const html = await resp.text();
        content.innerHTML = html;
        content.querySelectorAll('script').forEach(old => {
          const s = document.createElement('script');
          if (old.src) s.src = old.src;
          else s.textContent = old.textContent;
          old.replaceWith(s);
        });
        if (window.lucide) lucide.createIcons({ node: content });
      }
    } catch (e) {
      console.warn('[apps_panel] app overlay load failed:', e);
      content.textContent = '';
      const errP = document.createElement('p');
      errP.style.color = 'var(--text-secondary)';
      errP.textContent = 'App is not reachable.';
      content.appendChild(errP);
    }
  }

  _closeAppOverlay() {
    // Clear polling timers
    for (const id of this._overlayPollTimers) clearInterval(id);
    this._overlayPollTimers = [];
    this._overlayGateway = null;

    document.getElementById('appOverlay')?.classList.add('hidden');
    const content = document.getElementById('appOverlayContent');
    if (content) content.textContent = '';
  }

  /**
   * Wire execute buttons inside the app overlay.
   * Buttons with data-execute call the daemon capability via gateway,
   * optionally collecting form values and rendering response into a target container.
   */
  _wireOverlayActions(root) {
    root.querySelectorAll('[data-execute]').forEach(btn => {
      btn.addEventListener('click', async () => {
        if (btn.disabled) return;
        btn.disabled = true;

        const capability = btn.dataset.execute;
        const collectId = btn.dataset.collect;
        const targetId = btn.dataset.target;
        const wantsUrl = btn.dataset.openUrl === 'true';
        let params = {};

        // Collect form values if specified
        if (collectId) {
          params = this._collectOverlayFormValues(collectId);
        }

        // Merge static payload if present
        if (btn.dataset.payload) {
          try { Object.assign(params, JSON.parse(btn.dataset.payload)); } catch (e) { console.warn('[apps_panel] invalid payload JSON:', e); }
        }

        try {
          const resp = await fetch(`${this._overlayGateway}/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ capability, params }),
          });
          const data = await resp.json();

          // Open external URL if requested
          if (wantsUrl && data.openUrl) {
            window.open(data.openUrl, '_blank');
          }

          // Render response content into target container
          if (targetId && data.content) {
            const target = root.querySelector(`[data-container-id="${targetId}"]`);
            if (target) {
              target.textContent = '';
              renderMarkupTo(target, data.content);
              this._wireOverlayActions(target);
            }
          } else if (!targetId && data.content) {
            // No target — replace entire overlay content
            root.textContent = '';
            renderMarkupTo(root, data.content);
            this._wireOverlayActions(root);
            this._startOverlayPolling(root);
          }
        } catch (e) {
          console.error('Execute failed:', e);
        } finally {
          btn.disabled = false;
        }
      });
    });
  }

  /**
   * Start polling for containers with data-poll-capability.
   */
  _startOverlayPolling(root) {
    root.querySelectorAll('[data-poll-capability]').forEach(container => {
      const capability = container.dataset.pollCapability;
      const interval = Number.parseInt(container.dataset.pollInterval, 10) || 5000;
      let params = {};
      if (container.dataset.pollParams) {
        try { params = JSON.parse(container.dataset.pollParams); } catch (e) { console.warn('[apps_panel] invalid pollParams JSON:', e); }
      }

      const timerId = setInterval(async () => {
        try {
          const resp = await fetch(`${this._overlayGateway}/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ capability, params }),
          });
          const data = await resp.json();
          if (data.content) {
            container.textContent = '';
            renderMarkupTo(container, data.content);
            this._wireOverlayActions(container);
          }
          // Stop polling if response says so
          if (data.stopPolling) {
            clearInterval(timerId);
            this._overlayPollTimers = this._overlayPollTimers.filter(id => id !== timerId);
          }
        } catch (e) { console.warn('[apps_panel] poll failed:', e); }
      }, interval);

      this._overlayPollTimers.push(timerId);
    });
  }

  /**
   * Collect all form field values from a form block.
   */
  _collectOverlayFormValues(formId) {
    const form = document.querySelector(`[data-form-id="${formId}"]`);
    if (!form) return {};

    const values = {};
    form.querySelectorAll('[data-form-field]').forEach(field => {
      const name = field.dataset.name;
      if (!name) return;

      if (field.tagName === 'INPUT' || field.tagName === 'SELECT') {
        values[name] = field.value;
      } else if (field.dataset.value !== undefined) {
        // Toggle buttons store value in dataset
        values[name] = field.dataset.value === 'true';
      }
    });
    return values;
  }

  async _openAppDetail(appId) {
    const base = this._getHost() ? this._getHost().replace(/\/$/, '') : '';
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
    const base = this._getHost() ? this._getHost().replace(/\/$/, '') : '';

    try {
      await fetch(`${base}/api/apps/${app.id}/scopes`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ approved_scopes: approved }),
      });
      document.getElementById('appDetailDialog')?.close();
      this._loadApps();
    } catch (e) {
      console.error('Failed to save scopes:', e);
    }
  }

  async _removeCurrentApp() {
    const app = this._currentScopeApp;
    if (!app) return;
    const base = this._getHost() ? this._getHost().replace(/\/$/, '') : '';

    try {
      await fetch(`${base}/api/apps/${app.id}`, {
        method: 'DELETE',
        credentials: 'same-origin',
      });
      document.getElementById('appDetailDialog')?.close();
      this._loadApps();
    } catch (e) {
      console.error('Failed to remove app:', e);
    }
  }
}

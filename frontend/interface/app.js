/**
 * Chalie — Application bootstrap & orchestrator.
 */
import { ApiClient } from './api.js';
import { WSClient } from './ws.js';
import { Presence } from './presence.js';
import { Renderer } from './renderer.js';
import { VoiceIO } from './voice.js';
import { ClientHeartbeat } from './heartbeat.js';
import { AmbientSensor } from './ambient.js';
import { MomentSearch } from './moment_search.js';
import { BlockRenderer } from './blocks.js';

// Safe localStorage wrapper — private browsing on iOS Safari / Firefox throws SecurityError.
function _lsGet(key) { try { return localStorage.getItem(key); } catch { return null; } }
function _lsSet(key, val) { try { localStorage.setItem(key, val); } catch { /* ignore */ } }

class ChalieApp {
  constructor() {
    this._backendHost = _lsGet('chalie_backend_host') || '';
    this._isSending = false;
    this._healthRetryTimeout = null;
    this._deferredInstallPrompt = null;

    // Modules
    this.api = new ApiClient(() => this._backendHost);
    this.ws = new WSClient(() => this._backendHost);
    this.heartbeat = new ClientHeartbeat(() => this._backendHost);
    this.presence = null;
    this.renderer = null;
    this.voice = null;

    // Scroll-up pagination state
    this._historyOffset = 0;
    this._historyTotal = 0;
    this._historyLoading = false;
    this._historyExhausted = false;
    this._historyLimit = 12;
    this._historyMaxTurns = 120;

    // Attached images for current message
    this._attachedImages = [];  // [{id: string, element: HTMLElement}]

    // Pending image analysis: image_id → {element: HTMLElement, timeout: number}
    // Populated by _handleImageAttach; cleared by the 'image_ready' WS event or timeout.
    this._pendingImageAnalysis = new Map();

    this._momentSearch = null;
    this._blockRenderer = new BlockRenderer();
    this._overlayPollTimers = [];  // interval IDs for container polling
    this._overlayGateway = null;   // gateway base path for current overlay

    // Web Audio context (unlocked on first user gesture)
    this._audioCtx = null;

    this._init();
  }

  async _init() {
    // Wait for DOM
    if (document.readyState === 'loading') {
      await new Promise(r => document.addEventListener('DOMContentLoaded', r));
    }

    // Unlock AudioContext on first user gesture (required by browsers)
    const unlockAudio = () => {
      if (this._audioCtx && this._audioCtx.state === 'suspended') {
        this._audioCtx.resume();
      }
      document.removeEventListener('click', unlockAudio);
    };
    document.addEventListener('click', unlockAudio);

    // Onboarding guard: redirect to on-boarding only for fresh installs (no account)
    try {
      const r = await fetch('/auth/status', { credentials: 'same-origin' });
      if (r.ok) {
        const data = await r.json();
        if (!data.has_master_account) {
          window.location.replace('/on-boarding/');
          return;
        }
        if (!data.has_session) {
          window.location.replace('/login/');
          return;
        }
      } else {
        window.location.replace('/on-boarding/');
        return;
      }
    } catch (_) { /* backend unreachable — let the app handle it normally */ }

    this._registerServiceWorker();
    this._initInstallPrompt();
    this._checkVisionCapability();
    this._initPresence();
    this._initRenderer();
    this._initVoice();
    this._initTools();
    this._initApps();
    this._initMoments();
    this._initInput();
    this._initUpload();
    this._initAmbientSensor();
    this._initPwaDialog();
    this._initTaskStrip();
    this._initUpdateSystem();
    this._initAmbientCanvas();
    this._initVisibilityTracking();
    this._initConnectionMonitor();
    this._handleSharedContent();
    this.heartbeat.start();

    // Focus mode: glow presence bar when user is in deep focus
    document.addEventListener('chalie:attention', (e) => {
      const bar = document.querySelector('.presence-bar');
      if (bar) {
        bar.dataset.focus = e.detail.attention === 'deep_focus' ? 'deep' : '';
      }
    });

    // Show PWA install prompt first, then resume normal flow after dismiss/install
    await this._showPwaDialogIfNeeded();

    // Show the "Waking up" overlay while we wait for the backend
    this._readyPollActive = true;
    this._showLoadingOverlay();

    // Start the app
    await this._start();
  }

  _registerServiceWorker() {
    if (!('serviceWorker' in navigator)) return;
    navigator.serviceWorker.register('/sw.js').catch(err =>
      console.warn('SW registration failed:', err)
    );
  }

  _initInstallPrompt() {
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      this._deferredInstallPrompt = e;
      document.getElementById('installBtn')?.classList.remove('hidden');
    });
    window.addEventListener('appinstalled', () => {
      this._deferredInstallPrompt = null;
      document.getElementById('installBtn')?.classList.add('hidden');
    });
    document.getElementById('installBtn')?.addEventListener('click', async () => {
      if (!this._deferredInstallPrompt) return;
      this._deferredInstallPrompt.prompt();
      const { outcome } = await this._deferredInstallPrompt.userChoice;
      this._deferredInstallPrompt = null;
      if (outcome === 'accepted') document.getElementById('installBtn')?.classList.add('hidden');
    });
  }

  async _pollUntilReady() {
    const POLL_INTERVAL_MS = 2000;
    const MAX_WAIT_MS = 120_000;
    const deadline = Date.now() + MAX_WAIT_MS;

    while (this._readyPollActive && Date.now() < deadline) {
      const result = await this.api.readyCheck();
      if (result?.ready) return;
      if (!this._readyPollActive) return; // skip was clicked
      await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));
    }
    // Timed out or skipped — proceed anyway so the UI is not permanently blocked
  }

  async _start() {
    try {
      await this._pollUntilReady();
      this._dismissLoadingOverlay();
      this.presence.setState('resting');
      const voiceReady = await this.voice.init();
      if (voiceReady.stt) {
        document.body.classList.add('voice-available');
        document.getElementById('voiceModeBtn')?.classList.remove('hidden');
        if (_lsGet('chalie_voice_mode')) this._enterVoiceMode();
      }
      await this._loadRecentConversation();
      this._loadActiveTasks();
      // Poll every 60s as a safety net for tasks that complete without a drift event
      this._taskStripInterval = setInterval(() => this._loadActiveTasks(), 60_000);
      this._connectWebSocket();
      // Register scroll-up pagination AFTER initial load is done and scroll
      // position is at the bottom — prevents cascade of loads on startup.
      // Only trigger when the user actively scrolls near the top AND the page
      // is scrollable (scrollHeight > viewport), so short conversations don't
      // cascade into loading the full 120-exchange history.
      window.addEventListener('scroll', () => {
        const scrollable = document.documentElement.scrollHeight > window.innerHeight + 100;
        if (scrollable && window.scrollY < 150 && !this._historyLoading && !this._historyExhausted) {
          const anchor = document.body.scrollHeight - window.scrollY;
          this._loadRecentConversation().then(() => {
            window.scrollTo(0, document.body.scrollHeight - anchor);
          });
        }
      });
      window.addEventListener('beforeunload', () => {
        this.ws.close();
        clearInterval(this._taskStripInterval);
      }, { once: true });
      this._requestNotificationPermission();
      // Ask once for geolocation permission so the heartbeat can capture coordinates.
      // The browser shows its own permission dialog; we don't block on the answer.
      this.heartbeat.requestLocationPermission();
    } catch {
      this.presence.setState('error');
      this._showConnectionBanner();
    }
  }

  // ---------------------------------------------------------------------------
  // Presence
  // ---------------------------------------------------------------------------

  _initPresence() {
    const dot = document.querySelector('.presence-dot');
    const label = document.querySelector('.presence-label');
    this.presence = new Presence(dot, label);
  }

  // ---------------------------------------------------------------------------
  // Renderer
  // ---------------------------------------------------------------------------

  _initRenderer() {
    const spine = document.getElementById('conversationSpine');
    this.renderer = new Renderer(spine);
  }

  // ---------------------------------------------------------------------------
  // Voice
  // ---------------------------------------------------------------------------

  _initVoice() {
    this.voice = new VoiceIO(() => this._backendHost);
    this._voiceMode = false;
    this._initVoiceMode();
  }

  // ---------------------------------------------------------------------------
  // Voice Mode
  // ---------------------------------------------------------------------------

  _initVoiceMode() {
    document.getElementById('voiceModeBtn')?.addEventListener('click', () => {
      if (this._voiceMode) this._exitVoiceMode();
      else this._enterVoiceMode();
    });

    document.getElementById('voiceModeExit')?.addEventListener('click', () => {
      this._exitVoiceMode();
    });

    // Orb click — state machine
    document.getElementById('voiceOrb')?.addEventListener('click', async () => {
      // Unlock autoplay on every user gesture so TTS can play later
      this.voice.unlockAudio();

      const orb = document.getElementById('voiceOrb');
      const state = orb?.dataset.state;

      if (state === 'idle') {
        this._lastVoiceText = null;
        const sub = document.getElementById('voiceSubtitle');
        if (sub) { sub.textContent = ''; sub.classList.add('hidden'); }
        await this.voice.startRecording();
        if (this.voice._isRecording) this._setOrbState('listening');
      } else if (state === 'listening') {
        this._setOrbState('thinking');
        const text = await this.voice.stopRecording();
        if (text) {
          this._sendVoiceMessage(text);
        } else {
          this._setOrbState('idle');
        }
      }
    });

    // Auto-stop at 10-minute limit
    document.addEventListener('chalie:recording:maxed', async () => {
      if (!this._voiceMode) return;
      this._showToast('10-min max per voice message');
      this._setOrbState('thinking');
      const text = await this.voice.stopRecording();
      if (text) {
        this._sendVoiceMessage(text);
      } else {
        this._setOrbState('idle');
      }
    });

    // Subtitle: show current sentence during playback
    document.addEventListener('chalie:speak:chunk', (e) => {
      if (!this._voiceMode) return;
      const el = document.getElementById('voiceSubtitle');
      if (el) {
        el.textContent = e.detail.text;
        el.classList.remove('hidden');
      }
    });

    // TTS finished — back to idle (controls stay visible for re-listen)
    document.addEventListener('chalie:speak:done', () => {
      if (this._voiceMode) {
        this._setOrbState('idle');
        this._setPlayPauseIcon(false);
      }
    });
    document.addEventListener('chalie:speak:error', () => {
      if (this._voiceMode) {
        this._hasPlayback = false;
        this._setOrbState('idle');
        const sub = document.getElementById('voiceSubtitle');
        if (sub) { sub.textContent = ''; sub.classList.add('hidden'); }
      }
    });

    // Media controls
    document.getElementById('voiceCtrlForward')?.addEventListener('click', () => this.voice.skipForward());
    document.getElementById('voiceCtrlBack')?.addEventListener('click', () => this.voice.skipBack());
    document.getElementById('voiceCtrlPause')?.addEventListener('click', () => {
      this.voice.unlockAudio();

      // If playback finished and user presses play — replay from start
      if (!this.voice._speaking && this._hasPlayback && this._lastVoiceText) {
        this._setOrbState('speaking');
        this._setPlayPauseIcon(true);
        this.voice.speak(this._lastVoiceText);
        return;
      }

      const playing = this.voice.togglePause();
      this._setPlayPauseIcon(playing);
    });

    // Speak button on chat messages — enter voice mode and play
    document.addEventListener('chalie:speak-message', (e) => {
      const text = e.detail?.text;
      if (!text || !this.voice._available) return;
      this.voice.unlockAudio();
      this._enterVoiceMode();
      this._lastVoiceText = text;
      this._setOrbState('speaking');
      this._setPlayPauseIcon(true);
      this.voice.speak(text);
    });
  }

  _enterVoiceMode() {
    if (!this.voice._available) return;
    this._voiceMode = true;
    _lsSet('chalie_voice_mode', '1');

    document.getElementById('voiceModeOverlay')?.classList.remove('hidden');
    document.getElementById('conversationSpine')?.classList.add('hidden');
    document.querySelector('.input-dock')?.classList.add('hidden');
    document.getElementById('taskStrip')?.classList.add('hidden');
    document.getElementById('voiceModeBtn')?.classList.add('active');

    this._setOrbState('idle');
  }

  _exitVoiceMode() {
    this._voiceMode = false;
    _lsSet('chalie_voice_mode', '');

    // Stop any active recording or playback
    if (this.voice._isRecording) this.voice.stopRecording();
    this.voice.stopAudio();

    document.getElementById('voiceModeOverlay')?.classList.add('hidden');
    document.getElementById('conversationSpine')?.classList.remove('hidden');
    document.querySelector('.input-dock')?.classList.remove('hidden');
    document.getElementById('voiceModeBtn')?.classList.remove('active');

    this._loadActiveTasks();
  }

  _setOrbState(state) {
    const orb = document.getElementById('voiceOrb');
    const controls = document.getElementById('voiceModeControls');
    const label = document.getElementById('voiceModeLabel');

    if (orb) orb.dataset.state = state;
    if (label) {
      const labels = { idle: '', listening: 'Listening...', thinking: 'Thinking...', speaking: '' };
      label.textContent = labels[state] || '';
    }

    if (controls) {
      if (state === 'speaking') {
        this._hasPlayback = true;
        controls.classList.remove('hidden');
      } else if (state === 'listening' || state === 'thinking') {
        this._hasPlayback = false;
        controls.classList.add('hidden');
      }
    }

    // Audio-reactive visualizer
    if (state === 'speaking' && orb) {
      this.voice.startVisualizer(orb);
    } else {
      this.voice.stopVisualizer();
      if (orb) {
        orb.style.removeProperty('--orb-energy');
        orb.style.removeProperty('--orb-bass');
      }
    }
  }

  _setPlayPauseIcon(playing) {
    const btn = document.getElementById('voiceCtrlPause');
    if (!btn) return;
    btn.innerHTML = playing
      ? '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>'
      : '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
  }

  _sendVoiceMessage(text) {
    this.presence.setState('processing');

    // Render user message in the hidden spine (preserves history)
    const ts = new Date();
    this.renderer.appendUserForm(text, ts);

    let responseBlocks = [];

    this.ws.send(text, 'voice', {
      onStatus: (stage) => {
        this.presence.setState(stage);
      },
      onNarration: (data) => {
        this.renderer.appendNarrationBubble?.(data.text, data.step);
      },
      onMessage: (data) => {
        responseBlocks = data.blocks || [];
      },
      onError: () => {
        this._setOrbState('idle');
      },
      onDone: (data) => {
        // Render to spine for history
        if (responseBlocks.length) {
          this.renderer.appendChalieForm(responseBlocks, {
            topic: '', mode: '', confidence: 0, ts, duration_ms: data.duration_ms,
          });
        }
        this.presence.setState('resting');

        // Auto-TTS
        const plainText = this.renderer._extractPlainText(responseBlocks);
        if (plainText) {
          this._lastVoiceText = plainText;
          this._setOrbState('speaking');
          this._setPlayPauseIcon(true);
          this.voice.speak(plainText);
        } else {
          this._setOrbState('idle');
        }
      },
    });
  }

  // ---------------------------------------------------------------------------
  // Tools
  // ---------------------------------------------------------------------------

  _initTools() {
    document.getElementById('settingsBtn')?.addEventListener('click', () => {
      window.open('/brain/', '_blank');
    });
  }

  // ---------------------------------------------------------------------------
  // Apps (interface daemons)
  // ---------------------------------------------------------------------------

  _initApps() {
    this._apps = [];
    this._appsOpen = false;
    this._currentScopeApp = null;

    const appsBtn = document.getElementById('appsBtn');
    const panel = document.getElementById('appsPanel');
    const closePanel = document.getElementById('closeAppsPanel');
    const overlay = document.getElementById('appOverlay');
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

  async _loadApps() {
    try {
      const base = this._backendHost ? this._backendHost.replace(/\/$/, '') : '';
      const resp = await fetch(`${base}/api/apps`, { credentials: 'same-origin' });
      if (!resp.ok) return;
      this._apps = await resp.json();
      this._renderAppsList();
    } catch (_) { /* dashboard may not be running */ }
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
            <div class="scope-item__label">${this._esc(key)}</div>
            ${desc ? `<div class="scope-item__desc">${this._esc(String(desc))}</div>` : ''}
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
    const base = this._backendHost ? this._backendHost.replace(/\/$/, '') : '';

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
        // SDK v2: blocks response
        const data = await resp.json();
        this._overlayGateway = data.gateway || `/gw/${app.id}`;
        content.textContent = '';
        const fragment = this._blockRenderer.render(data.blocks || []);
        content.appendChild(fragment);
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
    } catch (_) {
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
          try { Object.assign(params, JSON.parse(btn.dataset.payload)); } catch (_) {}
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

          // Render response blocks into target container
          if (targetId && data.blocks) {
            const target = root.querySelector(`[data-container-id="${targetId}"]`);
            if (target) {
              target.textContent = '';
              target.appendChild(this._blockRenderer.render(data.blocks));
              this._wireOverlayActions(target);
            }
          } else if (!targetId && data.blocks) {
            // No target — replace entire overlay content
            root.textContent = '';
            root.appendChild(this._blockRenderer.render(data.blocks));
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
      const interval = parseInt(container.dataset.pollInterval, 10) || 5000;
      let params = {};
      if (container.dataset.pollParams) {
        try { params = JSON.parse(container.dataset.pollParams); } catch (_) {}
      }

      const timerId = setInterval(async () => {
        try {
          const resp = await fetch(`${this._overlayGateway}/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ capability, params }),
          });
          const data = await resp.json();
          if (data.blocks) {
            container.textContent = '';
            container.appendChild(this._blockRenderer.render(data.blocks));
            this._wireOverlayActions(container);
          }
          // Stop polling if response says so
          if (data.stopPolling) {
            clearInterval(timerId);
            this._overlayPollTimers = this._overlayPollTimers.filter(id => id !== timerId);
          }
        } catch (_) {}
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
    const base = this._backendHost ? this._backendHost.replace(/\/$/, '') : '';
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
          caps.map(c => `<div class="app-detail__cap-item">${this._esc(c.name || c.id || JSON.stringify(c))}</div>`).join('');
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
    const base = this._backendHost ? this._backendHost.replace(/\/$/, '') : '';

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
    const base = this._backendHost ? this._backendHost.replace(/\/$/, '') : '';

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

  // ---------------------------------------------------------------------------
  // Moments
  // ---------------------------------------------------------------------------

  _initMoments() {
    // Search overlay
    const backendHost = this._backendHost;
    this._momentSearch = new MomentSearch((path) => {
      const base = backendHost ? backendHost.replace(/\/$/, '') : '';
      return fetch(base + path, { credentials: 'same-origin' });
    });

    // Recall button in header
    document.getElementById('recallBtn')?.addEventListener('click', () => {
      this._momentSearch.open();
    });

    // Action button click (deterministic skill invocation, bypasses LLM)
    document.addEventListener('chalie:action', (e) => {
      const { payload } = e.detail;
      if (!payload || this._isSending) return;

      this._isSending = true;
      const pendingForm = this.renderer.createPendingForm();

      this.ws.sendAction(payload, {
        onMessage: (data) => {
          const blocks = data.blocks || [];
          const meta = {
            mode: data.mode || 'ACT',
            confidence: data.confidence || 0.95,
          };
          if (pendingForm.isConnected) {
            this.renderer.resolvePendingForm(pendingForm, blocks, meta);
          } else {
            this.renderer.appendChalieForm(blocks, meta);
          }
        },
        onError: (data) => {
          this.renderer.resolvePendingFormError(pendingForm, data.message);
        },
        onDone: () => {
          this._isSending = false;
          this.presence.setState('resting');
        },
      });
    });

    // Pin moment event (from remember button on Chalie messages)
    let pinDebounce = 0;
    document.addEventListener('chalie:pin-moment', async (e) => {
      const now = Date.now();
      if (now - pinDebounce < 250) return; // 250ms debounce
      pinDebounce = now;

      const { text, meta } = e.detail;
      const body = {
        message_text: text,
        exchange_id: meta.exchange_id || '',
        topic: meta.topic || '',
        thread_id: meta.thread_id || '',
      };

      try {
        const base = backendHost ? backendHost.replace(/\/$/, '') : '';
        const res = await fetch(base + '/moments', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify(body),
        });

        if (res.ok) {
          const data = await res.json();
          const momentId = data.item?.id;
          const isDuplicate = data.duplicate;

          const msg = isDuplicate ? 'Already remembered' : 'Remembered';
          this._showToast(msg, momentId ? () => this._undoMoment(momentId) : null);
        }
      } catch (err) {
        console.warn('Pin moment failed:', err);
      }
    });

    // First-use hint (one-time)
    if (!_lsGet('moments_hint_shown')) {
      this._showMomentsHintOnFirstResponse();
    }
  }

  async _undoMoment(momentId) {
    try {
      const base = this._backendHost ? this._backendHost.replace(/\/$/, '') : '';
      await fetch(base + `/moments/${momentId}/forget`, {
        method: 'POST',
        credentials: 'same-origin',
      });
    } catch (err) {
      console.warn('Undo moment failed:', err);
    }
  }

  _showToast(message, onUndo, duration = 4000) {
    // Remove existing toast
    document.querySelector('.chalie-toast')?.remove();

    const toast = document.createElement('div');
    toast.className = 'chalie-toast';
    const span = document.createElement('span');
    span.textContent = message;
    toast.appendChild(span);

    if (onUndo) {
      const undoBtn = document.createElement('button');
      undoBtn.className = 'chalie-toast__undo';
      undoBtn.textContent = 'Undo';
      undoBtn.addEventListener('click', () => {
        onUndo();
        toast.remove();
      });
      toast.appendChild(undoBtn);
    }

    document.body.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('chalie-toast--visible'));

    setTimeout(() => {
      toast.classList.remove('chalie-toast--visible');
      setTimeout(() => toast.remove(), 250);
    }, duration);
  }

  _showMomentsHintOnFirstResponse() {
    // Wait for first Chalie response to show hint
    const observer = new MutationObserver((mutations) => {
      for (const m of mutations) {
        for (const node of m.addedNodes) {
          if (node.classList?.contains('speech-form--chalie')) {
            const btn = node.querySelector('.speech-form__remember-btn');
            if (btn && !_lsGet('moments_hint_shown')) {
              _lsSet('moments_hint_shown', '1');
              observer.disconnect();

              // Show tooltip near the button
              const hint = document.createElement('div');
              hint.className = 'chalie-toast';
              hint.innerHTML = '<span>Remember important answers to find them later.</span>';
              document.body.appendChild(hint);
              requestAnimationFrame(() => hint.classList.add('chalie-toast--visible'));
              setTimeout(() => {
                hint.classList.remove('chalie-toast--visible');
                setTimeout(() => hint.remove(), 250);
              }, 5000);
            }
            return;
          }
        }
      }
    });
    observer.observe(document.getElementById('conversationSpine'), { childList: true });
  }

  // ---------------------------------------------------------------------------
  // Persistent Task Strip
  // ---------------------------------------------------------------------------

  _initTaskStrip() {
    const toggle = document.getElementById('taskStripToggle');
    const strip = document.getElementById('taskStrip');
    if (!toggle || !strip) return;

    toggle.addEventListener('click', () => {
      strip.classList.toggle('--expanded');
    });
  }

  async _loadActiveTasks() {
    try {
      const [taskData, schedData] = await Promise.all([
        this.api._get('/system/observability/tasks').catch((e) => { if (e?.message === 'AUTH') throw e; return {}; }),
        this.api._get('/scheduler?status=pending').catch((e) => { if (e?.message === 'AUTH') throw e; return {}; }),
      ]);
      const tasks = (taskData.persistent_tasks || []).filter(
        t => t.status === 'accepted' || t.status === 'in_progress' || t.status === 'paused'
      );
      const reminders = (schedData.items || []).filter(
        r => r.status === 'pending' && r.due_at
      );
      this._renderTaskStrip(tasks, reminders);
    } catch (e) {
      if (e?.message === 'AUTH') {
        this._handleAuthFailure();
      }
      // Other errors: silently fail — task strip is supplementary
    }
  }

  _renderTaskStrip(tasks, reminders = []) {
    const strip = document.getElementById('taskStrip');
    const list = document.getElementById('taskStripList');
    const countEl = document.getElementById('taskStripCount');
    if (!strip || !list) return;

    const totalCount = tasks.length + reminders.length;
    const wasEmpty = strip.classList.contains('hidden');

    if (totalCount === 0) {
      strip.classList.add('hidden');
      strip.classList.remove('--expanded');
      document.body.classList.remove('has-task-strip');
      return;
    }

    strip.classList.remove('hidden');
    document.body.classList.add('has-task-strip');
    countEl.textContent = totalCount;

    // Auto-expand when items appear for the first time (was hidden → now visible)
    if (wasEmpty) {
      strip.classList.add('--expanded');
    }

    const frag = document.createDocumentFragment();

    // Render persistent tasks
    for (const t of tasks) {
      const goal = (t.goal || 'Working\u2026').slice(0, 60);
      const progress = t.progress || {};
      const coverage = Math.round((progress.coverage_estimate || 0) * 100);
      const summary = progress.last_summary || '';

      const item = document.createElement('div');
      item.className = 'task-strip__item';
      if (t.status === 'paused') item.classList.add('--paused');

      const dot = document.createElement('span');
      dot.className = 'task-strip__kind-dot task-strip__kind-dot--task';
      item.appendChild(dot);

      const goalEl = document.createElement('div');
      goalEl.className = 'task-strip__goal';
      goalEl.textContent = goal;
      item.appendChild(goalEl);

      const bar = document.createElement('div');
      bar.className = 'task-strip__progress-bar';
      const fill = document.createElement('div');
      fill.className = 'task-strip__progress-fill';
      fill.style.width = `${coverage}%`;
      bar.appendChild(fill);
      item.appendChild(bar);

      // Step-level progress from plan DAG
      const plan = progress.plan;
      if (plan && plan.steps && plan.steps.length > 0) {
        const done = plan.steps.filter(s => s.status === 'completed' || s.status === 'skipped').length;
        const total = plan.steps.length;
        const current = plan.steps.find(s => s.status === 'in_progress');

        const stepsEl = document.createElement('div');
        stepsEl.className = 'task-strip__steps';
        stepsEl.textContent = `${done}/${total} steps`;
        item.appendChild(stepsEl);

        if (current) {
          const curEl = document.createElement('div');
          curEl.className = 'task-strip__current-step';
          curEl.textContent = current.description;
          item.appendChild(curEl);
        }
        if (plan.blocked_on) {
          const blockedEl = document.createElement('div');
          blockedEl.className = 'task-strip__blocked';
          blockedEl.textContent = `Blocked: ${plan.blocked_reason || 'dependency failed'}`;
          item.appendChild(blockedEl);
        }
      }

      if (summary) {
        const sumEl = document.createElement('div');
        sumEl.className = 'task-strip__summary';
        sumEl.textContent = summary;
        item.appendChild(sumEl);
      }

      const dismissBtn = document.createElement('button');
      dismissBtn.className = 'task-strip__dismiss';
      dismissBtn.setAttribute('aria-label', 'Dismiss task');
      dismissBtn.textContent = '\u00d7';
      dismissBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
          await this.api._delete(`/system/observability/tasks/${t.id}`);
        } catch { /* ignore */ }
        this._loadActiveTasks();
      });
      item.appendChild(dismissBtn);

      frag.appendChild(item);
    }

    // Render pending reminders
    for (const r of reminders) {
      const msg = (r.message || '').slice(0, 80);
      const due = r.due_at ? this._relativeTime(r.due_at) : '';

      const item = document.createElement('div');
      item.className = 'task-strip__item task-strip__item--reminder';

      const dot = document.createElement('span');
      dot.className = 'task-strip__kind-dot task-strip__kind-dot--reminder';
      item.appendChild(dot);

      const msgEl = document.createElement('span');
      msgEl.className = 'task-strip__msg';
      msgEl.textContent = msg;
      item.appendChild(msgEl);

      if (due) {
        const dueEl = document.createElement('span');
        dueEl.className = 'task-strip__due';
        dueEl.textContent = due;
        item.appendChild(dueEl);
      }

      const dismissBtn = document.createElement('button');
      dismissBtn.className = 'task-strip__dismiss';
      dismissBtn.setAttribute('aria-label', 'Dismiss reminder');
      dismissBtn.textContent = '\u00d7';
      dismissBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
          await this.api._delete(`/scheduler/${r.id}`);
        } catch { /* ignore */ }
        this._loadActiveTasks();
      });
      item.appendChild(dismissBtn);

      frag.appendChild(item);
    }

    // First-time hint
    if (!_lsGet('task_strip_hint_shown')) {
      _lsSet('task_strip_hint_shown', '1');
      const hint = document.createElement('div');
      hint.className = 'task-strip__hint';
      hint.textContent = 'I\'ll show what I\'m working on here.';
      frag.appendChild(hint);
    }

    list.textContent = '';
    list.appendChild(frag);
  }

  /** Convert an ISO date string to a short relative label ("in 5m", "in 2h", "tomorrow"). */
  _relativeTime(isoStr) {
    try {
      const due = new Date(isoStr);
      const now = Date.now();
      const diffMs = due.getTime() - now;
      if (diffMs < 0) return 'overdue';
      const mins = Math.round(diffMs / 60000);
      if (mins < 1) return 'now';
      if (mins < 60) return `in ${mins}m`;
      const hrs = Math.round(mins / 60);
      if (hrs < 24) return `in ${hrs}h`;
      const days = Math.round(hrs / 24);
      if (days === 1) return 'tomorrow';
      return `in ${days}d`;
    } catch {
      return '';
    }
  }

  _escHtml(str) {
    if (!str) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  // ---------------------------------------------------------------------------
  // Ambient Sensor
  // ---------------------------------------------------------------------------

  _initAmbientSensor() {
    this._ambientSensor = new AmbientSensor();
    this._ambientSensor.bindTypingInput(document.getElementById('messageInput'));
    this.heartbeat.setAmbientSensor(this._ambientSensor);
  }

  // ---------------------------------------------------------------------------
  // Share Target
  // ---------------------------------------------------------------------------

  _handleSharedContent() {
    const params = new URLSearchParams(window.location.search);
    const shared = params.get('shared');
    if (!shared) return;

    // Pre-fill the prompt textarea with shared content
    const textarea = document.getElementById('messageInput');
    if (textarea) {
      textarea.value = decodeURIComponent(shared);
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
      const sendBtn = document.getElementById('sendBtn');
      if (sendBtn) sendBtn.disabled = false;
    }

    // Clean URL without reload
    const cleanUrl = window.location.pathname;
    window.history.replaceState({}, '', cleanUrl);
  }

  // ---------------------------------------------------------------------------
  // Input Handling
  // ---------------------------------------------------------------------------

  _initInput() {
    const textarea = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');

    // Auto-resize textarea
    textarea.addEventListener('input', () => {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
      sendBtn.disabled = !textarea.value.trim() && !this._attachedImages.length;
    });

    // Enter to send (Shift+Enter for newline)
    textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this._sendMessage();
      }
    });

    // Send button click
    sendBtn.addEventListener('click', () => this._sendMessage());
  }

  async _sendMessage(source = 'text') {
    const textarea = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const text = textarea.value.trim();
    const imageIds = this._attachedImages.map(a => a.id);

    if (!text && !imageIds.length) return;

    // Only route as steer when the ACT loop is actively narrating.
    // This prevents normal replies (e.g. to clarifications) from being
    // misrouted as steering commands.
    if (this.ws._chatCallbacks && this.presence.state === 'narrating') {
      textarea.value = '';
      textarea.style.height = 'auto';
      this.ws.send(text, source, {});
      return;
    }

    this._isSending = true;
    this.presence.setState('processing');
    textarea.value = '';
    textarea.style.height = 'auto';
    const pendingImageIds = [...imageIds];
    this._clearImagePreview();
    sendBtn.disabled = true;

    // Capture timestamp for this exchange
    const exchangeTimestamp = new Date();

    // Render user form with timestamp (pass imageIds so thumbnails are shown inline)
    this.renderer.appendUserForm(text || '[Image attached]', exchangeTimestamp, {}, pendingImageIds);

    // Create pending form and store reference for potential early resolution
    const pendingForm = this.renderer.createPendingForm();
    this._pendingForm = pendingForm;

    // After 2 seconds, upgrade the "..." dots to a brief placeholder phrase
    const pendingUpgradeTimer = setTimeout(() => {
      this.renderer.upgradePendingText(pendingForm);
    }, 2000);

    let responseBlocks = [];
    let responseMeta = {};

    this.ws.send(text || '[Image attached]', source, {
      onStatus: (stage) => {
        this.presence.setState(stage);
      },
      onNarration: (data) => {
        // Live ACT narration: remove thinking dots, show narration bubble instead
        clearTimeout(pendingUpgradeTimer);
        // Remove the pending form entirely — narration bubbles replace it
        if (pendingForm.isConnected) pendingForm.remove();
        this.presence.setState('narrating');
        this.renderer.appendNarrationBubble(data.text, data.step);
      },
      onSteerSent: (steerText) => {
        // User sent a redirect while ACT is running — show inline
        this.renderer.appendSteerBubble(steerText);
      },
      onMessage: (data) => {
        clearTimeout(pendingUpgradeTimer);
        responseBlocks = data.blocks || [];
        responseMeta = {
          topic: data.topic,
          exchange_id: data.exchange_id,
          mode: data.mode || '',
          confidence: data.confidence || 0,
        };
        this.presence.setState('responding');
      },
      onError: (data) => {
        clearTimeout(pendingUpgradeTimer);
        this.renderer.resolvePendingFormError(pendingForm, data.message);
        if (!data.recoverable) {
          this._handleAuthFailure();
        }
      },
      onDone: (data) => {
        clearTimeout(pendingUpgradeTimer);
        if (this._ambientSensor) this._ambientSensor.recordResponse();
        if (responseBlocks.length) {
          responseMeta.duration_ms = data.duration_ms;
          responseMeta.ts = exchangeTimestamp;
          if (pendingForm.isConnected) {
            // No narrations — resolve the pending form in place
            this.renderer.resolvePendingForm(pendingForm, responseBlocks, responseMeta);
          } else {
            // Narrations replaced the pending form — append as a new message
            this.renderer.appendChalieForm(responseBlocks, responseMeta);
          }
          this._pendingForm = null;
          // Notify if user switched away while waiting for the response
          if (!document.hasFocus()) {
            const notifText = responseBlocks
              .filter(b => b.type === 'text' || b.type === 'header')
              .map(b => b.content || '')
              .join(' ');
            if (notifText) this._notifyBackground(notifText);
          }
        } else {
          // No blocks response — remove the pending bubble (if still in DOM)
          if (pendingForm.isConnected) pendingForm.remove();
          this._pendingForm = null;
        }
        this.presence.setState('resting');
        this._isSending = false;
        // Re-enable input
        document.getElementById('messageInput').focus();
      },
    }, pendingImageIds);

    // Safety: if onDone never fires (connection drop), reset after 5 minutes
    setTimeout(() => {
      if (this._isSending) {
        this._isSending = false;
        this._pendingForm = null;
      }
    }, 300000);
  }

  // ---------------------------------------------------------------------------
  // Load Recent Conversation
  // ---------------------------------------------------------------------------

  async _loadRecentConversation() {
    if (this._historyLoading || this._historyExhausted) return;
    this._historyLoading = true;

    const loader = document.getElementById('historyLoader');
    if (loader) loader.style.display = 'flex';

    try {
      const data = await this.api.getRecentConversation({
        limit: this._historyLimit,
        offset: this._historyOffset,
      });

      const exchanges = data.exchanges || [];

      if (exchanges.length === 0 && this._historyOffset === 0) {
        this._historyExhausted = true;
        this._showHistoryEndPill();
        return;
      }

      const isInitialLoad = this._historyOffset === 0;
      if (isInitialLoad) {
        // Initial load — append in chronological order; use in_working_memory from API
        for (const exchange of exchanges) {
          this._appendExchange(exchange, exchange.in_working_memory !== false);
        }
      } else {
        // Subsequent pages — prepend in reverse so oldest ends up at top
        for (let i = exchanges.length - 1; i >= 0; i--) {
          this._prependExchange(exchanges[i], false);
        }
      }

      this._historyOffset += exchanges.length;
      this._historyTotal = data.total ?? this._historyOffset;

      if (!data.has_more || this._historyOffset >= this._historyMaxTurns) {
        this._historyExhausted = true;
        this._showHistoryEndPill();
      }

      // After initial load, force-scroll to show the latest message.
      // Uses instant scroll to avoid race with smooth scroll + scroll tracking.
      if (isInitialLoad && exchanges.length > 0) {
        this.renderer.forceScrollToBottom();
      }
    } catch (err) {
      if (err.message === 'AUTH') {
        this._handleAuthFailure();
      }
      // Otherwise silently fail — conversation history is nice-to-have
    } finally {
      if (loader) loader.style.display = 'none';
      this._historyLoading = false;
    }
  }

  _appendExchange(exchange, inWorkingMemory) {
    if (exchange.prompt) {
      this.renderer.appendUserForm(exchange.prompt, exchange.timestamp, { inWorkingMemory });
    }
    if (exchange.blocks?.length) {
      this.renderer.appendChalieForm(exchange.blocks, {
        topic: exchange.topic,
        ts: exchange.timestamp,
        exchange_id: exchange.id,
      }, { inWorkingMemory });
    }
  }

  _prependExchange(exchange, inWorkingMemory) {
    if (exchange.blocks?.length) {
      this.renderer.prependChalieForm(exchange.blocks, {
        topic: exchange.topic,
        ts: exchange.timestamp,
        exchange_id: exchange.id,
      }, { inWorkingMemory });
    }
    if (exchange.prompt) {
      this.renderer.prependUserForm(exchange.prompt, exchange.timestamp, { inWorkingMemory });
    }
  }

  _showHistoryEndPill() {
    const pill = document.getElementById('historyEndPill');
    if (pill) pill.style.display = 'flex';
  }

  _showLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    const spine = document.getElementById('conversationSpine');
    const dock = document.querySelector('.input-dock');
    if (!overlay) return;

    overlay.classList.remove('hidden');
    if (spine) spine.style.display = 'none';
    if (dock) dock.style.display = 'none';

    // Skip button
    const skipBtn = overlay.querySelector('.loading-overlay__skip');
    if (skipBtn) {
      skipBtn.addEventListener('click', () => {
        this._readyPollActive = false;
        this._dismissLoadingOverlay();
      }, { once: true });
    }
  }

  _dismissLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    const spine = document.getElementById('conversationSpine');
    const dock = document.querySelector('.input-dock');

    if (overlay && !overlay.classList.contains('hidden')) {
      overlay.classList.add('loading-overlay--fading');
      setTimeout(() => {
        overlay.classList.add('hidden');
        overlay.classList.remove('loading-overlay--fading');
      }, 220);
    }

    if (spine) spine.style.display = '';
    if (dock) dock.style.display = '';
  }

  // ---------------------------------------------------------------------------
  // WebSocket (drift, tool follow-ups, delegate results, chat)
  // ---------------------------------------------------------------------------

  _connectWebSocket() {
    this.ws.onDrift((data) => this._handleEvent(data));
    this.ws.connect();
  }

  _handleEvent(data) {
    // App update notification
    if (data.type === 'app_update') {
      this._handleUpdateEvent(data);
      return;
    }

    // Task progress/completion — refresh the task strip (sticky bar only)
    if (data.type === 'task') {
      this._loadActiveTasks();
      return;
    }

    // TTS audio chunks pushed via WebSocket (replaces HTTP streaming)
    if (data.type === 'tts_chunk') {
      this.voice?.handleTtsChunk(data);
      return;
    }
    if (data.type === 'tts_done') {
      this.voice?.handleTtsDone();
      return;
    }

    // Step 4 / B4+B5 fix: image analysis completion pushed by _run_analysis via
    // output:events.  Remove the spinner on success, or show an error badge so
    // the user can decide whether to remove the failed image.
    if (data.type === 'image_ready') {
      const pending = this._pendingImageAnalysis.get(data.image_id);
      if (pending) {
        clearTimeout(pending.timeout);
        this._pendingImageAnalysis.delete(data.image_id);
        pending.element.classList.remove('analyzing');
        pending.element.querySelector('.image-preview__spinner')?.remove();
        if (data.status === 'failed') {
          // Surface the failure with an error badge on the thumbnail.
          const errBadge = document.createElement('span');
          errBadge.className = 'image-preview__error';
          errBadge.title = 'Image analysis failed — context unavailable';
          errBadge.textContent = '✕';
          pending.element.appendChild(errBadge);
          this._showToast?.('Image analysis failed');
        }
      }
      return;
    }

    const blocks = data.blocks || [];
    if (!blocks.length) return;

    // Ignore 'response' events from the drift stream while a /chat SSE request
    // is in flight — the chat SSE already renders the reply via resolvePendingForm.
    if (data.type === 'response' && this._isSending) return;

    // System notification + sound when tab is not focused
    if (!document.hasFocus()) {
      const notifText = blocks
        .filter(b => b.type === 'text' || b.type === 'header')
        .map(b => b.content || '')
        .join(' ');
      if (notifText) this._notifyBackground(notifText);
    }

    // Scheduler trigger events — refresh task strip, play sound
    if (data.type === 'notification') {
      this._playScheduleSound();
      this._loadActiveTasks();
      return;
    }

    const meta = {
      topic: data.topic,
      type: data.type,
      ts: new Date(),
      mode: data.mode || '',
      confidence: data.confidence || 0,
    };

    // Render in conversation spine as a Chalie message
    const formEl = this.renderer.appendChalieForm(blocks, meta);


    // Critic escalation — amber border to signal "needs your attention"
    if (data.type === 'escalation') {
      formEl.classList.add('speech-form--escalation');
    }
  }

  _playScheduleSound() {
    try {
      if (!this._audioCtx) {
        this._audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      const ctx = this._audioCtx;
      if (ctx.state === 'suspended') ctx.resume();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = 880; // A5
      osc.type = 'sine';
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.5);
    } catch (_) {
      // AudioContext unavailable or blocked — silently skip
    }
  }

  /**
   * Show system notification + play sound when the tab is not focused.
   * Uses ServiceWorkerRegistration.showNotification() so the tag deduplicates
   * against push notifications from sw.js (both use 'chalie-message').
   */
  _notifyBackground(text) {
    if (Notification.permission !== 'granted') return;

    const body = text.length > 200 ? text.slice(0, 200) + '…' : text;

    // System notification via SW registration (shared tag prevents duplicates with push)
    if (navigator.serviceWorker?.controller) {
      navigator.serviceWorker.ready.then(reg => {
        reg.showNotification('Chalie', {
          body,
          tag: 'chalie-message',
          data: { url: '/' },
        });
      }).catch(() => {});
    } else {
      // Fallback: Notification API directly (no SW available)
      try { new Notification('Chalie', { body, tag: 'chalie-message' }); } catch (_) {}
    }

    // Audible chime — Web Audio may be throttled in hidden tabs but works
    // when the window is just unfocused (another app in foreground).
    this._playScheduleSound();
  }

  async _requestNotificationPermission() {
    if (!('serviceWorker' in navigator) || !('PushManager' in window)) return;

    try {
      const reg = await navigator.serviceWorker.ready;

      // Request notification permission
      const permission = await Notification.requestPermission();
      if (permission !== 'granted') return;

      // Get VAPID public key from backend
      const vapidUrl = this._backendHost
        ? this._backendHost.replace(/\/$/, '') + '/push/vapid-key'
        : '/push/vapid-key';
      const res = await fetch(vapidUrl);
      if (!res.ok) return;
      const { publicKey } = await res.json();

      // Convert URL-safe base64 to Uint8Array
      const applicationServerKey = this._urlBase64ToUint8Array(publicKey);

      // Subscribe (or get existing subscription)
      let subscription = await reg.pushManager.getSubscription();
      if (!subscription) {
        subscription = await reg.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey,
        });
      }

      // Send subscription to backend
      const subscribeUrl = this._backendHost
        ? this._backendHost.replace(/\/$/, '') + '/push/subscribe'
        : '/push/subscribe';
      await fetch(subscribeUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify(subscription.toJSON()),
      });
    } catch (err) {
      console.warn('Push subscription failed:', err);
    }
  }

  _urlBase64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
    const raw = atob(base64);
    const output = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      output[i] = raw.charCodeAt(i);
    }
    return output;
  }

  // ---------------------------------------------------------------------------
  // Visibility Tracking (pause polling on hidden tab)
  // ---------------------------------------------------------------------------

  _initVisibilityTracking() {
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        // Reconnect WebSocket if it was closed (mobile sleep/resume)
        if (!this.ws.isConnected) {
          this.ws.connect();
        }
        // Dismiss stale notifications now that the user is back
        this._dismissNotifications();
        // Scroll to latest message so user sees current state
        this.renderer.forceScrollToBottom();
      }
    });
  }

  _dismissNotifications() {
    if (!navigator.serviceWorker?.controller) return;
    navigator.serviceWorker.ready.then(reg => {
      reg.getNotifications({ tag: 'chalie-message' }).then(notes => {
        notes.forEach(n => n.close());
      });
    }).catch(() => {});
  }

  // ---------------------------------------------------------------------------
  // Connection Monitor
  // ---------------------------------------------------------------------------

  _initConnectionMonitor() {
    // Periodically check health
    this._healthCheck();
  }

  async _healthCheck() {
    try {
      const data = await this.api.healthCheck();
      this._hideConnectionBanner();
      // Version-change detection (post-restart cache bust)
      if (data?.version) {
        if (!window.__chalieVersion) {
          window.__chalieVersion = data.version;
        } else if (data.version !== window.__chalieVersion) {
          window.__chalieVersion = data.version;
          location.reload();
          return;
        }
      }
      // Check again in 30s
      this._healthRetryTimeout = setTimeout(() => this._healthCheck(), 30000);
    } catch {
      this._showConnectionBanner();
      // Retry in 3s
      this._healthRetryTimeout = setTimeout(() => this._healthCheck(), 3000);
    }
  }

  _showConnectionBanner() {
    const banner = document.getElementById('connectionBanner');
    banner.classList.remove('hidden');
  }

  _hideConnectionBanner() {
    const banner = document.getElementById('connectionBanner');
    banner.classList.add('hidden');
  }

  // ---------------------------------------------------------------------------
  // API Key Dialog
  // ---------------------------------------------------------------------------

  _handleAuthFailure() {
    clearInterval(this._taskStripInterval);
    window.location.replace('/login/');
  }

  // ---------------------------------------------------------------------------
  // PWA Install Dialog
  // ---------------------------------------------------------------------------

  _initPwaDialog() {
    const dialog = document.getElementById('pwaInstallDialog');
    const closeBtn = dialog.querySelector('.pwa-dialog__close');
    const installBtn = document.getElementById('pwaInstallBtn');

    const dismiss = () => {
      _lsSet('chalie_pwa_dismissed', '1');
      dialog.close();
      // _init() owns the boot sequence — it awaits _showPwaDialogIfNeeded() and
      // calls _start() when the dialog closes. Do NOT call _start() here; doing so
      // would run _start() twice (once from dismiss, once from _init's await).
    };

    closeBtn.addEventListener('click', dismiss);
    dialog.addEventListener('cancel', dismiss); // Escape key

    installBtn.addEventListener('click', async () => {
      if (this._deferredInstallPrompt) {
        this._deferredInstallPrompt.prompt();
        const { outcome } = await this._deferredInstallPrompt.userChoice;
        this._deferredInstallPrompt = null;
      }
      dismiss();
    });
  }

  _showPwaDialogIfNeeded() {
    // Already installed as PWA
    if (window.matchMedia('(display-mode: standalone)').matches) return;
    // Already dismissed by user
    if (_lsGet('chalie_pwa_dismissed')) return;

    const dialog = document.getElementById('pwaInstallDialog');
    dialog.showModal();

    // Return a Promise that resolves when the dialog closes
    return new Promise(resolve => {
      dialog.addEventListener('close', resolve, { once: true });
    });
  }

  // ---------------------------------------------------------------------------
  // Ambient Canvas
  // ---------------------------------------------------------------------------

  _initAmbientCanvas() {
    const canvas = document.getElementById('ambientCanvas');
    const ctx = canvas.getContext('2d');

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    // Check for reduced motion preference
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (prefersReduced.matches) {
      // Draw a single static frame
      this._drawAmbientFrame(ctx, canvas, 0);
      return;
    }

    // Radiant drifting gradient blobs
    let t = 0;
    const animate = () => {
      this._drawAmbientFrame(ctx, canvas, t);
      t += 0.0012;
      requestAnimationFrame(animate);
    };
    animate();
  }

  _drawAmbientFrame(ctx, canvas, t) {
    const w = canvas.width;
    const h = canvas.height;
    const m = Math.min(w, h);

    // Near-black base. The orbs are the only light source.
    ctx.fillStyle = '#06080e';
    ctx.fillRect(0, 0, w, h);

    // Restrained orbs — think distant nebulae, not lava lamps.
    // Two warm (violet / magenta) and one cool (cyan) for contrast.
    // Low alpha keeps it atmospheric, not decorative.
    const orbs = [
      // Large violet field — dominates top-left — this IS the brand color
      { cx: 0.22, cy: 0.20, r: 0.70, color: [100, 60, 220], alpha: 0.08,
        dx: 0.07, dy: 0.06, sx: 1.0,  sy: 0.75, rBreath: 0.06, phase: 0.0  },
      // Magenta — lower-right — warm human counterpoint
      { cx: 0.78, cy: 0.65, r: 0.55, color: [180, 30, 140], alpha: 0.06,
        dx: 0.06, dy: 0.07, sx: 0.85, sy: 1.1,  rBreath: 0.06, phase: 2.4  },
      // Cyan accent — top-right — the "technology" color, small and precise
      { cx: 0.80, cy: 0.15, r: 0.30, color: [0, 180, 220],  alpha: 0.05,
        dx: 0.08, dy: 0.05, sx: 0.95, sy: 1.05, rBreath: 0.05, phase: 1.5  },
      // Deep indigo anchor — bottom — grounds the composition
      { cx: 0.40, cy: 0.90, r: 0.50, color: [60, 40, 140],  alpha: 0.06,
        dx: 0.05, dy: 0.08, sx: 1.10, sy: 0.90, rBreath: 0.06, phase: 4.2  },
    ];

    for (const orb of orbs) {
      const x = w * (orb.cx + orb.dx * Math.sin(t * orb.sx + orb.phase));
      const y = h * (orb.cy + orb.dy * Math.cos(t * orb.sy + orb.phase * 0.7));
      const r = m * orb.r * (1 + orb.rBreath * Math.sin(t * 0.5 + orb.phase));
      const a = orb.alpha * (0.75 + 0.25 * Math.cos(t * 0.3 + orb.phase));

      const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
      const [cr, cg, cb] = orb.color;
      grad.addColorStop(0,   `rgba(${cr},${cg},${cb},${a})`);
      grad.addColorStop(0.4, `rgba(${cr},${cg},${cb},${(a * 0.35).toFixed(4)})`);
      grad.addColorStop(1,   `rgba(${cr},${cg},${cb},0)`);

      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);
    }
  }

  // ─── Document upload ──────────────────────────────────────────────
  _initUpload() {
    const dialog = document.getElementById('uploadDialog');
    const closeBtn = document.getElementById('uploadDialogClose');
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('uploadFileInput');

    if (!dialog) return;

    closeBtn?.addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); });

    // Dropzone click → file picker
    dropzone?.addEventListener('click', () => fileInput?.click());

    // Drag & drop
    dropzone?.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone?.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone?.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer?.files?.length) this._handleFiles(e.dataTransfer.files);
    });

    // File input change
    fileInput?.addEventListener('change', () => {
      if (fileInput.files?.length) this._handleFiles(fileInput.files);
    });

    // Attach menu (+ button → slide-up with document and image options)
    this._initAttachMenu();
  }

  _initAttachMenu() {
    const attachBtn = document.getElementById('attachBtn');
    const menu = document.getElementById('attachMenu');
    const dialog = document.getElementById('uploadDialog');

    if (!attachBtn || !menu) return;

    attachBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const isOpen = !menu.classList.contains('hidden');
      menu.classList.toggle('hidden', isOpen);
      attachBtn.classList.toggle('active', !isOpen);
    });

    // Close on outside click
    document.addEventListener('click', (e) => {
      if (!menu.contains(e.target) && e.target !== attachBtn) {
        menu.classList.add('hidden');
        attachBtn.classList.remove('active');
      }
    });

    // "Attach Document" → existing upload dialog
    menu.querySelector('[data-action="document"]')?.addEventListener('click', () => {
      menu.classList.add('hidden');
      attachBtn.classList.remove('active');
      this._resetUploadDialog();
      dialog?.showModal();
    });

    // "Take Photo / Pick Image" → image file input
    menu.querySelector('[data-action="image"]')?.addEventListener('click', () => {
      menu.classList.add('hidden');
      attachBtn.classList.remove('active');
      document.getElementById('imageFileInput')?.click();
    });

    // Image file input change
    document.getElementById('imageFileInput')?.addEventListener('change', (e) => {
      if (e.target.files?.length) this._handleImageAttach(e.target.files[0]);
      e.target.value = '';
    });
  }

  async _checkVisionCapability() {
    try {
      const res = await fetch('/chat/vision-capable', { credentials: 'same-origin' });
      if (res.ok) {
        const data = await res.json();
        if (!data?.available) {
          document.getElementById('attachImageBtn')?.classList.add('hidden');
        }
      }
    } catch { /* silently hide on any error */ }
  }

  /**
   * Handle an image file selected by the user.
   *
   * Uploads the file via REST, then keeps the spinner and `analyzing` CSS class
   * on the thumbnail until the server pushes an `image_ready` WebSocket event
   * (Step 4 / B4 fix).  A 90-second safety-net timeout replaces the spinner
   * with a warning badge if the event never arrives.
   *
   * The send button is enabled as soon as the server returns an `image_id` so
   * the user is not blocked — the WebSocket handler (Step 5) will still attempt
   * a short poll for the result when the message is dispatched.
   *
   * @param {File} file - The image file selected by the user.
   */
  async _handleImageAttach(file) {
    if (this._attachedImages.length >= 3) {
      this._showToast?.('Maximum 3 images per message');
      return;
    }
    const thumbEl = this._addImagePreview(file);

    const formData = new FormData();
    formData.append('image', file);
    try {
      const res = await fetch('/chat/image', {
        method: 'POST',
        credentials: 'same-origin',
        body: formData,
      });
      const data = await res.json();
      if (res.ok && data.image_id) {
        this._attachedImages.push({ id: data.image_id, element: thumbEl });
        // Step 4 / B4 fix: keep the spinner and 'analyzing' class — they are
        // cleared when the server sends an 'image_ready' WebSocket event after
        // background analysis completes (see _handleEvent).  Do NOT remove them
        // here; the previous behaviour removed them immediately (before analysis
        // even started), which caused the LLM to silently miss image context.
        document.getElementById('sendBtn').disabled = false;

        // Safety net: if the 'image_ready' event does not arrive within 90 s,
        // replace the spinner with a warning badge so the user knows analysis
        // timed out.  The image remains attached — the WS handler will still do
        // a short fallback poll when the message is sent (Step 5).
        const timeoutId = setTimeout(() => {
          this._pendingImageAnalysis.delete(data.image_id);
          thumbEl.classList.remove('analyzing');
          thumbEl.querySelector('.image-preview__spinner')?.remove();
          const warn = document.createElement('span');
          warn.className = 'image-preview__warn';
          warn.title = 'Image analysis timed out — context may be unavailable';
          warn.textContent = '⚠';
          thumbEl.appendChild(warn);
        }, 90_000);

        this._pendingImageAnalysis.set(data.image_id, { element: thumbEl, timeout: timeoutId });
      } else {
        thumbEl.remove();
        this._updatePreviewVisibility();
        this._showToast?.(data.error || 'Image upload failed');
      }
    } catch {
      thumbEl.remove();
      this._updatePreviewVisibility();
      this._showToast?.('Image upload failed');
    }
  }

  _addImagePreview(file) {
    const strip = document.getElementById('imagePreview');
    strip.classList.remove('hidden');

    const thumb = document.createElement('div');
    thumb.className = 'image-preview__thumb analyzing';

    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.alt = file.name;
    thumb.appendChild(img);

    // Spinner overlay (shown while upload/analysis in-flight)
    const spinner = document.createElement('div');
    spinner.className = 'image-preview__spinner';
    spinner.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4m0 12v4m-7.07-3.93 2.83-2.83m8.48-8.48 2.83-2.83M2 12h4m12 0h4m-3.93 7.07-2.83-2.83M7.76 7.76 4.93 4.93"/></svg>';
    thumb.appendChild(spinner);

    // Remove button
    const removeBtn = document.createElement('button');
    removeBtn.className = 'image-preview__remove';
    removeBtn.setAttribute('aria-label', 'Remove image');
    removeBtn.textContent = '\u00d7';
    removeBtn.addEventListener('click', () => {
      const idx = this._attachedImages.findIndex(a => a.element === thumb);
      if (idx >= 0) this._attachedImages.splice(idx, 1);
      thumb.remove();
      this._updatePreviewVisibility();
      if (!this._attachedImages.length) {
        const textarea = document.getElementById('messageInput');
        document.getElementById('sendBtn').disabled = !textarea?.value.trim();
      }
    });
    thumb.appendChild(removeBtn);

    strip.appendChild(thumb);
    return thumb;
  }

  _updatePreviewVisibility() {
    const strip = document.getElementById('imagePreview');
    if (strip && !strip.children.length) strip.classList.add('hidden');
  }

  /**
   * Remove all image thumbnails and cancel pending analysis timeouts.
   *
   * Called when a message is sent or the user navigates away.  Cancels any
   * in-flight safety-net timers created by _handleImageAttach so they do not
   * fire after the preview strip has been cleared.
   */
  _clearImagePreview() {
    // Cancel all pending 90 s safety-net timers before clearing the DOM.
    for (const { timeout } of this._pendingImageAnalysis.values()) {
      clearTimeout(timeout);
    }
    this._pendingImageAnalysis.clear();

    this._attachedImages = [];
    const strip = document.getElementById('imagePreview');
    if (strip) {
      strip.innerHTML = '';
      strip.classList.add('hidden');
    }
  }

  _resetUploadDialog() {
    const progress = document.getElementById('uploadProgress');
    const dupWarning = document.getElementById('uploadDuplicateWarning');
    const fileInput = document.getElementById('uploadFileInput');
    progress?.classList.add('hidden');
    dupWarning?.classList.add('hidden');
    if (fileInput) fileInput.value = '';
  }

  async _handleFiles(files) {
    const file = files[0];
    if (!file) return;

    const dialog = document.getElementById('uploadDialog');
    const progress = document.getElementById('uploadProgress');
    const progressLabel = document.getElementById('uploadProgressLabel');
    const dupWarning = document.getElementById('uploadDuplicateWarning');

    // Show progress
    progress?.classList.remove('hidden');
    progressLabel.textContent = 'Uploading...';
    dupWarning?.classList.add('hidden');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await this.api.upload('/documents/upload', formData);

      if (!res || res.error) {
        progressLabel.textContent = res?.error || 'Upload failed';
        progressLabel.style.color = '#f87171';
        return;
      }

      // Check for duplicates
      if (res.duplicates?.length) {
        const dup = res.duplicates[0];
        const dateStr = dup.created_at ? new Date(dup.created_at).toLocaleDateString() : '';
        dupWarning.innerHTML = `
          <div>This looks like an updated version of <strong>${this._esc(dup.original_name)}</strong>${dateStr ? ` from ${dateStr}` : ''}. Replace the older version, or keep both?</div>
          <div class="upload-duplicate__actions">
            <button class="upload-duplicate__btn upload-duplicate__btn--primary" data-action="replace" data-new-id="${res.id}" data-old-id="${dup.id}">Replace</button>
            <button class="upload-duplicate__btn" data-action="keep">Keep Both</button>
          </div>`;
        dupWarning.classList.remove('hidden');

        dupWarning.querySelectorAll('button').forEach(btn => {
          btn.addEventListener('click', () => {
            dupWarning.classList.add('hidden');
            if (btn.dataset.action === 'replace') {
              const newId = btn.dataset.newId;
              const oldId = btn.dataset.oldId;
              this.api.post(`/documents/${newId}/supersede`, { old_id: oldId })
                .then(() => this._showToast('Replaced older version'))
                .catch(() => this._showToast('Could not replace — keeping both'));
            }
          });
        });
      }

      // Poll for status
      progressLabel.textContent = 'Extracting text...';
      this._pollDocumentStatus(res.id, progressLabel, dialog);

    } catch (e) {
      progressLabel.textContent = 'Upload failed';
      progressLabel.style.color = '#f87171';
    }
  }

  async _pollDocumentStatus(docId, label, dialog) {
    let attempts = 0;
    const maxAttempts = 60; // 2 minutes max
    let synthWaitAttempts = 0;
    const maxSynthWait = 15; // 30s max to wait for LLM synthesis before showing card anyway

    const poll = async () => {
      if (attempts++ > maxAttempts) {
        label.textContent = 'Processing taking longer than expected...';
        return;
      }

      try {
        const res = await this.api.get(`/documents/${docId}`);
        const status = res?.item?.status;

        if (status === 'ready') {
          label.textContent = 'Ready';
          label.style.color = '#34d399';
          setTimeout(() => dialog?.close(), 1500);
          this._showToast(`Document "${res.item.original_name}" processed`);
          return;
        } else if (status === 'failed') {
          label.textContent = res?.item?.error_message || 'Processing failed';
          label.style.color = '#f87171';
          return;
        } else if (status === 'awaiting_confirmation') {
          const hasSynthesis = res.item.extracted_metadata?._synthesis;
          if (!hasSynthesis && synthWaitAttempts++ < maxSynthWait) {
            // Synthesis LLM call still in progress — wait up to 30s then proceed anyway
            label.textContent = 'Generating summary...';
            setTimeout(poll, 2000);
            return;
          }
          dialog?.close();
          this._showDocumentSynthesis(docId, res.item);
          return;
        } else if (status === 'processing') {
          label.textContent = 'Understanding document...';
        }
      } catch { /* retry */ }

      setTimeout(poll, 2000);
    };

    poll();
  }

  _showDocumentSynthesis(docId, doc) {
    const meta = doc.extracted_metadata || {};
    const synthesis = meta._synthesis || doc.summary || '';
    const keyFacts = meta._key_facts || [];
    const docType = (meta.document_type || {}).value || '';

    const card = document.createElement('div');
    card.className = 'doc-synthesis-card';
    card.dataset.docId = docId;

    // Header
    const header = document.createElement('div');
    header.className = 'doc-synthesis__header';
    const nameSpan = document.createElement('span');
    nameSpan.className = 'doc-synthesis__name';
    nameSpan.textContent = doc.original_name || 'Document';
    header.appendChild(nameSpan);
    if (docType && docType !== 'document') {
      const typeTagSpan = document.createElement('span');
      typeTagSpan.className = 'doc-synthesis__tag';
      typeTagSpan.textContent = docType;
      header.appendChild(typeTagSpan);
    }
    card.appendChild(header);

    // Synthesis text
    const textP = document.createElement('p');
    textP.className = 'doc-synthesis__text';
    textP.textContent = synthesis;
    card.appendChild(textP);

    // Key facts
    if (keyFacts.length) {
      const factsDiv = document.createElement('div');
      factsDiv.className = 'doc-synthesis__facts';
      for (const f of keyFacts) {
        const factSpan = document.createElement('span');
        factSpan.className = 'doc-synthesis__fact';
        factSpan.textContent = f;
        factsDiv.appendChild(factSpan);
      }
      card.appendChild(factsDiv);
    }

    // Actions
    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'doc-synthesis__actions';

    const confirmBtn = document.createElement('button');
    confirmBtn.className = 'doc-synthesis__btn doc-synthesis__btn--confirm';
    confirmBtn.textContent = 'Looks good';
    actionsDiv.appendChild(confirmBtn);

    const augmentBtn = document.createElement('button');
    augmentBtn.className = 'doc-synthesis__btn doc-synthesis__btn--augment';
    augmentBtn.textContent = 'Add context';
    actionsDiv.appendChild(augmentBtn);

    const discardBtn = document.createElement('button');
    discardBtn.className = 'doc-synthesis__btn doc-synthesis__btn--discard';
    discardBtn.textContent = 'Discard';
    actionsDiv.appendChild(discardBtn);

    card.appendChild(actionsDiv);

    // Augment area
    const augmentArea = document.createElement('div');
    augmentArea.className = 'doc-synthesis__augment-area hidden';
    const textarea = document.createElement('textarea');
    textarea.className = 'doc-synthesis__textarea';
    textarea.placeholder = 'Add context about this document...';
    augmentArea.appendChild(textarea);
    const submitBtn = document.createElement('button');
    submitBtn.className = 'doc-synthesis__btn doc-synthesis__btn--submit';
    submitBtn.textContent = 'Save';
    augmentArea.appendChild(submitBtn);
    card.appendChild(augmentArea);

    confirmBtn.addEventListener('click', async () => {
      confirmBtn.disabled = true;
      confirmBtn.textContent = 'Confirming...';
      try {
        await this.api.post(`/documents/${docId}/confirm`, {});
        card.remove();
        this._showToast('Document ready');
      } catch (e) {
        confirmBtn.disabled = false;
        confirmBtn.textContent = 'Looks good';
        this._showToast('Confirmation failed');
      }
    });

    augmentBtn.addEventListener('click', () => {
      augmentArea.classList.toggle('hidden');
      textarea.focus();
    });

    submitBtn.addEventListener('click', async () => {
      const ctx = textarea.value.trim();
      if (!ctx) return;
      submitBtn.disabled = true;
      submitBtn.textContent = 'Saving...';
      try {
        await this.api.post(`/documents/${docId}/augment`, { context: ctx });
        card.remove();
        this._showToast('Document ready with your context');
      } catch (e) {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Save';
        this._showToast('Failed to save context');
      }
    });

    discardBtn.addEventListener('click', async () => {
      discardBtn.disabled = true;
      discardBtn.textContent = 'Discarding...';
      try {
        await this.api.del(`/documents/${docId}/purge`);
        card.remove();
        this._showToast('Document discarded');
      } catch (e) {
        discardBtn.disabled = false;
        discardBtn.textContent = 'Discard';
        this._showToast('Discard failed');
      }
    });

    // Inject into chat spine
    const spine = document.getElementById('conversationSpine');
    if (spine) spine.appendChild(card);
    card.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }

  _esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

  // ---------------------------------------------------------------------------
  // Update System
  // ---------------------------------------------------------------------------

  _initUpdateSystem() {
    document.getElementById('updateBannerBtn')?.addEventListener('click', () => this._showUpdateDialog());
    document.getElementById('updateBannerDismiss')?.addEventListener('click', () => this._dismissUpdateBanner());
    document.getElementById('updateDialogClose')?.addEventListener('click', () => this._closeUpdateDialog());
    document.getElementById('updateCancelBtn')?.addEventListener('click', () => this._closeUpdateDialog());
    document.getElementById('updateApplyBtn')?.addEventListener('click', () => this._applyUpdate());
  }

  _handleUpdateEvent(data) {
    this._pendingUpdate = data;
    const dismissedVersion = _lsGet('chalie_update_dismissed');
    if (dismissedVersion === data.latest_tag) return;
    this._showUpdateBanner(data);
  }

  _showUpdateBanner(data) {
    const banner = document.getElementById('updateBanner');
    const versionEl = document.getElementById('updateBannerVersion');
    if (!banner || !versionEl) return;

    versionEl.textContent = `v${data.latest_version}`;

    if (data.deployment_mode === 'docker' || data.deployment_mode === 'dev') {
      const btn = document.getElementById('updateBannerBtn');
      if (btn) btn.textContent = 'Details';
    }

    banner.classList.remove('hidden');
  }

  _dismissUpdateBanner() {
    const banner = document.getElementById('updateBanner');
    if (banner) banner.classList.add('hidden');
    if (this._pendingUpdate) {
      _lsSet('chalie_update_dismissed', this._pendingUpdate.latest_tag);
    }
  }

  _showUpdateDialog() {
    const dialog = document.getElementById('updateDialog');
    if (!dialog || !this._pendingUpdate) return;

    const data = this._pendingUpdate;
    const currentEl = document.getElementById('updateCurrentVer');
    const newEl = document.getElementById('updateNewVer');
    const notesEl = document.getElementById('updateNotes');
    const actionsEl = document.getElementById('updateActions');
    const progressEl = document.getElementById('updateProgress');
    const instructionsEl = document.getElementById('updateInstructions');

    if (currentEl) currentEl.textContent = `v${data.current_version}`;
    if (newEl) newEl.textContent = `v${data.latest_version}`;
    if (notesEl) notesEl.textContent = data.release_notes || 'No release notes.';

    if (actionsEl) actionsEl.classList.remove('hidden');
    if (progressEl) progressEl.classList.add('hidden');
    if (instructionsEl) instructionsEl.classList.add('hidden');

    if (data.deployment_mode === 'docker') {
      if (actionsEl) actionsEl.classList.add('hidden');
      if (instructionsEl) {
        instructionsEl.innerHTML = `<p>You're running Chalie in Docker. To update:</p><code>docker pull chalie/chalie:${this._esc(data.latest_tag)}\ndocker compose up -d</code>`;
        instructionsEl.classList.remove('hidden');
      }
    } else if (data.deployment_mode === 'dev') {
      if (actionsEl) actionsEl.classList.add('hidden');
      if (instructionsEl) {
        instructionsEl.innerHTML = `<p>You're running from a git clone. To update:</p><code>git pull origin main</code>`;
        instructionsEl.classList.remove('hidden');
      }
    }

    dialog.showModal();
  }

  _closeUpdateDialog() {
    document.getElementById('updateDialog')?.close();
  }

  async _applyUpdate() {
    if (!this._pendingUpdate) return;

    const actionsEl = document.getElementById('updateActions');
    const progressEl = document.getElementById('updateProgress');
    if (actionsEl) actionsEl.classList.add('hidden');
    if (progressEl) progressEl.classList.remove('hidden');

    try {
      const resp = await fetch('/system/update/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ tag: this._pendingUpdate.latest_tag }),
      });
      const result = await resp.json();

      if (!result.ok) {
        const statusEl = progressEl?.querySelector('.update-dialog__status');
        if (statusEl) statusEl.textContent = result.message || 'Update failed.';
        setTimeout(() => {
          if (actionsEl) actionsEl.classList.remove('hidden');
          if (progressEl) progressEl.classList.add('hidden');
        }, 3000);
        return;
      }

      const statusEl = progressEl?.querySelector('.update-dialog__status');
      if (statusEl) statusEl.textContent = 'Restarting Chalie...';
    } catch {
      const statusEl = progressEl?.querySelector('.update-dialog__status');
      if (statusEl) statusEl.textContent = 'Update request failed.';
    }
  }
}

// Boot
new ChalieApp();

if (typeof lucide !== 'undefined') lucide.createIcons();

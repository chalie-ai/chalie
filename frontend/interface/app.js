/**
 * Chalie — Application bootstrap & orchestrator.
 *
 * Thin wiring layer: creates modules, injects dependencies via constructors
 * and callbacks, delegates all domain logic to focused ES6 modules.
 */
import { ApiClient } from './api.js';
import { WSClient } from './ws.js';
import { Presence } from './presence.js';
import { Renderer } from './renderer.js';
import { VoiceRecorder } from './voice_recorder.js';
import { VoicePlayer } from './voice_player.js';
import { ClientHeartbeat } from './heartbeat.js';
import { AmbientSensor } from './ambient.js';
import { AmbientCanvas } from './ambient_canvas.js';
import { MomentSearch } from './moment_search.js';
import { Chat } from './chat.js';
import { TaskStrip } from './task_strip.js';
import { EventRouter } from './event_router.js';
import { Notifications } from './notifications.js';
import { ImageAttach } from './image_attach.js';
import { DocumentUpload } from './document_upload.js';
import { UpdateSystem } from './update_system.js';
import { PermissionNotifications } from './permission_notifications.js';
import { showToast, lsGet, lsSet } from './utils.js';

// Disable the browser's scroll-restoration so a refresh never lands the
// user mid-conversation. Renderer.forceScrollToBottom takes over after
// history loads. Set at module load — earlier than DOMContentLoaded — so
// the browser never applies its stored scroll position in the first place.
if ('scrollRestoration' in history) {
  history.scrollRestoration = 'manual';
}

class ChalieApp {
  constructor() {
    this._backendHost = lsGet('chalie_backend_host') || '';
    this._deferredInstallPrompt = null;
    this._activeCapabilityAlerts = new Map();

    // Core modules
    this.api = new ApiClient(() => this._backendHost);
    this.ws = new WSClient(() => this._backendHost);
    this.heartbeat = new ClientHeartbeat(() => this._backendHost);
    this.presence = null;
    this.renderer = null;

    this._init();
  }

  async _init() {
    // Wait for DOM
    if (document.readyState === 'loading') {
      await new Promise(r => document.addEventListener('DOMContentLoaded', r));
    }

    // Auth + provider gate — shared across all pages.
    const gate = await globalThis.chalieGateReady;
    if (!gate.stay) return;

    this._registerServiceWorker();
    this._initInstallPrompt();
    this._initPresence();
    this._initRenderer();
    this._checkVaultReinit();

    // Notifications module
    this._notifications = new Notifications({ getHost: () => this._backendHost });

    // Image attach module
    this._imageAttach = new ImageAttach({
      getHost: () => this._backendHost,
      onDocumentDrop: (file) => this._docUpload?.uploadFile(file),
    });
    this._imageAttach.init();

    // Voice recorder (mic → STT → paste into input)
    this._voiceRecorder = new VoiceRecorder({
      getHost: () => this._backendHost,
      onTranscript: (text) => this._pasteVoiceTranscript(text),
    });

    // Voice player (speaker button → overlay audio player)
    this._voicePlayer = new VoicePlayer({ getHost: () => this._backendHost });

    // Chat module (send + history)
    this._chat = new Chat({
      api: this.api,
      ws: this.ws,
      renderer: this.renderer,
      presence: this.presence,
      notifications: this._notifications,
      imageAttach: this._imageAttach,
    });
    this._chat.onAuthFailure(() => this._handleAuthFailure());

    // Task strip module
    this._taskStrip = new TaskStrip({ api: this.api });
    this._taskStrip.onAuthFailure(() => this._handleAuthFailure());

    // Heartbeat — redirect to login if the session becomes invalid
    // (server restart that locks the vault, session expiry, etc.)
    this.heartbeat.onAuthFailure(() => this._handleAuthFailure());

    // Document upload module
    this._docUpload = new DocumentUpload({ api: this.api, getHost: () => this._backendHost });
    this._docUpload.init();

    // Update system module
    this._updateSystem = new UpdateSystem({ getHost: () => this._backendHost });
    this._updateSystem.init();

    // Permission notifications module
    const permNotifications = new PermissionNotifications();
    permNotifications.init();
    this._permNotifications = permNotifications;

    // Event router — dispatches WS drift events to modules
    this._eventRouter = new EventRouter({ ws: this.ws, renderer: this.renderer });
    this._wireEventRouter();

    // Ambient sensor — passive behavioral observer
    this._ambientSensor = new AmbientSensor();
    this._ambientSensor.bindTypingInput(document.getElementById('messageInput'));
    this.heartbeat.setAmbientSensor(this._ambientSensor);

    // Record response timestamps in ambient sensor
    this._chat.onResponseReceived(() => this._ambientSensor.recordResponse());

    // Moments (recall search)
    this._initMoments();

    // Developer helper: append ?debug_thought=1 to the URL to inject a mock
    // thought card after 2 seconds, enabling visual testing without a backend.
    // The event is fed directly into the event router's internal handler so it
    // exercises the full thought → renderer pipeline.
    if (new URLSearchParams(window.location.search).has('debug_thought')) {
      setTimeout(() => {
        this._eventRouter._handleEvent({
          type: 'thought',
          topic: 'curiosity',
          mode: 'proactive',
          confidence: 0.82,
          content: '<p>You might enjoy reading about octopus cognition — it connects to things we discussed about distributed intelligence.</p>',
        });
      }, 2000);
    }

    // Input (textarea, send button)
    this._initInput();

    // Attach menu (+ button)
    this._initAttachMenu();

    // PWA install dialog
    this._initPwaDialog();

    // Ambient canvas (background animation)
    const canvas = new AmbientCanvas();
    canvas.init();

    // Visibility tracking
    this._initVisibilityTracking();

    // Connection monitor
    this._initConnectionMonitor();

    // Capability alert banner dismiss wiring
    this._initCapabilityAlertBanner();

    // Share target (PWA shared content)
    this._handleSharedContent();

    // Start heartbeat
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

  // ---------------------------------------------------------------------------
  // Service Worker
  // ---------------------------------------------------------------------------

  _registerServiceWorker() {
    if (!('serviceWorker' in navigator)) return;
    navigator.serviceWorker.register('/sw.js').catch(err =>
      console.warn('SW registration failed:', err)
    );
  }

  // ---------------------------------------------------------------------------
  // Vault Reinit Warning
  // ---------------------------------------------------------------------------

  async _checkVaultReinit() {
    try {
      const r = await fetch('/auth/vault-status', { credentials: 'same-origin' });
      if (!r.ok) return;
      const { reinitialized_at } = await r.json();
      if (!reinitialized_at) return;
      const banner = document.getElementById('vaultReinitBanner');
      if (!banner) return;
      banner.classList.remove('hidden');
      document.getElementById('vaultReinitDismiss')?.addEventListener('click', async () => {
        banner.classList.add('hidden');
        await fetch('/auth/vault-status/dismiss', { method: 'POST', credentials: 'same-origin' });
      }, { once: true });
    } catch (err) {
      console.debug('[vault-reinit] check failed (non-fatal):', err);
    }
  }

  // ---------------------------------------------------------------------------
  // Install Prompt
  // ---------------------------------------------------------------------------

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

  // ---------------------------------------------------------------------------
  // Backend Ready Poll
  // ---------------------------------------------------------------------------

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

  // ---------------------------------------------------------------------------
  // Start
  // ---------------------------------------------------------------------------

  /**
   * Start the application after the backend is ready.
   *
   * Sequence:
   *  1. Poll until the backend signals ready (or timeout / skip).
   *  2. Dismiss the loading overlay.
   *  3. Initialise voice I/O.
   *  4. Show the first-visit welcome message exactly once (localStorage guard).
   *  5. Load conversation history and bind the chat module.
   *  6. Boot remaining modules (task strip, apps panel, WebSocket event router).
   *
   * @returns {Promise<void>}
   */
  async _start() {
    try {
      await this._pollUntilReady();
      this._dismissLoadingOverlay();
      this.presence.setState('resting');

      // Voice recorder + player — init DOM bindings now that DOM is stable
      this._voiceRecorder.init();
      this._voicePlayer.init();

      // Check voice availability and hide controls if the service is unavailable.
      this._initVoiceAvailability();

      // First-visit welcome message — shown exactly once via a localStorage flag.
      // Rendered before history so it appears at the top of the conversation spine.
      if (!lsGet('chalie_welcomed')) {
        lsSet('chalie_welcomed', '1');
        this.renderer.appendChalieForm(
          "<p>Hello. I'm Chalie.</p>",
          { mode: 'UNIFIED', confidence: 1 },
        );
      }

      // Load conversation history
      await this._chat.loadRecentConversation();
      this._chat.init(); // bind scroll-up pagination after initial load

      // Task strip — init + 60s safety-net polling
      this._taskStrip.init();

      // Settings button → brain dashboard
      document.getElementById('settingsBtn')?.addEventListener('click', () => {
        window.open('/brain/', '_blank');
      });

      // Connect WebSocket and drift event router
      this._eventRouter.init();

      window.addEventListener('beforeunload', () => {
        this.ws.close();
        this._taskStrip.destroy();
        this._voiceRecorder.destroy();
      }, { once: true });

      this._notifications.requestPushSubscription();

      // Ask once for geolocation permission so the heartbeat can capture coordinates.
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
  // Voice Availability
  // ---------------------------------------------------------------------------

  _initVoiceAvailability() {
    const POLL_INTERVAL_MS = 2000;
    const MAX_POLL_MS = 60_000;
    const deadline = Date.now() + MAX_POLL_MS;

    // Page loads with body.voice-loading set (see index.html). That class
    // hides #voiceRecBtn and every .speech-form__speak-btn via style.css so
    // mic + speaker icons never appear before voice is actually usable.
    // We only clear it after /voice/health returns {status: "ok"}.
    const reveal = () => {
      document.body.classList.remove('voice-loading');
    };

    const hide = () => {
      // Keep voice-loading on so the icons stay hidden, and tag the body
      // as unavailable for any additional CSS selectors.
      document.getElementById('voiceRecBtn')?.classList.add('hidden');
      document.body.classList.add('voice-unavailable');
    };

    const check = async () => {
      try {
        // Same-origin relative path — no host concatenation, no taint concerns.
        const resp = await fetch('/voice/health', { credentials: 'same-origin' });
        const data = resp.ok ? await resp.json().catch(() => ({})) : {};
        const status = data.status;
        if (status === 'ok') { reveal(); return; }
        if (status === 'unavailable') { hide(); return; }
        // 'loading' — re-poll until deadline
        if (Date.now() < deadline) {
          setTimeout(check, POLL_INTERVAL_MS);
        } else {
          hide(); // timeout — treat as unavailable
        }
      } catch (err) {
        console.warn('[voice] health check unreachable:', err);
        hide();
      }
    };

    check();
  }

  // ---------------------------------------------------------------------------
  // Voice Transcript Paste
  // ---------------------------------------------------------------------------

  _pasteVoiceTranscript(text) {
    const textarea = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    if (!textarea) return;
    textarea.value = text;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    if (sendBtn) sendBtn.disabled = !text.trim();
    textarea.focus();
    // Move cursor to end
    textarea.selectionStart = textarea.selectionEnd = text.length;
  }

  // ---------------------------------------------------------------------------
  // Event Router Wiring
  // ---------------------------------------------------------------------------

  _wireEventRouter() {
    this._eventRouter.setIsSendingGetter(() => this._chat.isSending);

    this._eventRouter.onUpdateEvent((data) => {
      this._updateSystem.handleUpdateEvent(data);
    });

    this._eventRouter.onTaskEvent(() => {
      this._taskStrip.loadActiveTasks();
    });

    this._eventRouter.onImageReady((data) => {
      this._imageAttach.handleImageReady(data);
    });

    this._eventRouter.onNotification(() => {
      this._notifications.playChime();
      this._taskStrip.loadActiveTasks();
    });

    this._eventRouter.onBackgroundContent((text) => {
      this._notifications.notifyBackground(text);
    });

    this._eventRouter.onCapabilityAlert((data) => {
      if (data.recovered) {
        this._removeCapabilityAlert(data.cap_id);
      } else {
        this._addCapabilityAlert(data.cap_id, data.cap_name, data.error);
      }
    });

    this._eventRouter.onPermissionRequest((data) => {
      this._permNotifications.handleRequest(data);
    });
  }

  // ---------------------------------------------------------------------------
  // Moments (Recall)
  // ---------------------------------------------------------------------------

  _initMoments() {
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
      if (!payload || this._chat.isSending) return;

      this._chat.isSending = true;
      const actEl = this.renderer.createActCycle();

      this.ws.sendAction(payload, {
        onMessage: (data) => {
          const content = data.content || '';
          const meta = {
            mode: data.mode || 'ACT',
            confidence: data.confidence || 0.95,
          };
          this.renderer.replaceActWithResponse(actEl, content, meta);
        },
        onError: (data) => {
          this.renderer.replaceActWithError(actEl, data.message);
        },
        onDone: () => {
          this._chat.isSending = false;
          this.presence.setState('resting');
        },
      });
    });

    // Silent action (rich-card interactions like list checkboxes — no chat bubble).
    // Caller is responsible for any optimistic UI; onError lets the card revert.
    document.addEventListener('chalie:silent-action', (e) => {
      const { payload, onMessage, onError, onDone } = e.detail || {};
      if (!payload) return;
      this.ws.sendAction(payload, {
        onMessage: onMessage || (() => {}),
        onError: onError || (() => {}),
        onDone: onDone || (() => {}),
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
          showToast(msg, momentId ? () => this._undoMoment(momentId) : null);
        }
      } catch (err) {
        console.warn('Pin moment failed:', err);
      }
    });

    // First-use hint (one-time)
    if (!lsGet('moments_hint_shown')) {
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

  _showMomentsHintOnFirstResponse() {
    // Wait for first Chalie response to show hint
    const observer = new MutationObserver((mutations) => {
      for (const m of mutations) {
        for (const node of m.addedNodes) {
          if (node.classList?.contains('speech-form--chalie')) {
            const btn = node.querySelector('.speech-form__remember-btn');
            if (btn && !lsGet('moments_hint_shown')) {
              lsSet('moments_hint_shown', '1');
              observer.disconnect();

              // Show tooltip near the button
              showToast('Remember important answers to find them later.', null, 5000);
            }
            return;
          }
        }
      }
    });
    observer.observe(document.getElementById('conversationSpine'), { childList: true });
  }

  // ---------------------------------------------------------------------------
  // Input Handling
  // ---------------------------------------------------------------------------

  _initInput() {
    const textarea = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');

    // Unlock audio on first gesture
    const unlockAudio = () => {
      this._notifications.unlockAudio();
      document.removeEventListener('click', unlockAudio);
    };
    document.addEventListener('click', unlockAudio);

    // Scroll to bottom when user focuses the input (clicking or tab-switching)
    textarea.addEventListener('focus', () => {
      this.renderer.forceScrollToBottom();
    });

    // Auto-resize textarea
    textarea.addEventListener('input', () => {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
      sendBtn.disabled = !textarea.value.trim() && !this._imageAttach.count;
    });

    // Enter to send (Shift+Enter for newline)
    textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this._chat.sendMessage();
      }
    });

    // Send button click
    sendBtn.addEventListener('click', () => this._chat.sendMessage());

    // Re-enable input after send completes
    this._chat.onSendComplete(() => {
      document.getElementById('messageInput').focus();
    });

    // Drop targets — input-dock and textarea paste
    this._imageAttach.enableDropTargets({
      dropzone: document.querySelector('.input-dock'),
      pasteTarget: textarea,
    });

    // Global drop overlay — shown when any file is dragged over the viewport
    this._initGlobalDropOverlay();
  }

  _initGlobalDropOverlay() {
    if (document.querySelector('.global-drop-overlay')) return; // idempotent
    const overlay = document.createElement('div');
    overlay.className = 'global-drop-overlay';
    overlay.innerHTML = '<span class="global-drop-overlay__label">Drop image or document here</span>';
    document.body.appendChild(overlay);

    let enterCount = 0;
    const hideOverlay = () => {
      enterCount = 0;
      overlay.classList.remove('active');
    };

    document.addEventListener('dragenter', (ev) => {
      if (!ev.dataTransfer?.types?.includes('Files')) return;
      ev.preventDefault();
      enterCount++;
      overlay.classList.add('active');
    });

    document.addEventListener('dragover', (ev) => {
      if (!ev.dataTransfer?.types?.includes('Files')) return;
      ev.preventDefault();
    });

    document.addEventListener('dragleave', (ev) => {
      // Only decrement on real exits (relatedTarget is null when leaving the viewport)
      if (ev.relatedTarget) return;
      enterCount--;
      if (enterCount <= 0) hideOverlay();
    });

    // Safety net — dragend always fires even if the drop is cancelled,
    // preventing a stuck overlay on drag-out-of-window.
    document.addEventListener('dragend', hideOverlay);
    window.addEventListener('blur', hideOverlay);

    overlay.addEventListener('drop', (ev) => {
      ev.preventDefault();
      hideOverlay();
      const files = ev.dataTransfer?.files;
      if (!files?.length) return;
      for (const file of files) {
        if (file.type.startsWith('image/')) {
          this._imageAttach.handleFile(file);
        } else {
          this._docUpload.uploadFile(file);
        }
      }
    });

    // Hide overlay when drop is handled elsewhere (e.g. input-dock)
    document.addEventListener('drop', hideOverlay);
  }

  // ---------------------------------------------------------------------------
  // Attach Menu (+ button)
  // ---------------------------------------------------------------------------

  _initAttachMenu() {
    const attachBtn = document.getElementById('attachBtn');
    const menu = document.getElementById('attachMenu');

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

    // "Attach Document" → upload dialog
    menu.querySelector('[data-action="document"]')?.addEventListener('click', () => {
      menu.classList.add('hidden');
      attachBtn.classList.remove('active');
      this._docUpload.openDialog();
    });

    // "Take Photo / Pick Image" → image file input
    menu.querySelector('[data-action="image"]')?.addEventListener('click', () => {
      menu.classList.add('hidden');
      attachBtn.classList.remove('active');
      document.getElementById('imageFileInput')?.click();
    });
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
  // Loading Overlay
  // ---------------------------------------------------------------------------

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
  // Visibility Tracking
  // ---------------------------------------------------------------------------

  _initVisibilityTracking() {
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        // Reconnect WebSocket if it was closed (mobile sleep/resume)
        if (!this.ws.isConnected) {
          this.ws.connect();
        }
        // Dismiss stale notifications now that the user is back
        this._notifications.dismissNotifications();
        // Scroll to latest message so user sees current state
        this.renderer.forceScrollToBottom();
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Connection Monitor
  // ---------------------------------------------------------------------------

  _initConnectionMonitor() {
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
  // Capability Alert Banner
  // ---------------------------------------------------------------------------

  _initCapabilityAlertBanner() {
    document.getElementById('capabilityAlertDismiss')?.addEventListener('click', () => {
      this._activeCapabilityAlerts.clear();
      this._syncCapabilityAlertBanner();
    });
  }

  _addCapabilityAlert(capId, capName, error) {
    if (!capId) return;
    this._activeCapabilityAlerts.set(capId, {
      capName: capName || capId,
      error: error || 'unknown error',
    });
    this._syncCapabilityAlertBanner();
  }

  _removeCapabilityAlert(capId) {
    if (!capId) return;
    this._activeCapabilityAlerts.delete(capId);
    this._syncCapabilityAlertBanner();
  }

  _syncCapabilityAlertBanner() {
    const banner = document.getElementById('capabilityAlertBanner');
    const text = document.getElementById('capabilityAlertText');
    if (!banner || !text) return;

    const count = this._activeCapabilityAlerts.size;
    if (count === 0) {
      banner.classList.add('hidden');
      return;
    }

    if (count === 1) {
      const [, info] = [...this._activeCapabilityAlerts.entries()][0];
      text.textContent = `Interface ${info.capName} has degraded — ${info.error}`;
    } else {
      text.textContent = `${count} interfaces degraded`;
    }
    banner.classList.remove('hidden');
  }

  // ---------------------------------------------------------------------------
  // Auth Failure
  // ---------------------------------------------------------------------------

  _handleAuthFailure() {
    this._taskStrip?.destroy();
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
      lsSet('chalie_pwa_dismissed', '1');
      dialog.close();
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
    if (lsGet('chalie_pwa_dismissed')) return;

    const dialog = document.getElementById('pwaInstallDialog');
    dialog.showModal();

    // Return a Promise that resolves when the dialog closes
    return new Promise(resolve => {
      dialog.addEventListener('close', resolve, { once: true });
    });
  }
}

// Boot
new ChalieApp();

if (typeof lucide !== 'undefined') lucide.createIcons();

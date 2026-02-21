/**
 * Chalie — Application bootstrap & orchestrator.
 */
import { ApiClient } from './api.js';
import { SSEClient } from './sse.js';
import { Presence } from './presence.js';
import { Renderer } from './renderer.js';
import { VoiceIO } from './voice.js';
import { ClientHeartbeat } from './heartbeat.js';
import { MemoryCard } from './cards/memory.js';
import { TimelineCard } from './cards/timeline.js';
import { ToolsCard } from './cards/tools.js';
import { ToolResultCard } from './cards/tool_result.js';

class ChalieApp {
  constructor() {
    this._backendHost = localStorage.getItem('chalie_backend_host') || '';
    this._isSending = false;
    this._healthRetryTimeout = null;
    this._driftSource = null;
    this._deferredInstallPrompt = null;

    // Modules
    this.api = new ApiClient(() => this._backendHost);
    this.sse = new SSEClient(() => this._backendHost);
    this.heartbeat = new ClientHeartbeat(() => this._backendHost);
    this.presence = null;
    this.renderer = null;
    this.voice = null;

    // Cards
    this._memoryCard = null;
    this._timelineCard = null;
    this._toolsCard = null;

    this._init();
  }

  async _init() {
    // Wait for DOM
    if (document.readyState === 'loading') {
      await new Promise(r => document.addEventListener('DOMContentLoaded', r));
    }

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
          await this._showLoginDialog();
          return;
        }
      } else {
        window.location.replace('/on-boarding/');
        return;
      }
    } catch (_) { /* backend unreachable — let the app handle it normally */ }

    this._registerServiceWorker();
    this._initInstallPrompt();
    this._initPresence();
    this._initRenderer();
    this._initVoice();
    this._initCards();
    this._initTools();
    this._initInput();
    this._initPwaDialog();
    this._initAmbientCanvas();
    this._initVisibilityTracking();
    this._initConnectionMonitor();
    this.heartbeat.start();

    // Show PWA install prompt first, then resume normal flow after dismiss/install
    await this._showPwaDialogIfNeeded();

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
      if (outcome === 'accepted') document.getElementById('installBtn').classList.add('hidden');
    });
  }

  async _start() {
    try {
      await this.api.healthCheck();
      this.presence.setState('resting');
      await this._loadVoiceConfig();
      this._loadRecentConversation();
      this._connectDriftStream();
      this._requestNotificationPermission();
      // Ask once for geolocation permission so the heartbeat can capture coordinates.
      // The browser shows its own permission dialog; we don't block on the answer.
      this.heartbeat.requestLocationPermission();
    } catch {
      this.presence.setState('error');
      this._showConnectionBanner();
    }
  }

  async _loadVoiceConfig() {
    try {
      const cfg = await this.api._get('/system/voice-config');
      this.voice.configure(cfg.tts_endpoint, cfg.stt_endpoint);
      if (cfg.stt_endpoint) {
        document.getElementById('micBtn')?.classList.remove('hidden');
      }
      this.renderer.setTtsEnabled(!!cfg.tts_endpoint);
    } catch (_) {
      // silently ignore if voice config unavailable
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
    this.voice = new VoiceIO({
      micBtn: document.getElementById('micBtn'),
      recordingOverlay: document.getElementById('recordingOverlay'),
      stopRecordingBtn: document.getElementById('stopRecordingBtn'),
      audioPlayer: document.getElementById('audioPlayer'),
      audioPlayPause: document.getElementById('audioPlayPause'),
      audioTime: document.getElementById('audioTime'),
      audioClose: document.getElementById('audioClose'),
      micError: document.getElementById('micError'),
    });
  }

  // ---------------------------------------------------------------------------
  // Cards
  // ---------------------------------------------------------------------------

  _initCards() {
    this._memoryCard = new MemoryCard(this.api, this.renderer);
    this._timelineCard = new TimelineCard(this.api, this.renderer);
    this._toolResultCard = new ToolResultCard();
  }

  // ---------------------------------------------------------------------------
  // Tools
  // ---------------------------------------------------------------------------

  _initTools() {
    this._toolsCard = new ToolsCard(this.api, this.renderer);
    document.getElementById('settingsBtn')?.addEventListener('click', () => {
      window.open('/brain/', '_blank');
    });
  }

  // ---------------------------------------------------------------------------
  // Input Handling
  // ---------------------------------------------------------------------------

  _initInput() {
    const textarea = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const micBtn = document.getElementById('micBtn');

    // Auto-resize textarea
    textarea.addEventListener('input', () => {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
      sendBtn.disabled = !textarea.value.trim();
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

    // Shared stop-and-transcribe logic
    const stopAndTranscribe = async () => {
      // Show transcribing state
      textarea.value = '';
      textarea.placeholder = 'Transcribing...';
      textarea.disabled = true;
      textarea.classList.add('transcribing');
      sendBtn.disabled = true;

      const text = await this.voice.stopRecording();

      // Restore textarea
      textarea.disabled = false;
      textarea.classList.remove('transcribing');
      textarea.placeholder = 'Message Chalie...';

      if (text) {
        textarea.value = text;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
        sendBtn.disabled = false;
        textarea.focus();
      } else {
        textarea.placeholder = 'Could not transcribe — try again';
        setTimeout(() => { textarea.placeholder = 'Message Chalie...'; }, 2500);
      }
    };

    // Mic button
    micBtn.addEventListener('click', async () => {
      if (this.voice._isRecording) {
        await stopAndTranscribe();
      } else {
        await this.voice.startRecording();
      }
    });

    // Stop recording button in overlay
    document.getElementById('stopRecordingBtn').addEventListener('click', async () => {
      if (this.voice._isRecording) {
        await stopAndTranscribe();
      }
    });
  }

  async _sendMessage(source = 'text') {
    const textarea = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const text = textarea.value.trim();

    if (!text) return;

    // If a message is in-flight, resolve it before starting a new one
    if (this._isSending && this._pendingForm) {
      this.renderer.resolvePendingForm(this._pendingForm, '', {});
    }

    this._isSending = true;
    this.presence.setState('processing');
    textarea.value = '';
    textarea.style.height = 'auto';

    // Capture timestamp for this exchange
    const exchangeTimestamp = new Date();

    // Detect tools-related query before sending
    const isToolsQuery = /connect|integrat|tool|what.{0,10}(linked|set.?up|active|running)/i.test(text);

    // Render user form with timestamp
    this.renderer.appendUserForm(text, exchangeTimestamp);

    // Create pending form and store reference for potential early resolution
    const pendingForm = this.renderer.createPendingForm();
    this._pendingForm = pendingForm;

    let responseText = '';
    let responseMeta = {};

    await this.sse.send(text, source, {
      onStatus: (stage) => {
        this.presence.setState(stage);
      },
      onMessage: (data) => {
        responseText = data.text;
        responseMeta = { topic: data.topic };
        if (data.removed_by) {
          responseMeta.removed_by = data.removed_by;
          // Register immediately so the drift stream can find this element
          // before onDone fires — prevents a race where a tool follow-up
          // arrives via output:events before resolvePendingForm has run.
          this.renderer.registerPendingRemoval(data.removed_by, pendingForm);
        }
        if (data.removes) responseMeta.removes = data.removes;
        this.presence.setState('responding');
      },
      onError: (data) => {
        this.renderer.resolvePendingFormError(pendingForm, data.message);
        if (!data.recoverable) {
          this._handleAuthFailure();
        }
      },
      onDone: (data) => {
        if (responseText) {
          responseMeta.duration_ms = data.duration_ms;
          responseMeta.ts = exchangeTimestamp;
          this.renderer.resolvePendingForm(pendingForm, responseText, responseMeta);
        }
        this.presence.setState('resting');
        this._isSending = false;
        this._pendingForm = null;
        // Re-enable input
        document.getElementById('messageInput').focus();
        // Show ToolsCard if the message was about connected integrations
        if (isToolsQuery) {
          this._toolsCard.fetch();
        }
      },
    });

    // Fallback in case onDone never fires (connection drop)
    this._isSending = false;
  }

  // ---------------------------------------------------------------------------
  // Load Recent Conversation
  // ---------------------------------------------------------------------------

  async _loadRecentConversation() {
    try {
      const data = await this.api.getRecentConversation();
      if (!data.exchanges || data.exchanges.length === 0) return;

      for (const exchange of data.exchanges) {
        if (exchange.prompt) {
          this.renderer.appendUserForm(exchange.prompt, exchange.timestamp);
        }
        if (exchange.response) {
          this.renderer.appendChalieForm(exchange.response, { topic: exchange.topic, ts: exchange.timestamp });
        }
      }
    } catch (err) {
      if (err.message === 'AUTH') {
        this._handleAuthFailure();
      }
      // Otherwise silently fail — conversation history is nice-to-have
    }
  }

  // ---------------------------------------------------------------------------
  // Unified Event Stream (SSE — drift, tool follow-ups, delegate results)
  // ---------------------------------------------------------------------------

  _connectDriftStream() {
    this._closeDriftStream();

    let baseUrl = '/events/stream';
    if (this._backendHost) {
      baseUrl = this._backendHost.replace(/\/$/, '') + '/events/stream';
    }
    this._driftSource = new EventSource(baseUrl, { withCredentials: true });

    const handler = (e) => {
      try {
        const data = JSON.parse(e.data);
        this._handleEvent(data);
      } catch { /* ignore parse errors */ }
    };

    this._driftSource.addEventListener('drift', handler);
    this._driftSource.addEventListener('tool_followup', handler);
    this._driftSource.addEventListener('delegate_followup', handler);
    this._driftSource.addEventListener('response', handler);
    this._driftSource.addEventListener('card', handler);
    this._driftSource.addEventListener('reminder', handler);
    this._driftSource.addEventListener('task', handler);

    this._driftSource.onerror = () => {
      // EventSource auto-reconnects; nothing to do
    };
  }

  _closeDriftStream() {
    if (this._driftSource) {
      this._driftSource.close();
      this._driftSource = null;
    }
  }

  _handleEvent(data) {
    // Tool-status event — show ToolsCard in the spine
    if (data.type === 'tools') {
      this._toolsCard.fetch();
      return;
    }

    // Tool result card event
    if (data.type === 'card') {
      const cardEl = this._toolResultCard.build(data);
      this.renderer.appendToolCard(cardEl);
      return;
    }

    const content = data.content || '';
    if (!content) return;

    // Ignore 'response' events from the drift stream while a /chat SSE request
    // is in flight — the chat SSE already renders the reply via resolvePendingForm.
    if (data.type === 'response' && this._isSending) return;

    // Build metadata, including removed_by and removes if present
    const meta = {
      topic: data.topic,
      type: data.type,
      ts: new Date(),
    };
    if (data.removed_by) meta.removed_by = data.removed_by;
    if (data.removes) meta.removes = data.removes;

    // Render in conversation spine as a Chalie message
    this.renderer.appendChalieForm(content, meta);
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
        // Reconnect drift stream if it was closed
        if (!this._driftSource || this._driftSource.readyState === EventSource.CLOSED) {
          this._connectDriftStream();
        }
      }
    });
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
      await this.api.healthCheck();
      this._hideConnectionBanner();
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
    this._showLoginDialog();
  }

  _showLoginDialog() {
    return new Promise((resolve) => {
      const dialog = document.getElementById('loginDialog');
      const submitBtn = document.getElementById('loginSubmitBtn');
      const statusEl = document.getElementById('loginStatus');
      const usernameEl = document.getElementById('loginUsername');
      const passwordEl = document.getElementById('loginPassword');

      statusEl.textContent = '';
      statusEl.className = 'api-key-dialog__status';

      const doLogin = async () => {
        const username = usernameEl.value.trim();
        const password = passwordEl.value;
        if (!username || !password) {
          statusEl.textContent = 'Username and password required.';
          statusEl.className = 'api-key-dialog__status api-key-dialog__status--error';
          return;
        }
        submitBtn.disabled = true;
        submitBtn.textContent = 'Logging in...';
        try {
          const res = await fetch('/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'same-origin',
            body: JSON.stringify({ username, password }),
          });
          if (res.ok) {
            dialog.close();
            resolve();
            // Re-run init now that session is established
            this._init();
          } else {
            statusEl.textContent = res.status === 401 ? 'Invalid credentials.' : 'Login failed.';
            statusEl.className = 'api-key-dialog__status api-key-dialog__status--error';
            submitBtn.disabled = false;
            submitBtn.textContent = 'Login';
          }
        } catch {
          statusEl.textContent = 'Network error.';
          statusEl.className = 'api-key-dialog__status api-key-dialog__status--error';
          submitBtn.disabled = false;
          submitBtn.textContent = 'Login';
        }
      };

      submitBtn.onclick = doLogin;
      passwordEl.onkeydown = (e) => { if (e.key === 'Enter') doLogin(); };

      dialog.showModal();
    });
  }

  // ---------------------------------------------------------------------------
  // PWA Install Dialog
  // ---------------------------------------------------------------------------

  _initPwaDialog() {
    const dialog = document.getElementById('pwaInstallDialog');
    const closeBtn = dialog.querySelector('.pwa-dialog__close');
    const installBtn = document.getElementById('pwaInstallBtn');

    const dismiss = () => {
      localStorage.setItem('chalie_pwa_dismissed', '1');
      dialog.close();
      // Resume normal init flow (auth already checked in _init)
      this._start();
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
    if (localStorage.getItem('chalie_pwa_dismissed')) return;

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
}

// Boot
new ChalieApp();

/**
 * Chalie — Application bootstrap & orchestrator.
 */
import { ApiClient } from './api.js';
import { SSEClient } from './sse.js';
import { Presence } from './presence.js';
import { Renderer } from './renderer.js';
import { VoiceIO } from './voice.js';
import { ClientHeartbeat } from './heartbeat.js';
import { AmbientSensor } from './ambient.js';
import { MemoryCard } from './cards/memory.js';
import { TimelineCard } from './cards/timeline.js';
import { ToolResultCard } from './cards/tool_result.js';
import { MomentSearch } from './moment_search.js';
import { MomentCard } from './cards/moment.js';

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
    this._momentSearch = null;

    // Scheduler trigger dedup: topic → timestamp of last card event (60s window)
    this._recentToolCardTopics = new Map();
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
    this._initMoments();
    this._initInput();
    this._initAmbientSensor();
    this._initPwaDialog();
    this._initTaskStrip();
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
    this._showSparkOverlay();

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
      this._dismissSparkOverlay();
      this.presence.setState('resting');
      const voiceReady = await this.voice.init();
      if (voiceReady.stt) document.getElementById('micBtn')?.classList.remove('hidden');
      this.renderer.setTtsEnabled(voiceReady.tts);
      this._loadRecentConversation();
      this._loadActiveTasks();
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
    document.getElementById('settingsBtn')?.addEventListener('click', () => {
      window.open('/brain/', '_blank');
    });
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

    // Show moment event (from search overlay click)
    document.addEventListener('chalie:show-moment', (e) => {
      const { moment } = e.detail;
      const card = new MomentCard(moment);
      this.renderer.appendToolCard(card.build());
    });

    // Forget moment event (from Forget button on a moment card)
    document.addEventListener('chalie:forget-moment', (e) => {
      const { momentId, cardElement } = e.detail;

      // Dim the card to signal pending deletion — don't remove it yet
      cardElement.classList.add('moment-card--pending-forget');

      // Schedule actual forget after the undo window
      const forgetTimer = setTimeout(async () => {
        cardElement.classList.remove('moment-card--pending-forget');
        cardElement.classList.add('moment-card--forgetting');
        setTimeout(() => cardElement.remove(), 310);

        try {
          const base = backendHost ? backendHost.replace(/\/$/, '') : '';
          await fetch(base + `/moments/${momentId}/forget`, {
            method: 'POST',
            credentials: 'same-origin',
          });
        } catch (err) {
          console.warn('Forget moment failed:', err);
        }
      }, 10000);

      // Undo — cancel the timer and restore the card
      this._showToast('Forgotten', () => {
        clearTimeout(forgetTimer);
        cardElement.classList.remove('moment-card--pending-forget');
      }, 10000);
    });

    // First-use hint (one-time)
    if (!localStorage.getItem('moments_hint_shown')) {
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
    toast.innerHTML = `<span>${message}</span>`;

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
            if (btn && !localStorage.getItem('moments_hint_shown')) {
              localStorage.setItem('moments_hint_shown', '1');
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
      const data = await this.api._get('/system/observability/tasks');
      const tasks = (data.persistent_tasks || []).filter(
        t => t.status === 'accepted' || t.status === 'in_progress' || t.status === 'paused'
      );
      this._renderTaskStrip(tasks);
    } catch {
      // Silently fail — task strip is supplementary
    }
  }

  _renderTaskStrip(tasks) {
    const strip = document.getElementById('taskStrip');
    const list = document.getElementById('taskStripList');
    const countEl = document.getElementById('taskStripCount');
    if (!strip || !list) return;

    if (tasks.length === 0) {
      strip.classList.add('hidden');
      document.body.classList.remove('has-task-strip');
      return;
    }

    strip.classList.remove('hidden');
    document.body.classList.add('has-task-strip');
    countEl.textContent = tasks.length;

    let html = '';
    for (const t of tasks) {
      const goal = (t.goal || 'Working…').slice(0, 60);
      const progress = t.progress || {};
      const coverage = Math.round((progress.coverage_estimate || 0) * 100);
      const summary = progress.last_summary || '';
      const pausedClass = t.status === 'paused' ? ' --paused' : '';

      html += `<div class="task-strip__item${pausedClass}">
        <div class="task-strip__goal">${this._escHtml(goal)}</div>
        <div class="task-strip__progress-bar">
          <div class="task-strip__progress-fill" style="width:${coverage}%"></div>
        </div>
        ${summary ? `<div class="task-strip__summary">${this._escHtml(summary)}</div>` : ''}
      </div>`;
    }

    // First-time hint
    if (!localStorage.getItem('task_strip_hint_shown')) {
      localStorage.setItem('task_strip_hint_shown', '1');
      html += '<div class="task-strip__hint">I\'ll show what I\'m working on here.</div>';
    }

    list.innerHTML = html;
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

    // Render user form with timestamp
    this.renderer.appendUserForm(text, exchangeTimestamp);

    // Create pending form and store reference for potential early resolution
    const pendingForm = this.renderer.createPendingForm();
    this._pendingForm = pendingForm;

    // After 2 seconds, upgrade the "..." dots to a brief placeholder phrase
    const pendingUpgradeTimer = setTimeout(() => {
      this.renderer.upgradePendingText(pendingForm);
    }, 2000);

    let responseText = '';
    let responseMeta = {};

    await this.sse.send(text, source, {
      onStatus: (stage) => {
        this.presence.setState(stage);
      },
      onMessage: (data) => {
        clearTimeout(pendingUpgradeTimer);
        responseText = data.text;
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
        if (responseText) {
          responseMeta.duration_ms = data.duration_ms;
          responseMeta.ts = exchangeTimestamp;
          if (pendingForm.isConnected) {
            // Normal path: pending bubble still in the DOM
            this.renderer.resolvePendingForm(pendingForm, responseText, responseMeta);
          } else {
            // Card arrived via drift stream first — pending form was already removed.
            // Append the synthesis as a new message so it isn't silently lost.
            this.renderer.appendChalieForm(responseText, responseMeta);
          }
          this._pendingForm = null;
          // Notify if user switched away while waiting for the response
          if (!document.hasFocus()) {
            this._notifyBackground(responseText);
          }
        } else {
          // card-only: keep the pending bubble visible until the card arrives via drift stream
          this.renderer.upgradePendingText(pendingForm); // no-op if already upgraded
          // _pendingForm stays set; drift card handler will remove it
        }
        this.presence.setState('resting');
        this._isSending = false;
        // Re-enable input
        document.getElementById('messageInput').focus();
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
      if (!data.exchanges || data.exchanges.length === 0) {
        return;
      }

      for (const exchange of data.exchanges) {
        if (exchange.prompt) {
          this.renderer.appendUserForm(exchange.prompt, exchange.timestamp);
        }
        if (exchange.response) {
          this.renderer.appendChalieForm(exchange.response, {
            topic: exchange.topic,
            ts: exchange.timestamp,
            exchange_id: exchange.id,
          });
        }
      }
    } catch (err) {
      if (err.message === 'AUTH') {
        this._handleAuthFailure();
      }
      // Otherwise silently fail — conversation history is nice-to-have
    }
  }

  _showSparkOverlay() {
    const overlay = document.getElementById('sparkOverlay');
    const spine = document.getElementById('conversationSpine');
    const dock = document.querySelector('.input-dock');
    if (!overlay) return;

    overlay.classList.remove('hidden');
    if (spine) spine.style.display = 'none';
    if (dock) dock.style.display = 'none';

    // Skip button
    const skipBtn = overlay.querySelector('.spark-overlay__skip');
    if (skipBtn) {
      skipBtn.addEventListener('click', () => this._dismissSparkOverlay(), { once: true });
    }
  }

  _dismissSparkOverlay() {
    const overlay = document.getElementById('sparkOverlay');
    const spine = document.getElementById('conversationSpine');
    const dock = document.querySelector('.input-dock');

    if (overlay && !overlay.classList.contains('hidden')) {
      overlay.classList.add('spark-overlay--fading');
      setTimeout(() => {
        overlay.classList.add('hidden');
        overlay.classList.remove('spark-overlay--fading');
      }, 220);
    }

    if (spine) spine.style.display = '';
    if (dock) dock.style.display = '';
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
    this._driftSource.addEventListener('escalation', handler);

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
    // Task progress/completion — refresh the task strip
    if (data.type === 'task') {
      this._loadActiveTasks();
    }

    // Tool result card event
    if (data.type === 'card') {
      if (this._pendingForm) {
        this._pendingForm.remove();
        this._pendingForm = null;
      }
      const cardEl = this._toolResultCard.build(data);
      this.renderer.appendToolCard(cardEl);
      // Track this topic so a follow-up reminder/task event doesn't double-render
      if (data.topic) {
        this._recentToolCardTopics.set(data.topic, Date.now());
      }
      return;
    }

    const content = data.content || '';
    if (!content) return;

    // Ignore 'response' events from the drift stream while a /chat SSE request
    // is in flight — the chat SSE already renders the reply via resolvePendingForm.
    if (data.type === 'response' && this._isSending) return;

    // System notification + sound when tab is not focused
    if (!document.hasFocus()) {
      this._notifyBackground(content);
    }

    // Scheduler trigger events — render as styled trigger card or plain message
    if (data.type === 'notification') {
      this._handleSchedulerTrigger(data);
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
    const formEl = this.renderer.appendChalieForm(content, meta);

    // Spark presence messages get ambient treatment (softer, not conversational)
    if (data.topic && data.topic.startsWith('spark_')) {
      formEl.classList.add('speech-form--ambient');
    }

    // Critic escalation — amber border to signal "needs your attention"
    if (data.type === 'escalation') {
      formEl.classList.add('speech-form--escalation');
    }
  }

  _handleSchedulerTrigger(data) {
    // Expire stale entries from the tool card topic map (60s window)
    const now = Date.now();
    for (const [topic, ts] of this._recentToolCardTopics) {
      if (now - ts > 60000) this._recentToolCardTopics.delete(topic);
    }

    const topic = data.topic;
    const recentCard = topic && this._recentToolCardTopics.has(topic);

    if (recentCard) {
      // Tool card already visible — render as plain Chalie message to avoid duplication
      this.renderer.appendChalieForm(data.content, { topic, type: data.type, ts: new Date() });
      return;
    }

    // No tool card — render a styled trigger card and play sound
    const cardEl = this._buildTriggerCard(data);
    this.renderer.appendToolCard(cardEl);
    this._playScheduleSound();
  }

  _buildTriggerCard(data) {
    const label = 'Notification';
    const card = document.createElement('div');
    card.className = 'tool-result-card';
    card.setAttribute('data-tool', `scheduler_trigger`);

    const body = document.createElement('div');
    body.className = 'tool-result-card__body scheduler-trigger-card';

    const labelEl = document.createElement('div');
    labelEl.className = 'scheduler-trigger-card__label';

    const bellSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    bellSvg.setAttribute('width', '12');
    bellSvg.setAttribute('height', '12');
    bellSvg.setAttribute('viewBox', '0 0 24 24');
    bellSvg.setAttribute('fill', '#00F0FF');
    bellSvg.setAttribute('aria-hidden', 'true');
    bellSvg.style.cssText = 'flex-shrink:0;opacity:0.85;vertical-align:middle;margin-right:4px;';
    const bellPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    bellPath.setAttribute('d', 'M12 22c1.1 0 2-.9 2-2h-4c0 1.1.9 2 2 2zm6-6V11c0-3.07-1.64-5.64-4.5-6.32V4c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v.68C7.63 5.36 6 7.92 6 11v5l-2 2v1h16v-1l-2-2z');
    bellSvg.appendChild(bellPath);
    labelEl.appendChild(bellSvg);
    labelEl.appendChild(document.createTextNode(label));

    const textEl = document.createElement('div');
    textEl.className = 'scheduler-trigger-card__text';
    textEl.textContent = data.content;

    body.appendChild(labelEl);
    body.appendChild(textEl);
    card.appendChild(body);
    return card;
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
        // Reconnect drift stream if it was closed
        if (!this._driftSource || this._driftSource.readyState === EventSource.CLOSED) {
          this._connectDriftStream();
        }
        // Dismiss stale notifications now that the user is back
        this._dismissNotifications();
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

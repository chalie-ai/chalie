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

// Safe localStorage wrapper — private browsing on iOS Safari / Firefox throws SecurityError.
function _lsGet(key) { try { return localStorage.getItem(key); } catch { return null; } }
function _lsSet(key, val) { try { localStorage.setItem(key, val); } catch { /* ignore */ } }

class ChalieApp {
  constructor() {
    this._backendHost = _lsGet('chalie_backend_host') || '';
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
    // B10 fix: output_id-based dedup for card events (prevents SSE reconnect replays)
    this._seenCardIds = new Set();
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
    this._initUpload();
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
    this._readyPollActive = true;
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
      this._dismissSparkOverlay();
      this.presence.setState('resting');
      const voiceReady = await this.voice.init();
      if (voiceReady.stt) document.getElementById('micBtn')?.classList.remove('hidden');
      this.renderer.setTtsEnabled(voiceReady.tts);
      this._loadRecentConversation();
      this._loadActiveTasks();
      // Poll every 60s as a safety net for tasks that complete without a drift event
      this._taskStripInterval = setInterval(() => this._loadActiveTasks(), 60_000);
      this._connectDriftStream();
      window.addEventListener('beforeunload', () => {
        this._closeDriftStream();
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
        this.api._get('/system/observability/tasks').catch(() => ({})),
        this.api._get('/scheduler?status=pending').catch(() => ({})),
      ]);
      const tasks = (taskData.persistent_tasks || []).filter(
        t => t.status === 'accepted' || t.status === 'in_progress' || t.status === 'paused'
      );
      const reminders = (schedData.items || []).filter(
        r => r.status === 'pending' && r.due_at
      );
      this._renderTaskStrip(tasks, reminders);
    } catch {
      // Silently fail — task strip is supplementary
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

    let html = '';

    // Render persistent tasks
    for (const t of tasks) {
      const goal = (t.goal || 'Working…').slice(0, 60);
      const progress = t.progress || {};
      const coverage = Math.round((progress.coverage_estimate || 0) * 100);
      const summary = progress.last_summary || '';
      const pausedClass = t.status === 'paused' ? ' --paused' : '';

      // Step-level progress from plan DAG
      const plan = progress.plan;
      let stepsHtml = '';
      if (plan && plan.steps && plan.steps.length > 0) {
        const done = plan.steps.filter(s => s.status === 'completed' || s.status === 'skipped').length;
        const total = plan.steps.length;
        const current = plan.steps.find(s => s.status === 'in_progress');
        stepsHtml = `<div class="task-strip__steps">${done}/${total} steps</div>`;
        if (current) {
          stepsHtml += `<div class="task-strip__current-step">${this._escHtml(current.description)}</div>`;
        }
        if (plan.blocked_on) {
          stepsHtml += `<div class="task-strip__blocked">Blocked: ${this._escHtml(plan.blocked_reason || 'dependency failed')}</div>`;
        }
      }

      html += `<div class="task-strip__item${pausedClass}">
        <span class="task-strip__kind-dot task-strip__kind-dot--task"></span>
        <div class="task-strip__goal">${this._escHtml(goal)}</div>
        <div class="task-strip__progress-bar">
          <div class="task-strip__progress-fill" style="width:${coverage}%"></div>
        </div>
        ${stepsHtml}
        ${summary ? `<div class="task-strip__summary">${this._escHtml(summary)}</div>` : ''}
        <button class="task-strip__dismiss" data-dismiss-task="${t.id}" aria-label="Dismiss task">&times;</button>
      </div>`;
    }

    // Render pending reminders
    for (const r of reminders) {
      const msg = (r.message || '').slice(0, 80);
      const due = r.due_at ? this._relativeTime(r.due_at) : '';
      const id = r.id;

      html += `<div class="task-strip__item task-strip__item--reminder">
        <span class="task-strip__kind-dot task-strip__kind-dot--reminder"></span>
        <span class="task-strip__msg">${this._escHtml(msg)}</span>
        ${due ? `<span class="task-strip__due">${this._escHtml(due)}</span>` : ''}
        <button class="task-strip__dismiss" data-dismiss-reminder="${this._escHtml(id)}" aria-label="Dismiss reminder">&times;</button>
      </div>`;
    }

    // First-time hint
    if (!_lsGet('task_strip_hint_shown')) {
      _lsSet('task_strip_hint_shown', '1');
      html += '<div class="task-strip__hint">I\'ll show what I\'m working on here.</div>';
    }

    list.innerHTML = html;

    // Wire dismiss buttons — reminders
    list.querySelectorAll('[data-dismiss-reminder]').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const remId = btn.dataset.dismissReminder;
        try {
          await this.api._delete(`/scheduler/${remId}`);
        } catch { /* ignore */ }
        this._loadActiveTasks();
      });
    });

    // Wire dismiss buttons — persistent tasks
    list.querySelectorAll('[data-dismiss-task]').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const taskId = btn.dataset.dismissTask;
        try {
          await this.api._delete(`/system/observability/tasks/${taskId}`);
        } catch { /* ignore */ }
        this._loadActiveTasks();
      });
    });
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
      skipBtn.addEventListener('click', () => {
        this._readyPollActive = false;
        this._dismissSparkOverlay();
      }, { once: true });
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
    this._driftSource.addEventListener('notification', handler);

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
      // B10 fix: deduplicate card events on SSE reconnect
      const cardId = data.output_id
        || `${data.tool || 'unknown'}:${data.topic || ''}:${Math.floor(Date.now() / 5000)}`;
      if (this._seenCardIds.has(cardId)) return;
      this._seenCardIds.add(cardId);
      // Periodic cleanup (keep set bounded)
      if (this._seenCardIds.size > 200) {
        const arr = [...this._seenCardIds];
        this._seenCardIds = new Set(arr.slice(-100));
      }

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

    // Scheduler trigger events — show in task strip only, no chat card
    if (data.type === 'notification') {
      this._playScheduleSound();
      this._loadActiveTasks();  // Refresh strip; fired reminder changes status → drops out
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
      _lsSet('chalie_pwa_dismissed', '1');
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
    const btn = document.getElementById('uploadBtn');
    const dialog = document.getElementById('uploadDialog');
    const closeBtn = document.getElementById('uploadDialogClose');
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('uploadFileInput');
    const progress = document.getElementById('uploadProgress');
    const progressLabel = document.getElementById('uploadProgressLabel');
    const dupWarning = document.getElementById('uploadDuplicateWarning');

    if (!btn || !dialog) return;

    btn.addEventListener('click', () => {
      this._resetUploadDialog();
      dialog.showModal();
    });

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
          if (!hasSynthesis && attempts < maxAttempts) {
            // Synthesis LLM call still in progress — wait for it
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
    const name = this._esc(doc.original_name || 'Document');
    const typeTag = docType && docType !== 'document'
      ? `<span class="doc-synthesis__tag">${this._esc(docType)}</span>` : '';

    const factsHtml = keyFacts.length
      ? `<div class="doc-synthesis__facts">${keyFacts.map(f =>
          `<span class="doc-synthesis__fact">${this._esc(f)}</span>`).join('')}</div>`
      : '';

    const card = document.createElement('div');
    card.className = 'doc-synthesis-card';
    card.dataset.docId = docId;
    card.innerHTML = `
      <div class="doc-synthesis__header">
        <span class="doc-synthesis__name">${name}</span>
        ${typeTag}
      </div>
      <p class="doc-synthesis__text">${this._esc(synthesis)}</p>
      ${factsHtml}
      <div class="doc-synthesis__actions">
        <button class="doc-synthesis__btn doc-synthesis__btn--confirm">Looks good</button>
        <button class="doc-synthesis__btn doc-synthesis__btn--augment">Add context</button>
        <button class="doc-synthesis__btn doc-synthesis__btn--discard">Discard</button>
      </div>
      <div class="doc-synthesis__augment-area hidden">
        <textarea class="doc-synthesis__textarea"
          placeholder="Add context about this document..."></textarea>
        <button class="doc-synthesis__btn doc-synthesis__btn--submit">Save</button>
      </div>`;

    // Wire button handlers
    const confirmBtn = card.querySelector('.doc-synthesis__btn--confirm');
    const augmentBtn = card.querySelector('.doc-synthesis__btn--augment');
    const discardBtn = card.querySelector('.doc-synthesis__btn--discard');
    const augmentArea = card.querySelector('.doc-synthesis__augment-area');
    const submitBtn = card.querySelector('.doc-synthesis__btn--submit');
    const textarea = card.querySelector('.doc-synthesis__textarea');

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
}

// Boot
new ChalieApp();

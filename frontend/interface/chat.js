/**
 * Message sending orchestration and conversation history.
 */
export class Chat {
  /**
   * @param {{ api, ws, renderer, presence, notifications, imageAttach, documentUpload }} deps
   */
  constructor({ api, ws, renderer, presence, notifications, imageAttach, documentUpload }) {
    this._api = api;
    this._ws = ws;
    this._renderer = renderer;
    this._presence = presence;
    this._notifications = notifications;
    this._imageAttach = imageAttach || null;
    this._documentUpload = documentUpload || null;

    // Send state
    this._isSending = false;
    this._pendingForm = null;

    // Scroll-up pagination state
    this._historyOffset = 0;

    this._historyLoading = false;
    this._historyExhausted = false;
    this._historyLimit = 12;
    this._historyMaxTurns = 120;

    // Callbacks
    this._onAuthFailureCb = null;
    this._onResponseReceivedCb = null;
    this._onSendCompleteCb = null;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Bind the scroll-up pagination listener.
   * Must be called AFTER the initial history load and scroll-to-bottom so that
   * short conversations do not trigger a cascade of loads on startup.
   */
  init() {
    window.addEventListener('scroll', () => {
      const scrollable = document.documentElement.scrollHeight > window.innerHeight + 100;
      if (scrollable && window.scrollY < 150 && !this._historyLoading && !this._historyExhausted) {
        const anchor = document.body.scrollHeight - window.scrollY;
        this.loadRecentConversation().then(() => {
          window.scrollTo(0, document.body.scrollHeight - anchor);
        });
      }
    });
  }

  /** Whether a send is currently in-flight (read by event router and action handler). */
  get isSending() {
    return this._isSending;
  }

  /** Mark send state externally (used by action handler to block concurrent sends). */
  set isSending(val) {
    this._isSending = !!val;
  }

  /** Register callback for 401 auth failures. */
  onAuthFailure(cb) {
    this._onAuthFailureCb = cb;
  }

  /** Register callback when a response arrives (used by ambient sensor). */
  onResponseReceived(cb) {
    this._onResponseReceivedCb = cb;
  }

  /** Register callback when the full send cycle is complete (input re-enable, etc.). */
  onSendComplete(cb) {
    this._onSendCompleteCb = cb;
  }

  // ---------------------------------------------------------------------------
  // Stop
  // ---------------------------------------------------------------------------

  /**
   * Transition the send button to stop mode.
   * Called immediately when a send turn starts.
   */
  _enterStopMode() {
    const sendBtn = document.getElementById('sendBtn');
    if (!sendBtn) return;
    sendBtn.disabled = false;
    sendBtn.classList.remove('btn-action--send');
    sendBtn.classList.add('btn-action--stop');
    sendBtn.setAttribute('aria-label', 'Stop');
    sendBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
      <rect x="2" y="2" width="12" height="12" rx="2"/>
    </svg>`;
  }

  /**
   * Transition the send button to stopping mode (after stop clicked, before done).
   */
  _enterStoppingMode() {
    const sendBtn = document.getElementById('sendBtn');
    if (!sendBtn) return;
    sendBtn.disabled = true;
    sendBtn.classList.remove('btn-action--stop');
    sendBtn.classList.add('btn-action--stopping');
    sendBtn.setAttribute('aria-label', 'Stopping...');
  }

  /**
   * Revert the send button to normal send mode.
   * Called when a turn completes (done event) or is cancelled.
   */
  _exitStopMode() {
    const sendBtn = document.getElementById('sendBtn');
    if (!sendBtn) return;
    sendBtn.disabled = true;
    sendBtn.classList.remove('btn-action--stop', 'btn-action--stopping');
    sendBtn.classList.add('btn-action--send');
    sendBtn.setAttribute('aria-label', 'Send message');
    sendBtn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <line x1="22" y1="2" x2="11" y2="13"></line>
      <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
    </svg>`;
  }

  /**
   * Request stop of the active turn. Posts to /api/chat/stop and enters
   * stopping mode. The button remains disabled until the done event fires.
   */
  async requestStop() {
    this._enterStoppingMode();
    try {
      await this._api._post('/chat/stop', {});
    } catch (err) {
      console.warn('[Chat] Stop request failed:', err);
    }
  }

  // ---------------------------------------------------------------------------
  // Send
  // ---------------------------------------------------------------------------

  /**
   * Main send orchestrator.
   *
   * Reads #messageInput and #sendBtn from the DOM, detects steer routing,
   * manages the pending-form lifecycle, calls ws.send() with all callbacks,
   * and fires the registered completion hooks.
   *
   * @param {string} [source='text']
   */
  async sendMessage(source = 'text') {
    const textarea = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const text = textarea.value.trim();
    const attachments = this._imageAttach ? this._imageAttach.getAttachmentPaths() : [];

    if (!text && !attachments.length) return;

    // Block send while any upload is still in-flight.
    if (this._imageAttach?.isUploading) return;

    // Turn in-flight — route to server, which decides steer vs new turn.
    // Server returns {routed: "steer"|"turn"} and _postChat fires
    // onSteerSent on the active callbacks if it was a steer.
    if (this._ws._chatCallbacks) {
      textarea.value = '';
      textarea.style.height = 'auto';
      this._ws.send(text, source, {});
      return;
    }

    this._isSending = true;
    this._presence.setState('processing');
    textarea.value = '';
    textarea.style.height = 'auto';
    if (this._imageAttach) this._imageAttach.clear();
    this._enterStopMode();

    // Capture timestamp for this exchange
    const exchangeTimestamp = new Date();

    this._renderer.appendUserForm(text || '[File attached]', exchangeTimestamp, {});

    // ACT cycle host: blinking logo + (optional) narrative + cumulative tool list.
    // Persists for the entire turn; replaced wholesale by the final response.
    const actEl = this._renderer.createActCycle();
    this._pendingForm = actEl;

    let responseContent = '';
    let responseMeta = {};

    this._ws.send(text || '[File attached]', source, {
      onStatus: (stage) => {
        this._presence.setState(stage);
      },
      onNarration: (data) => {
        // Each new narration REPLACES the previous one — no stacking.
        this._presence.setState('narrating');
        this._renderer.setActNarrative(actEl, data.text, data.step);
      },
      onToolStart: (msg) => {
        // Tools accumulate in a single flat list spanning the whole ACT loop.
        this._renderer.appendToolPill(actEl, msg.call_id, msg.name);
      },
      onToolEnd: (msg) => {
        this._renderer.resolveToolPill(msg.call_id, msg.ms || 0, !!msg.ok);
      },
      onSteerSent: (steerText) => {
        this._renderer.appendSteerBubble(steerText, actEl);
      },
      onMessage: (data) => {
        responseContent = data.content || '';
        responseMeta = {
          topic: data.topic,
          exchange_id: data.exchange_id,
          mode: data.mode || '',
          confidence: data.confidence || 0,
          segments: data.segments || null,
        };
        this._presence.setState('responding');
      },
      onError: (data) => {
        this._renderer.replaceActWithError(actEl, data.message);
        if (!data.recoverable) {
          this._onAuthFailureCb?.();
        }
      },
      onDone: (data) => {
        this._onResponseReceivedCb?.();
        this._finaliseTurn(actEl, responseContent, responseMeta, data, exchangeTimestamp);
      },
    }, attachments);

  }

  // ---------------------------------------------------------------------------
  // History
  // ---------------------------------------------------------------------------

  /**
   * Load (or paginate) conversation history.
   * Safe to call multiple times — guards against concurrent loads and
   * exhausted history.
   */
  async loadRecentConversation() {
    if (this._historyLoading || this._historyExhausted) return;
    this._historyLoading = true;

    const loader = document.getElementById('historyLoader');
    if (loader) loader.style.display = 'flex';

    try {
      const data = await this._api.getRecentConversation({
        limit: this._historyLimit,
        offset: this._historyOffset,
      });

      const messages = data.messages || [];

      if (messages.length === 0 && this._historyOffset === 0) {
        this._historyExhausted = true;
        this._showHistoryEndPill();
        return;
      }

      const isInitialLoad = this._historyOffset === 0;
      if (isInitialLoad) {
        for (const msg of messages) this._appendMessage(msg, true);
      } else {
        for (let i = messages.length - 1; i >= 0; i--) {
          this._prependMessage(messages[i], false);
        }
      }

      this._historyOffset += messages.length;

      if (!data.has_more || this._historyOffset >= this._historyMaxTurns) {
        this._historyExhausted = true;
        this._showHistoryEndPill();
      }

      if (isInitialLoad && messages.length > 0) {
        this._renderer.forceScrollToBottom();
      }
    } catch (err) {
      if (err.message === 'AUTH') {
        this._onAuthFailureCb?.();
      } else {
        console.error('[Chat] Failed to load conversation history:', err);
      }
    } finally {
      if (loader) loader.style.display = 'none';
      this._historyLoading = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  /**
   * Finalise a completed send turn: swap or remove the ACT placeholder,
   * fire background notification if needed, then reset send state.
   */
  _finaliseTurn(actEl, responseContent, responseMeta, doneData, exchangeTimestamp) {
    if (responseContent) {
      responseMeta.duration_ms = doneData.duration_ms;
      responseMeta.ts = exchangeTimestamp;
      // The ACT cycle UI vanishes entirely; a normal chat bubble takes its place.
      this._renderer.replaceActWithResponse(actEl, responseContent, responseMeta);
      this._pendingForm = null;
      this._notifyBackgroundIfUnfocused(responseContent);
    } else {
      // No content response — remove the ACT placeholder
      if (actEl.isConnected) actEl.remove();
      this._pendingForm = null;
    }
    this._presence.setState('resting');
    this._isSending = false;
    this._exitStopMode();
    this._onSendCompleteCb?.();
  }

  /**
   * Fire a background (tab-unfocused) notification for the given text,
   * extracting plaintext from the XML response markup first.
   */
  _notifyBackgroundIfUnfocused(responseContent) {
    if (document.hasFocus()) return;
    import('./markup_extract.js').then(({ extractPlaintext }) => {
      const notifText = extractPlaintext(responseContent);
      if (notifText) this._notifications.notifyBackground(notifText);
    }).catch(() => {});
  }

  _appendMessage(msg, inWorkingMemory) {
    if (msg.role === 'user') {
      this._renderer.appendUserForm(msg.content, msg.timestamp, { inWorkingMemory });
    } else if (msg.content || msg.segments) {
      const meta = { ts: msg.timestamp };
      if (msg.segments) meta.segments = msg.segments;
      this._renderer.appendChalieForm(msg.content || '', meta, { inWorkingMemory });
    }
  }

  _prependMessage(msg, inWorkingMemory) {
    if (msg.role === 'user') {
      this._renderer.prependUserForm(msg.content, msg.timestamp, { inWorkingMemory });
    } else if (msg.content || msg.segments) {
      const meta = { ts: msg.timestamp };
      if (msg.segments) meta.segments = msg.segments;
      this._renderer.prependChalieForm(msg.content || '', meta, { inWorkingMemory });
    }
  }

  _showHistoryEndPill() {
    const pill = document.getElementById('historyEndPill');
    if (pill) pill.style.display = 'flex';
  }
}

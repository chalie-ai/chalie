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

    // The text of the most recently sent message; used to combine with a
    // mid-ACT redirect message when the user types while a turn is in-flight.
    this._lastSentText = '';

    // When set, _finaliseTurn redirects to a new turn with this payload
    // instead of settling into a normal response. Set by sendMessage() when
    // the user submits a message while a turn is already in-flight.
    this._pendingRedirect = null;

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
  // Stop / Interrupt
  // ---------------------------------------------------------------------------

  /**
   * Request a clean stop of the active turn.
   * Clears any pending redirect (plain stop, not a redirect), disables the
   * ACT-cycle stop button, and POSTs to /chat/interrupt.
   */
  async requestStop() {
    this._pendingRedirect = null;
    const stopBtn = this._pendingForm?.querySelector('.act-stop-btn');
    if (stopBtn) {
      stopBtn.disabled = true;
      stopBtn.classList.add('act-stop-btn--stopping');
    }
    await this._postInterrupt();
  }

  // ---------------------------------------------------------------------------
  // Send
  // ---------------------------------------------------------------------------

  /**
   * Main send orchestrator.
   *
   * If a turn is already in-flight, shows the new message as a user bubble,
   * stores a pending redirect (combined message), and POSTs /chat/interrupt.
   * The done event from the interrupted turn triggers _finaliseTurn, which
   * detects the redirect and starts a fresh turn with the combined text.
   *
   * @param {string} [source='text']
   */
  async sendMessage(source = 'text') {
    const textarea = document.getElementById('messageInput');
    const text = textarea.value.trim();
    const attachments = this._imageAttach ? this._imageAttach.getAttachmentPaths() : [];

    if (!text && !attachments.length) return;

    // Block send while any upload is still in-flight.
    if (this._imageAttach?.isUploading) return;

    // Mid-ACT interrupt — cancel the current turn and redirect with a combined
    // message (original text + separator + new text).
    if (this._isSending) {
      textarea.value = '';
      textarea.style.height = 'auto';
      this._renderer.appendUserForm(text, null, {});
      const combined = this._lastSentText + '\n\n' + text;
      this._pendingRedirect = { text: combined, source };
      this._postInterrupt();
      return;
    }

    this._isSending = true;
    this._lastSentText = text || '[File attached]';
    this._presence.setState('processing');
    textarea.value = '';
    textarea.style.height = 'auto';
    if (this._imageAttach) this._imageAttach.clear();

    this._startTurn(text || '[File attached]', source, true, attachments);
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
   * Wire and launch a turn: create ACT cycle, register WS callbacks, post.
   *
   * Extracted from sendMessage() so the redirect path in _finaliseTurn can
   * start a fresh turn without duplicating callback wiring.
   *
   * @param {string} text — message body to send
   * @param {string} source — "text" | "voice" | "subagent" etc.
   * @param {boolean} showUserBubble — whether to render a user speech-form
   * @param {string[]} attachments — file paths from POST /upload
   */
  _startTurn(text, source, showUserBubble = true, attachments = []) {
    if (showUserBubble) {
      this._renderer.appendUserForm(text || '[File attached]', null, {});
    }

    const actEl = this._renderer.createActCycle();
    this._pendingForm = actEl;

    // Wire the stop button embedded in the ACT cycle element.
    const stopBtn = actEl.querySelector('.act-stop-btn');
    if (stopBtn) stopBtn.addEventListener('click', () => this.requestStop());

    let responseContent = '';
    let responseMeta = {};

    this._ws.send(text, source, {
      onStatus: (stage) => this._presence.setState(stage),
      onNarration: (data) => {
        this._presence.setState('narrating');
        this._renderer.setActNarrative(actEl, data.text, data.step);
      },
      onToolStart: (msg) => {
        this._renderer.appendToolPill(actEl, msg.call_id, msg.name, msg.act_summary);
      },
      onToolEnd: (msg) => {
        this._renderer.resolveToolPill(msg.call_id, msg.ms || 0, !!msg.ok);
      },
      onMessage: (data) => {
        responseContent = data.content || '';
        responseMeta = {
          topic: data.topic,
          exchange_id: data.exchange_id,
          mode: data.mode || '',
          confidence: data.confidence || 0,
          segments: data.segments || null,
          ts: data.timestamp || '',
        };
        this._presence.setState('responding');
      },
      onError: (data) => {
        this._renderer.replaceActWithError(actEl, data.message);
        if (!data.recoverable) this._onAuthFailureCb?.();
      },
      onDone: (data) => {
        this._onResponseReceivedCb?.();
        this._finaliseTurn(actEl, responseContent, responseMeta, data);
      },
    }, attachments);
  }

  /**
   * Finalise a completed send turn.
   *
   * If _pendingRedirect is set (mid-ACT interrupt case), removes the current
   * ACT element and immediately starts a new turn with the combined message.
   * Otherwise, swaps or removes the ACT placeholder and resets send state.
   */
  _finaliseTurn(actEl, responseContent, responseMeta, doneData) {
    // Redirect: the previous turn was interrupted; start a fresh turn with
    // the combined original+new message. No user bubble — the new message
    // bubble was already rendered when the user typed it mid-ACT.
    if (this._pendingRedirect) {
      const { text, source } = this._pendingRedirect;
      this._pendingRedirect = null;
      if (actEl.isConnected) actEl.remove();
      this._pendingForm = null;
      this._lastSentText = text;
      this._startTurn(text, source, false);
      return;
    }

    if (responseContent) {
      responseMeta.duration_ms = doneData.duration_ms;
      // The ACT cycle UI vanishes entirely; a normal chat bubble takes its place.
      this._renderer.replaceActWithResponse(actEl, responseContent, responseMeta);
      this._pendingForm = null;
      this._notifyBackgroundIfUnfocused(responseContent);
    } else {
      // No content response — remove the ACT placeholder.
      if (actEl.isConnected) actEl.remove();
      this._pendingForm = null;
    }
    this._presence.setState('resting');
    this._isSending = false;
    this._onSendCompleteCb?.();
  }

  /**
   * POST to /chat/interrupt. Returns silently on failure — the cancel signal
   * is best-effort; the UI handles the done event regardless.
   */
  async _postInterrupt() {
    try {
      await this._api._post('/chat/interrupt', {});
    } catch (err) {
      console.warn('[Chat] Interrupt request failed:', err);
    }
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

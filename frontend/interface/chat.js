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
    this._lastUserBubble = null;

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
   * Stop + undo: cancel the active turn, remove the ACT cycle and user
   * bubble, and restore the message text to the input box.
   */
  async requestStop() {
    // Restore message text to textarea
    const textarea = document.getElementById('messageInput');
    if (this._lastUserBubble) {
      const textEl = this._lastUserBubble.querySelector('.speech-form__text');
      if (textEl && textarea) textarea.value = textEl.textContent || '';
      this._lastUserBubble.remove();
      this._lastUserBubble = null;
    }

    // Remove ACT cycle element
    if (this._pendingForm?.isConnected) this._pendingForm.remove();
    this._pendingForm = null;

    // Abort WS callbacks so stale events are ignored
    this._ws.abort();

    // Reset send state
    this._isSending = false;
    this._presence.setState('resting');
    this._onSendCompleteCb?.();

    // Fire cancel to backend (best-effort, cleanup happens server-side)
    this._postInterrupt();
  }

  // ---------------------------------------------------------------------------
  // Send
  // ---------------------------------------------------------------------------

  /**
   * Main send orchestrator.
   *
   * If a turn is already in-flight, appends the new text to the existing user
   * bubble and POSTs /chat. The backend cancels the active turn, concatenates
   * the original + new message, and starts a fresh turn with the combined text.
   *
   * @param {string} [source='text']
   */
  async sendMessage(source = 'text') {
    const textarea = document.getElementById('messageInput');
    const text = textarea.value.trim();
    // Raw File objects ride the multipart POST /chat (no pre-upload).
    const files = this._imageAttach ? this._imageAttach.getFiles() : [];
    // Capture preview metadata (object URLs, filenames) BEFORE clear() wipes the
    // strip — used to render the attachments inside the user's own turn bubble.
    const attachmentPreviews = this._imageAttach ? this._imageAttach.getAttachments() : [];

    if (!text && !files.length) return;

    // Mid-ACT: append to existing user bubble, remove old ACT cycle, and
    // POST /chat. The backend cancels the active turn, concatenates original +
    // new message, and starts a fresh turn whose events arrive on fresh callbacks.
    if (this._isSending) {
      textarea.value = '';
      textarea.style.height = 'auto';
      if (this._lastUserBubble) {
        const textEl = this._lastUserBubble.querySelector('.speech-form__text');
        if (textEl) textEl.textContent += '\n\n' + text;
      }
      if (this._pendingForm?.isConnected) this._pendingForm.remove();
      this._pendingForm = null;
      this._startTurn(text, source, false);
      return;
    }

    this._isSending = true;
    this._presence.setState('processing');
    textarea.value = '';
    textarea.style.height = 'auto';
    if (this._imageAttach) this._imageAttach.clear();

    this._startTurn(text || '[File attached]', source, true, files, attachmentPreviews);
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
   * @param {File[]} files — raw File objects appended to the multipart POST /chat
   * @param {Array<{filename: string, objectUrl: string|null, isImage: boolean}>} attachmentPreviews
   *        — preview metadata rendered inside the user bubble
   */
  _startTurn(text, source, showUserBubble = true, files = [], attachmentPreviews = []) {
    if (showUserBubble) {
      this._lastUserBubble = this._renderer.appendUserForm(text || '[File attached]', null, {
        attachments: attachmentPreviews,
      });
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
        // Backend (abilities/_base.py) emits the act_tool_start event with
        // keys `id` and `summary`; align to that contract (was call_id /
        // act_summary, which silently dropped every pill via the !callId guard
        // in renderer.appendToolPill).
        this._renderer.appendToolPill(actEl, msg.id, msg.name, msg.summary);
      },
      onToolEnd: (msg) => {
        // Backend sends `id` and `ok` but no `ms`; pass 0 and let the renderer
        // fall back to the client-measured elapsed for the duration display.
        this._renderer.resolveToolPill(msg.id, msg.ms || 0, !!msg.ok);
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
        // A turn-level error (provider failure, quota/429, tool error) is NOT
        // an auth event — surface it in place and let the turn finalise via the
        // `done` event that always follows. Genuine session loss is detected by
        // its own channels: the heartbeat (/auth/status poll) and api.js (any
        // 401 → 'AUTH'). Only an explicit auth signal triggers the login
        // redirect — never a generic non-recoverable turn error, which used to
        // bounce an authenticated user through /login/ and reload the page,
        // discarding the turn.
        if (data.auth_failed) this._onAuthFailureCb?.();
      },
      onDone: (data) => {
        this._onResponseReceivedCb?.();
        this._finaliseTurn(actEl, responseContent, responseMeta, data);
      },
    }, files);
  }

  /**
   * Finalise a completed send turn.
   * Swaps or removes the ACT placeholder and resets send state.
   */
  _finaliseTurn(actEl, responseContent, responseMeta, doneData) {
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

  /**
   * Map persisted attachment refs from /conversation/recent into the shape the
   * renderer expects. The preview URL becomes the <img>/chip source, re-rendering
   * the upload that the live blob: preview showed before refresh.
   */
  _attachmentsFor(msg) {
    return (msg.attachments || []).map(a => ({
      filename: a.filename,
      objectUrl: a.url,
      isImage: a.is_image,
    }));
  }

  _appendMessage(msg, inWorkingMemory) {
    if (msg.role === 'user') {
      const attachments = this._attachmentsFor(msg);
      this._renderer.appendUserForm(msg.content, msg.timestamp, { inWorkingMemory, attachments });
    } else if (msg.content || msg.segments) {
      const meta = { ts: msg.timestamp };
      if (msg.segments) meta.segments = msg.segments;
      this._renderer.appendChalieForm(msg.content || '', meta, { inWorkingMemory });
    }
  }

  _prependMessage(msg, inWorkingMemory) {
    if (msg.role === 'user') {
      const attachments = this._attachmentsFor(msg);
      this._renderer.prependUserForm(msg.content, msg.timestamp, { inWorkingMemory, attachments });
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

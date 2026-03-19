/**
 * Message sending orchestration and conversation history.
 */
export class Chat {
  /**
   * @param {{ api, ws, renderer, presence, notifications, imageAttach }} deps
   */
  constructor({ api, ws, renderer, presence, notifications, imageAttach }) {
    this._api = api;
    this._ws = ws;
    this._renderer = renderer;
    this._presence = presence;
    this._notifications = notifications;
    this._imageAttach = imageAttach || null;

    // Send state
    this._isSending = false;
    this._pendingForm = null;

    // Scroll-up pagination state
    this._historyOffset = 0;
    this._historyTotal = 0;
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
    const imageIds = this._imageAttach ? this._imageAttach.getImageIds() : [];

    if (!text && !imageIds.length) return;

    // Only route as steer when the ACT loop is actively narrating.
    // This prevents normal replies (e.g. to clarifications) from being
    // misrouted as steering commands.
    if (this._ws._chatCallbacks && this._presence.state === 'narrating') {
      textarea.value = '';
      textarea.style.height = 'auto';
      this._ws.send(text, source, {});
      return;
    }

    this._isSending = true;
    this._presence.setState('processing');
    textarea.value = '';
    textarea.style.height = 'auto';
    const pendingImageIds = [...imageIds];
    if (this._imageAttach) this._imageAttach.clear();
    sendBtn.disabled = true;

    // Capture timestamp for this exchange
    const exchangeTimestamp = new Date();

    // Render user form with timestamp (pass imageIds so thumbnails are shown inline)
    this._renderer.appendUserForm(text || '[Image attached]', exchangeTimestamp, {}, pendingImageIds);

    // Create pending form and store reference for potential early resolution
    const pendingForm = this._renderer.createPendingForm();
    this._pendingForm = pendingForm;

    // After 2 seconds, upgrade the "..." dots to a brief placeholder phrase
    const pendingUpgradeTimer = setTimeout(() => {
      this._renderer.upgradePendingText(pendingForm);
    }, 2000);

    let responseBlocks = [];
    let responseMeta = {};

    this._ws.send(text || '[Image attached]', source, {
      onStatus: (stage) => {
        this._presence.setState(stage);
      },
      onNarration: (data) => {
        // Live ACT narration: remove thinking dots, show narration bubble instead
        clearTimeout(pendingUpgradeTimer);
        // Remove the pending form entirely — narration bubbles replace it
        if (pendingForm.isConnected) pendingForm.remove();
        this._presence.setState('narrating');
        this._renderer.appendNarrationBubble(data.text, data.step);
      },
      onSteerSent: (steerText) => {
        // User sent a redirect while ACT is running — show inline
        this._renderer.appendSteerBubble(steerText);
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
        this._presence.setState('responding');
      },
      onError: (data) => {
        clearTimeout(pendingUpgradeTimer);
        this._renderer.resolvePendingFormError(pendingForm, data.message);
        if (!data.recoverable) {
          this._onAuthFailureCb?.();
        }
      },
      onDone: (data) => {
        clearTimeout(pendingUpgradeTimer);
        this._onResponseReceivedCb?.();
        if (responseBlocks.length) {
          responseMeta.duration_ms = data.duration_ms;
          responseMeta.ts = exchangeTimestamp;
          this._renderer.resolvePendingForm(pendingForm, responseBlocks, responseMeta);
          this._pendingForm = null;
          // Notify if user switched away while waiting for the response
          if (!document.hasFocus()) {
            const notifText = responseBlocks
              .filter(b => b.type === 'text' || b.type === 'header')
              .map(b => b.content || '')
              .join(' ');
            if (notifText) this._notifications.notifyBackground(notifText);
          }
        } else {
          // No blocks response — remove the pending bubble
          pendingForm.remove();
          this._pendingForm = null;
        }
        this._presence.setState('resting');
        this._isSending = false;
        this._onSendCompleteCb?.();
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
        this._renderer.forceScrollToBottom();
      }
    } catch (err) {
      if (err.message === 'AUTH') {
        this._onAuthFailureCb?.();
      }
      // Otherwise silently fail — conversation history is nice-to-have
    } finally {
      if (loader) loader.style.display = 'none';
      this._historyLoading = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  _appendExchange(exchange, inWorkingMemory) {
    if (exchange.prompt) {
      this._renderer.appendUserForm(exchange.prompt, exchange.timestamp, { inWorkingMemory });
    }
    if (exchange.blocks?.length) {
      this._renderer.appendChalieForm(exchange.blocks, {
        topic: exchange.topic,
        ts: exchange.timestamp,
        exchange_id: exchange.id,
      }, { inWorkingMemory });
    }
  }

  _prependExchange(exchange, inWorkingMemory) {
    if (exchange.blocks?.length) {
      this._renderer.prependChalieForm(exchange.blocks, {
        topic: exchange.topic,
        ts: exchange.timestamp,
        exchange_id: exchange.id,
      }, { inWorkingMemory });
    }
    if (exchange.prompt) {
      this._renderer.prependUserForm(exchange.prompt, exchange.timestamp, { inWorkingMemory });
    }
  }

  _showHistoryEndPill() {
    const pill = document.getElementById('historyEndPill');
    if (pill) pill.style.display = 'flex';
  }
}

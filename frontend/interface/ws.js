/**
 * WebSocket client — receive-only server→client push channel.
 *
 * All client→server requests use HTTP:
 *   POST /chat           — user message (multipart: text + raw file attachments)
 *   POST /chat/interrupt — cancel the active UMP turn
 *   POST /action         — action button click
 *
 * The WebSocket delivers server-pushed events:
 *   ← status, message, act_tool_start, act_tool_end,
 *     done, error, ping, drift/task/reminder/escalation/notification,
 *     permission_request, intent, capability_alert
 *
 * The only client→server WS message is pong (keepalive response).
 *
 * Reconnect strategy: a clean `onclose` triggers exponential-backoff
 * reconnection. A reverse proxy can also idle-drop the socket at the TCP layer
 * without sending a close frame (half-open) — `onclose` never fires — so a
 * liveness watchdog force-reconnects when the backend's periodic pings stop
 * arriving. `ensureAlive()` performs the same check on demand (e.g. tab refocus).
 */
export class WSClient {
  /**
   * @param {() => string} getHost — returns the current backend host
   */
  constructor(getHost) {
    this._getHost = getHost;
    this._ws = null;
    this._reconnectDelay = 1000;
    this._maxReconnectDelay = 30000;
    this._reconnectTimer = null;
    this._chatCallbacks = null;
    this._driftHandler = null;
    this._anyHandler = null;
    this._disconnectHandler = null;
    this._connected = false;
    this._intentionallyClosed = false;
    // Liveness watchdog (half-open detection). The backend pings every 60s of
    // client silence (backend/api/websocket.py `_ws_handler`); if no frame
    // arrives for longer than the stale threshold, the pipe is dead even though
    // the browser still reports readyState === OPEN.
    this._staleThresholdMs = 90000;  // 1.5× the 60s server ping — fire on full silence only
    this._livenessCheckMs = 30000;   // how often the watchdog re-checks
    this._lastInboundAt = 0;
    this._livenessTimer = null;
  }

  _buildWsUrl() {
    const host = this._getHost?.() || '';
    let base;
    if (host) {
      base = host.replace(/\/$/, '');
    } else {
      base = globalThis.location.origin;
    }
    const wsBase = base.replace(/^http/, 'ws');
    return wsBase + '/ws';
  }

  _buildHttpUrl(path) {
    const host = this._getHost?.() || '';
    let base;
    if (host) {
      base = host.replace(/\/$/, '');
    } else {
      base = globalThis.location.origin;
    }
    return base + path;
  }

  /**
   * Set the handler for drift/push events (cards, tasks, etc.)
   * @param {(data: object) => void} handler
   */
  onDrift(handler) {
    this._driftHandler = handler;
  }

  /**
   * Set a handler invoked for EVERY inbound WS message, before any type-specific
   * routing. Used by the composer to re-fetch the context-size indicator on each
   * message. Must never throw — failures are swallowed so the WS pipe is safe.
   * @param {(data: object) => void} handler
   */
  onAny(handler) {
    this._anyHandler = handler;
  }

  /**
   * Set handler called when the WS connection drops (before reconnect).
   * @param {() => void} handler
   */
  onDisconnect(handler) {
    this._disconnectHandler = handler;
  }

  connect() {
    if (this._ws && (this._ws.readyState === WebSocket.OPEN || this._ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this._intentionallyClosed = false;
    const url = this._buildWsUrl();

    try {
      this._ws = new WebSocket(url);
    } catch (err) {
      console.warn('[WS] Failed to create WebSocket:', err);
      this._scheduleReconnect();
      return;
    }

    // A backoff reconnect we scheduled earlier is now redundant — this connect()
    // owns the new socket (e.g. a tab refocus called ensureAlive() while a
    // _scheduleReconnect timer was still pending). Clear it so it cannot fire a
    // second, no-op connect().
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }

    this._ws.onopen = () => {
      this._connected = true;
      this._reconnectDelay = 1000;
      // Count connect time as the last inbound so the first server ping (≤60s)
      // has a full window to arrive before the watchdog can judge staleness.
      this._lastInboundAt = Date.now();
      this._startLivenessWatch();
    };

    this._ws.onmessage = (event) => {
      // Any inbound frame proves the pipe is alive — stamp before parsing so a
      // malformed frame still counts toward liveness.
      this._lastInboundAt = Date.now();
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }
      this._dispatch(data);
    };

    this._ws.onclose = () => {
      this._connected = false;
      this._stopLivenessWatch();
      this._disconnectHandler?.();
      if (!this._intentionallyClosed) {
        this._scheduleReconnect();
      }
    };

    this._ws.onerror = () => {
      // onclose will fire after onerror — reconnect handled there
    };
  }

  close() {
    this._intentionallyClosed = true;
    this._stopLivenessWatch();
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
    this._connected = false;
  }

  get isConnected() {
    return this._connected && this._ws?.readyState === WebSocket.OPEN;
  }

  _scheduleReconnect() {
    if (this._intentionallyClosed) return;
    if (this._reconnectTimer) return;

    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this.connect();
    }, this._reconnectDelay);

    this._reconnectDelay = Math.min(this._reconnectDelay * 1.5, this._maxReconnectDelay);
  }

  // ---------------------------------------------------------------------------
  // Liveness watchdog — detect a half-open socket (silent reverse-proxy idle
  // drop) that never fires `onclose`, and force a reconnect. Without this, the
  // socket reports readyState === OPEN forever, `isConnected` lies, and pushed
  // replies vanish into the dead pipe while the user sees nothing.
  // ---------------------------------------------------------------------------

  _startLivenessWatch() {
    this._stopLivenessWatch();
    this._livenessTimer = setInterval(() => this._checkLiveness(), this._livenessCheckMs);
  }

  _stopLivenessWatch() {
    if (this._livenessTimer) {
      clearInterval(this._livenessTimer);
      this._livenessTimer = null;
    }
  }

  _checkLiveness() {
    if (this._intentionallyClosed) return;
    if (Date.now() - this._lastInboundAt > this._staleThresholdMs) {
      this._forceReconnect();
    }
  }

  /**
   * Tear down a socket that is open at the browser level but has gone silent,
   * then immediately reconnect. The stale socket's handlers are detached first
   * so its delayed `onclose` (fired once the browser finally gives up) cannot
   * trigger a second, competing reconnect.
   */
  _forceReconnect() {
    this._stopLivenessWatch();
    const stale = this._ws;
    this._ws = null;
    this._connected = false;
    if (stale) {
      stale.onopen = null;
      stale.onmessage = null;
      stale.onclose = null;
      stale.onerror = null;
      try { stale.close(); } catch { /* already dead — nothing to close */ }
    }
    this._disconnectHandler?.();
    this.connect();
  }

  /**
   * Verify the connection is genuinely alive and heal it if not. Used when the
   * tab regains focus: `isConnected` alone is unreliable because a half-open
   * socket still reports readyState === OPEN.
   */
  ensureAlive() {
    if (this._intentionallyClosed) return;
    if (!this.isConnected) {
      this.connect();
      return;
    }
    if (Date.now() - this._lastInboundAt > this._staleThresholdMs) {
      this._forceReconnect();
    }
  }

  abort() {
    this._chatCallbacks = null;
  }

  /**
   * Send a chat message via POST /chat. Response arrives via WS push.
   *
   * Always registers fresh callbacks and starts a new UMP turn. Mid-ACT
   * user messages are handled by Chat.sendMessage() via POST /chat/interrupt
   * before calling this method.
   *
   * @param {string} text
   * @param {"text"|"voice"} source
   * @param {{
   *   onStatus?:    (stage: string) => void,
   *   onMessage?:   (data: object) => void,
   *   onError?:     (data: object) => void,
   *   onDone?:      (data: object) => void,
   *   onToolStart?: (data: object) => void,
   *   onToolEnd?:   (data: object) => void,
   * }} callbacks
   * @param {File[]} [files] - raw File objects appended to the multipart request
   */
  send(text, source, callbacks = {}, files = []) {
    this._chatCallbacks = callbacks;

    if (!this.isConnected) {
      callbacks.onError?.({ message: 'Not connected. Please wait...', recoverable: true });
      callbacks.onDone?.({ duration_ms: 0 });
      this._chatCallbacks = null;
      return;
    }

    this._postChat(text, source, files);
  }

  /**
   * Send a deterministic action (button click) via POST /action.
   *
   * @param {object} payload — action payload from the button
   * @param {{
   *   onMessage?: (data: object) => void,
   *   onError?:   (data: object) => void,
   *   onDone?:    (data: object) => void,
   * }} callbacks
   */
  sendAction(payload, callbacks = {}) {
    this.abort();
    this._chatCallbacks = callbacks;

    if (!this.isConnected) {
      callbacks.onError?.({ message: 'Not connected.', recoverable: true });
      callbacks.onDone?.({ duration_ms: 0 });
      this._chatCallbacks = null;
      return;
    }

    fetch(this._buildHttpUrl('/action'), {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).catch(() => {
      callbacks.onError?.({ message: 'Action request failed.', recoverable: true });
      callbacks.onDone?.({ duration_ms: 0 });
      this._chatCallbacks = null;
    });
  }

  _postChat(text, source, files) {
    // multipart/form-data — text + raw files. The browser sets the multipart
    // boundary, so no Content-Type header is supplied. The backend ingests each
    // file via document.upload by PATH, never bytes through the act-trail.
    const form = new FormData();
    form.append('text', text);
    form.append('source', source);
    for (const file of files || []) form.append('files', file, file.name);

    fetch(this._buildHttpUrl('/chat'), {
      method: 'POST',
      credentials: 'same-origin',
      body: form,
    })
      .catch(() => {
        this._chatCallbacks?.onError?.({ message: 'Chat request failed.', recoverable: true });
        this._chatCallbacks?.onDone?.({ duration_ms: 0 });
        this._chatCallbacks = null;
      });
  }

  _dispatch(data) {
    // Fire the "any message" hook first so it observes every inbound message
    // (including pings and chat events), regardless of type-specific routing.
    if (this._anyHandler) {
      try { this._anyHandler(data); } catch { /* never break the WS pipe */ }
    }

    const type = data.type;

    if (type === 'ping') {
      if (this._ws?.readyState === WebSocket.OPEN) {
        this._ws.send(JSON.stringify({ type: 'pong' }));
      }
      return;
    }

    if (this._chatCallbacks) {
      switch (type) {
        case 'status':
          this._chatCallbacks.onStatus?.(data.stage);
          return;
        case 'act_tool_start':
          this._chatCallbacks.onToolStart?.(data);
          return;
        case 'act_tool_end':
          this._chatCallbacks.onToolEnd?.(data);
          return;
        case 'message':
          this._chatCallbacks.onMessage?.(data);
          return;
        case 'error':
          this._chatCallbacks.onError?.(data);
          return;
        case 'done': {
          // Null _chatCallbacks BEFORE calling onDone so that if onDone
          // triggers a new send() (redirect turn), the new callbacks are not
          // immediately overwritten by this done handler.
          const cb = this._chatCallbacks;
          this._chatCallbacks = null;
          cb?.onDone?.(data);
          return;
        }
      }
    }

    if (this._driftHandler) {
      this._driftHandler(data);
    }
  }
}

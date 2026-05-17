/**
 * WebSocket client — receive-only server→client push channel.
 *
 * All client→server requests use HTTP:
 *   POST /chat    — user message (new turn or steer)
 *   POST /action  — action button click
 *   POST /upload  — file attachment
 *
 * The WebSocket delivers server-pushed events:
 *   ← status, message, act_narration, act_tool_start, act_tool_end,
 *     done, error, ping, drift/task/reminder/escalation/notification,
 *     permission_request, intent, capability_alert
 *
 * The only client→server WS message is pong (keepalive response).
 * On disconnect, the page reloads to re-establish state.
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
    this._connected = false;
    this._intentionallyClosed = false;
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

    this._ws.onopen = () => {
      this._connected = true;
      this._reconnectDelay = 1000;
    };

    this._ws.onmessage = (event) => {
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

  abort() {
    this._chatCallbacks = null;
  }

  /**
   * Send a chat message via POST /chat. Response arrives via WS push.
   *
   * If a chat is already in-flight and this is a steer (empty callbacks),
   * the server routes it as a steer injection into the active ACT loop.
   *
   * @param {string} text
   * @param {"text"|"voice"} source
   * @param {{
   *   onStatus?:    (stage: string) => void,
   *   onMessage?:   (data: object) => void,
   *   onNarration?: (data: object) => void,
   *   onError?:     (data: object) => void,
   *   onDone?:      (data: object) => void,
   *   onSteerSent?: (text: string) => void,
   *   onToolStart?: (data: object) => void,
   *   onToolEnd?:   (data: object) => void,
   * }} callbacks
   * @param {string[]} [attachments] - tmp_paths from POST /upload
   */
  send(text, source, callbacks = {}, attachments = []) {
    if (this._chatCallbacks) {
      this._chatCallbacks.onSteerSent?.(text);
      this._postChat(text, source, []);
      return;
    }

    this._chatCallbacks = callbacks;

    if (!this.isConnected) {
      callbacks.onError?.({ message: 'Not connected. Please wait...', recoverable: true });
      callbacks.onDone?.({ duration_ms: 0 });
      this._chatCallbacks = null;
      return;
    }

    this._postChat(text, source, attachments);
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

  _postChat(text, source, attachments) {
    const body = { text, source };
    if (attachments?.length) body.attachments = attachments;

    fetch(this._buildHttpUrl('/chat'), {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .catch(() => {
        this._chatCallbacks?.onError?.({ message: 'Chat request failed.', recoverable: true });
        this._chatCallbacks?.onDone?.({ duration_ms: 0 });
        this._chatCallbacks = null;
      });
  }

  _dispatch(data) {
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
        case 'act_narration':
          this._chatCallbacks.onNarration?.(data);
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
        case 'done':
          this._chatCallbacks.onDone?.(data);
          this._chatCallbacks = null;
          return;
      }
    }

    if (this._driftHandler) {
      this._driftHandler(data);
    }
  }
}

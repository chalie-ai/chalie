/**
 * WebSocket client — single bidirectional channel replacing both SSE streams.
 *
 * Handles:
 *   - Chat request/response (replaces POST-based SSEClient)
 *   - Drift events: cards, tasks, reminders, escalations (replaces EventSource)
 *   - Reconnect with sequence-based catch-up
 *   - Keepalive ping/pong
 */
export class WSClient {
  /**
   * @param {() => string} getHost — returns the current backend host
   */
  constructor(getHost) {
    this._getHost = getHost;
    this._ws = null;
    this._lastSeq = 0;
    this._reconnectDelay = 1000;
    this._maxReconnectDelay = 30000;
    this._reconnectTimer = null;
    this._chatCallbacks = null;   // active chat request callbacks
    this._driftHandler = null;    // drift event handler
    this._connected = false;
    this._intentionallyClosed = false;
  }

  /**
   * Build WebSocket URL from HTTP host.
   */
  _buildWsUrl() {
    const host = this._getHost?.() || '';
    let base;
    if (host) {
      base = host.replace(/\/$/, '');
    } else {
      base = window.location.origin;
    }
    // Convert http(s) to ws(s)
    const wsBase = base.replace(/^http/, 'ws');
    return wsBase + '/ws';
  }

  /**
   * Set the handler for drift/push events (cards, tasks, etc.)
   * @param {(data: object) => void} handler
   */
  onDrift(handler) {
    this._driftHandler = handler;
  }

  /**
   * Connect to the WebSocket server.
   */
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

      // If reconnecting, send resume with last sequence number
      if (this._lastSeq > 0) {
        this._send({ type: 'resume', last_seq: this._lastSeq });
      }
    };

    this._ws.onmessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }

      // Track sequence numbers
      if (data.seq) {
        this._lastSeq = data.seq;
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

  /**
   * Close the WebSocket connection.
   */
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

  /**
   * Whether the WebSocket is currently open.
   */
  get isConnected() {
    return this._connected && this._ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Schedule a reconnect with exponential backoff.
   */
  _scheduleReconnect() {
    if (this._intentionallyClosed) return;
    if (this._reconnectTimer) return;

    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this.connect();
    }, this._reconnectDelay);

    // Exponential backoff
    this._reconnectDelay = Math.min(this._reconnectDelay * 1.5, this._maxReconnectDelay);
  }

  /**
   * Send a JSON message.
   */
  _send(data) {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(data));
    }
  }

  /**
   * Abort any in-flight chat request (clears callbacks).
   */
  abort() {
    this._chatCallbacks = null;
  }

  /**
   * Send a chat message via WebSocket.
   *
   * @param {string} text
   * @param {"text"|"voice"} source
   * @param {{
   *   onStatus?:  (stage: string) => void,
   *   onMessage?: (data: object) => void,
   *   onCard?:    (data: object) => void,
   *   onError?:   (data: object) => void,
   *   onDone?:    (data: object) => void,
   * }} callbacks
   */
  send(text, source, callbacks = {}) {
    this.abort();
    this._chatCallbacks = callbacks;

    if (!this.isConnected) {
      callbacks.onError?.({ message: 'Not connected. Please wait...', recoverable: true });
      callbacks.onDone?.({ duration_ms: 0 });
      this._chatCallbacks = null;
      return;
    }

    this._send({ type: 'chat', text, source });
  }

  /**
   * Dispatch incoming messages to appropriate handlers.
   */
  _dispatch(data) {
    const type = data.type;

    // Ping/pong keepalive
    if (type === 'ping') {
      this._send({ type: 'pong' });
      return;
    }

    // Chat response events (while a chat is in-flight)
    if (this._chatCallbacks) {
      switch (type) {
        case 'status':
          this._chatCallbacks.onStatus?.(data.stage);
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

    // Drift/push events → delegate to drift handler
    if (this._driftHandler) {
      this._driftHandler(data);
    }
  }
}

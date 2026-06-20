import type { GetHost } from './types';

// Server → client inbound events, discriminated on `type`.
export interface WsStatusEvent {
  type: 'status';
  stage: string;
}
export interface WsActNarrationEvent {
  type: 'act_narration';
  [k: string]: unknown;
}
export interface WsActToolStartEvent {
  type: 'act_tool_start';
  [k: string]: unknown;
}
export interface WsActToolEndEvent {
  type: 'act_tool_end';
  [k: string]: unknown;
}
export interface WsMessageEvent {
  type: 'message';
  /**
   * True for a chain step's interim assistant prose: text streams ahead of the
   * step's tool batch with NO trailing `done`. Absent/false on the final turn
   * result, which carries the rich segments and is followed by `done`.
   */
  interim?: boolean;
  [k: string]: unknown;
}
export interface WsErrorEvent {
  type: 'error';
  message: string;
  recoverable?: boolean;
}
export interface WsDoneEvent {
  type: 'done';
  duration_ms: number;
}
export interface WsPingEvent {
  type: 'ping';
}

/** Push/drift family. */
export type WsPushType =
  | 'drift'
  | 'task'
  | 'reminder'
  | 'escalation'
  | 'notification'
  | 'permission_request'
  | 'intent'
  | 'capability_alert'
  | 'app_update'
  | 'quick_tip'
  | 'subagent_start'
  | 'subagent_end'
  | 'thought'
  | 'response'
  // Echo: server re-broadcasts every user message so surfaces stay in sync; the
  // sender drops its own via echo_id, peers render the bubble.
  | 'user_message';
export interface WsPushEvent {
  type: WsPushType;
  [k: string]: unknown;
}

export type WsInboundEvent =
  | WsStatusEvent
  | WsActNarrationEvent
  | WsActToolStartEvent
  | WsActToolEndEvent
  | WsMessageEvent
  | WsErrorEvent
  | WsDoneEvent
  | WsPingEvent
  | WsPushEvent;

export interface ChatCallbacks {
  onStatus?: (stage: string) => void;
  onMessage?: (data: WsMessageEvent) => void;
  onNarration?: (data: WsActNarrationEvent) => void;
  onError?: (data: { message: string; recoverable?: boolean }) => void;
  onDone?: (data: { duration_ms: number }) => void;
  onToolStart?: (data: WsActToolStartEvent) => void;
  onToolEnd?: (data: WsActToolEndEvent) => void;
}

export interface ActionCallbacks {
  onMessage?: (data: WsMessageEvent) => void;
  onError?: (data: { message: string; recoverable?: boolean }) => void;
  onDone?: (data: { duration_ms: number }) => void;
}

type Timer = ReturnType<typeof setTimeout>;
type Interval = ReturnType<typeof setInterval>;

/**
 * WebSocket client — receive-only server→client push channel. Client→server
 * requests go over HTTP (POST /chat, /chat/interrupt, /action); the only
 * client→server WS frame is `pong`.
 */
export class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectDelay = 1000;
  private readonly maxReconnectDelay = 30000;
  private reconnectTimer: Timer | null = null;
  private chatCallbacks: (ChatCallbacks & ActionCallbacks) | null = null;
  private driftHandler: ((data: WsPushEvent) => void) | null = null;
  private anyHandler: ((data: WsInboundEvent) => void) | null = null;
  private disconnectHandler: (() => void) | null = null;
  private connectHandler: (() => void) | null = null;
  private connected = false;
  private intentionallyClosed = false;

  // Echo ids this surface minted. It rendered its own message optimistically, so
  // it drops any broadcast echo whose id it owns. Bounded so a missed echo can't
  // grow the set forever.
  private readonly ownEchoIds = new Set<string>();
  private readonly maxOwnEchoIds = 64;

  // Half-open detection: backend pings every 60s of client silence; fire only on
  // full silence past 90s.
  private readonly staleThresholdMs = 90000;
  private readonly livenessCheckMs = 30000;
  private lastInboundAt = 0;
  private livenessTimer: Interval | null = null;

  constructor(private readonly getHost: GetHost) {}

  private baseUrl(): string {
    const host = this.getHost();
    return host ? host.replace(/\/$/, '') : globalThis.location.origin;
  }
  private buildWsUrl(): string {
    return this.baseUrl().replace(/^http/, 'ws') + '/ws';
  }
  private buildHttpUrl(path: string): string {
    return this.baseUrl() + path;
  }

  onDrift(handler: (data: WsPushEvent) => void): void {
    this.driftHandler = handler;
  }
  onAny(handler: (data: WsInboundEvent) => void): void {
    this.anyHandler = handler;
  }
  onDisconnect(handler: () => void): void {
    this.disconnectHandler = handler;
  }
  onConnect(handler: () => void): void {
    this.connectHandler = handler;
  }

  connect(): void {
    if (
      this.ws &&
      (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }
    this.intentionallyClosed = false;
    let ws: WebSocket;
    try {
      ws = new WebSocket(this.buildWsUrl());
    } catch (err) {
      console.warn('[WS] Failed to create WebSocket:', err);
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;

    // This connect() owns the socket, so a pending backoff reconnect is redundant.
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    ws.onopen = () => {
      this.connected = true;
      this.reconnectDelay = 1000;
      this.lastInboundAt = Date.now();
      this.startLivenessWatch();
      try {
        this.connectHandler?.();
      } catch {
        /* never break onopen */
      }
    };
    ws.onmessage = (event: MessageEvent) => {
      this.lastInboundAt = Date.now();
      let data: WsInboundEvent;
      try {
        data = JSON.parse(event.data as string) as WsInboundEvent;
      } catch {
        return;
      }
      this.dispatch(data);
    };
    ws.onclose = () => {
      this.connected = false;
      this.stopLivenessWatch();
      this.disconnectHandler?.();
      if (!this.intentionallyClosed) this.scheduleReconnect();
    };
    ws.onerror = () => {
      /* onclose fires after onerror — reconnect handled there */
    };
  }

  close(): void {
    this.intentionallyClosed = true;
    this.stopLivenessWatch();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.connected = false;
  }

  get isConnected(): boolean {
    return this.connected && this.ws?.readyState === WebSocket.OPEN;
  }

  private scheduleReconnect(): void {
    if (this.intentionallyClosed || this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, this.reconnectDelay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, this.maxReconnectDelay);
  }

  private startLivenessWatch(): void {
    this.stopLivenessWatch();
    this.livenessTimer = setInterval(() => this.checkLiveness(), this.livenessCheckMs);
  }
  private stopLivenessWatch(): void {
    if (this.livenessTimer) {
      clearInterval(this.livenessTimer);
      this.livenessTimer = null;
    }
  }
  private checkLiveness(): void {
    if (this.intentionallyClosed) return;
    if (Date.now() - this.lastInboundAt > this.staleThresholdMs) this.forceReconnect();
  }
  private forceReconnect(): void {
    this.stopLivenessWatch();
    const stale = this.ws;
    this.ws = null;
    this.connected = false;
    if (stale) {
      stale.onopen = null;
      stale.onmessage = null;
      stale.onclose = null;
      stale.onerror = null;
      try {
        stale.close();
      } catch {
        /* already dead */
      }
    }
    this.disconnectHandler?.();
    this.connect();
  }

  /** Heal the connection if it has silently died (tab refocus). */
  ensureAlive(): void {
    if (this.intentionallyClosed) return;
    if (!this.isConnected) {
      this.connect();
      return;
    }
    if (Date.now() - this.lastInboundAt > this.staleThresholdMs) this.forceReconnect();
  }

  abort(): void {
    this.chatCallbacks = null;
  }

  send(text: string, source: 'text' | 'voice', callbacks: ChatCallbacks = {}, files: File[] = []): void {
    this.chatCallbacks = callbacks;
    if (!this.isConnected) {
      callbacks.onError?.({ message: 'Not connected. Please wait...', recoverable: true });
      callbacks.onDone?.({ duration_ms: 0 });
      this.chatCallbacks = null;
      return;
    }
    this.postChat(text, source, files, this.mintEchoId());
  }

  /**
   * Mint a globally-unique echo id and remember it so this surface ignores its
   * own broadcast echo. Uniqueness must be global (every surface checks echoes
   * against its own set), hence crypto.randomUUID with a random-token fallback
   * for non-secure contexts.
   */
  private mintEchoId(): string {
    const id =
      globalThis.crypto?.randomUUID?.() ??
      `echo-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    this.ownEchoIds.add(id);
    if (this.ownEchoIds.size > this.maxOwnEchoIds) {
      const oldest = this.ownEchoIds.values().next().value;
      if (oldest !== undefined) this.ownEchoIds.delete(oldest);
    }
    return id;
  }

  sendAction(payload: unknown, callbacks: ActionCallbacks = {}): void {
    this.abort();
    this.chatCallbacks = callbacks;
    if (!this.isConnected) {
      callbacks.onError?.({ message: 'Not connected.', recoverable: true });
      callbacks.onDone?.({ duration_ms: 0 });
      this.chatCallbacks = null;
      return;
    }
    fetch(this.buildHttpUrl('/action'), {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).catch(() => {
      callbacks.onError?.({ message: 'Action request failed.', recoverable: true });
      callbacks.onDone?.({ duration_ms: 0 });
      this.chatCallbacks = null;
    });
  }

  private postChat(text: string, source: string, files: File[], echoId: string): void {
    const form = new FormData();
    form.append('text', text);
    form.append('source', source);
    form.append('echo_id', echoId);
    for (const file of files) form.append('files', file, file.name);
    fetch(this.buildHttpUrl('/chat'), {
      method: 'POST',
      credentials: 'same-origin',
      body: form,
    }).catch(() => {
      this.chatCallbacks?.onError?.({ message: 'Chat request failed.', recoverable: true });
      this.chatCallbacks?.onDone?.({ duration_ms: 0 });
      this.chatCallbacks = null;
    });
  }

  private dispatch(data: WsInboundEvent): void {
    if (this.anyHandler) {
      try {
        this.anyHandler(data);
      } catch {
        /* never break the WS pipe */
      }
    }
    if (data.type === 'ping') {
      if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: 'pong' }));
      return;
    }
    // Echo: if we minted the id we already rendered it — drop. Otherwise a peer
    // sent it, so route to the drift handler to render here too.
    if (data.type === 'user_message') {
      const echoId = (data as { echo_id?: string }).echo_id;
      if (echoId && this.ownEchoIds.has(echoId)) {
        this.ownEchoIds.delete(echoId);
        return;
      }
      this.driftHandler?.(data as WsPushEvent);
      return;
    }
    if (this.chatCallbacks) {
      switch (data.type) {
        case 'status':
          this.chatCallbacks.onStatus?.(data.stage);
          return;
        case 'act_narration':
          this.chatCallbacks.onNarration?.(data);
          return;
        case 'act_tool_start':
          this.chatCallbacks.onToolStart?.(data);
          return;
        case 'act_tool_end':
          this.chatCallbacks.onToolEnd?.(data);
          return;
        case 'message':
          this.chatCallbacks.onMessage?.(data);
          return;
        case 'error':
          this.chatCallbacks.onError?.(data);
          return;
        case 'done': {
          const cb = this.chatCallbacks;
          this.chatCallbacks = null;
          cb?.onDone?.(data);
          return;
        }
      }
    }
    this.driftHandler?.(data as WsPushEvent);
  }
}

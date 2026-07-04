import type { GetHost } from './types';
import { ConfigType } from '../config/configType';

// Server → client inbound events, discriminated on `type`.
export interface WsMessageEvent {
  type: 'message';
  [k: string]: unknown;
}
export interface WsErrorEvent {
  type: 'error';
  message: string;
  recoverable?: boolean;
}
export interface WsPingEvent {
  type: 'ping';
}

/** Mid-turn progress signals — the sole WS turn chokepoint (MessageProcessor.broadcast).
 *  Discriminated on `status`; `type` carries the ProcessorConfig identity (`user` →
 *  main spine, `scheduled` → dock + that schedule's thread) the surface routes by.
 *  The surface reacts by refetching the turn block over REST, so these frames carry
 *  only ids (plus a tool call's id/name/summary for the live act-trail timer) — never
 *  prose. `provider_retry` is the one status shown directly (a transient toast): a
 *  provider call failed mid-turn and the backend is resending, so the turn stays in
 *  flight (no error bubble). Turn *lifecycle* (working/done) is a separate frame —
 *  see `WsTurnExecutionEvent` below. */
export type WsTurnStatus = 'updated' | 'tool_called' | 'tool_done' | 'provider_retry';
export interface WsTurnSignal {
  status: WsTurnStatus;
  /** ConfigType — the ProcessorConfig identity the FE routes by. */
  type: string;
  turn_id?: number | null;
  [k: string]: unknown;
}

/** One `turn_executions` row, pushed whole on every state flip (see
 *  services/execution_tracker.py). It carries no frame-kind tag of its own —
 *  `type` here is the SAME ConfigType-identity field `WsTurnSignal.type` carries,
 *  so a surface recognises this frame by the presence of `state` instead, exactly
 *  as it recognises a `WsTurnSignal` by the presence of `status`. `working` is the
 *  sole non-terminal state; `completed` / `cancelled` / `crashed` all settle the
 *  turn the same way — a crash now reaches the surface too, so a turn that dies
 *  before producing a reply never leaves a spinner hanging. */
export type TurnExecutionState = 'working' | 'completed' | 'cancelled' | 'crashed';
export interface WsTurnExecutionEvent {
  id: number;
  channel: string;
  /** ConfigType — the ProcessorConfig identity the FE routes by. */
  type: string | null;
  turn_id: number;
  started_at: string;
  ended_at: string | null;
  cancel_requested: boolean;
  state: TurnExecutionState;
  stop_reason: string | null;
}

/** Push family — content/side-channel frames routed through the drift handler,
 *  discriminated on `type` (distinct from the turn signals above, keyed on `status`).
 *  The turn-level `error` toast arrives here too (WsErrorEvent). */
export type WsPushType =
  | 'task'
  | 'reminder'
  | 'notification'
  | 'permission_request'
  | 'intent'
  | 'capability_alert'
  | 'app_update'
  | 'quick_tip'
  | 'subagent_start'
  | 'subagent_end';
export interface WsPushEvent {
  type: WsPushType;
  [k: string]: unknown;
}

export type WsInboundEvent =
  | WsMessageEvent
  | WsErrorEvent
  | WsPingEvent
  | WsTurnSignal
  | WsTurnExecutionEvent
  | WsPushEvent;

/** Action channel (rich-card button → POST /action) is the only per-request
 *  binding left: it sets these callbacks for the optimistic card render. The
 *  user-turn channel is signal-only (see WsPushType) and never binds. */
export interface ActionCallbacks {
  onMessage?: (data: WsMessageEvent) => void;
  onError?: (data: { message: string; recoverable?: boolean }) => void;
  onDone?: (data: { duration_ms?: number }) => void;
}

type Timer = ReturnType<typeof setTimeout>;
type Interval = ReturnType<typeof setInterval>;

/**
 * WebSocket client — receive-only server→client push channel. Client→server
 * requests go over HTTP (POST /api/thread/<turn_id>, DELETE /api/thread/<turn_id>,
 * POST /api/action); the only client→server WS frame is `pong`.
 */
export class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectDelay = 1000;
  private readonly maxReconnectDelay = 30000;
  private reconnectTimer: Timer | null = null;
  private chatCallbacks: ActionCallbacks | null = null;
  private driftHandler: ((data: WsPushEvent) => void) | null = null;
  private anyHandler: ((data: WsInboundEvent) => void) | null = null;
  private disconnectHandler: (() => void) | null = null;
  private connectHandler: (() => void) | null = null;
  private connected = false;
  private intentionallyClosed = false;

  // Half-open detection: backend pings every 60s of client silence; fire only on
  // full silence past 90s.
  private readonly staleThresholdMs = 90000;
  private readonly livenessCheckMs = 30000;
  private lastInboundAt = 0;
  private livenessTimer: Interval | null = null;

  constructor(
    private readonly getHost: GetHost,
    private readonly getToken: GetHost,
  ) {}

  /**
   * Bearer header when a token is configured, else `{}`. Spreading the empty
   * object is a no-op, so the web (cookie) path is unchanged.
   */
  private authHeaders(): Record<string, string> {
    const token = this.getToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  private baseUrl(): string {
    const host = this.getHost();
    return host ? host.replace(/\/$/, '') : globalThis.location.origin;
  }
  private buildWsUrl(): string {
    // The bearer token is NOT carried in the URL — a query-string credential
    // leaks into reverse-proxy/access logs, Referer, and history. The native
    // client sends it as the first WS frame instead (see ``connect``).
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
      // Native clients have no cookie, so they present the bearer token as the
      // first WS frame — never in the URL. The web (cookie) path sends none.
      const token = this.getToken();
      if (token) {
        try {
          ws.send(JSON.stringify({ type: 'auth', token }));
        } catch {
          /* frame send failure surfaces as onclose → reconnect */
        }
      }
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
      this.notifyDisconnect();
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

  /** Connection lost: drop any in-flight action's callbacks before notifying
   *  listeners. Its done went to the dead socket and is never resent, so a
   *  lingering closure would otherwise swallow a later action's done. The turn
   *  channel is signal-only — surfaces resync by refetching on reconnect. */
  private notifyDisconnect(): void {
    this.chatCallbacks = null;
    this.disconnectHandler?.();
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
    this.notifyDisconnect();
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

  /** Fire a user turn. Signal-only: the turn's render flows entirely through the
   *  `updated` broadcast signal plus the `turn_execution` lifecycle frame
   *  (→ drift handler → REST refetch), so no callbacks are bound. `onSendFailure`
   *  reports ONLY a local send failure (offline / POST rejected) — the case where
   *  no signal will ever arrive — so the surface can release its send guard.
   *
   *  Resolves with the POST body `{turn_id, type}` so the caller can bind the
   *  lane handle the moment the server allocates it (no WS round-trip needed). */
  send(
    text: string,
    onSendFailure: (message: string) => void = () => {
      /* no-op */
    },
    files: File[] = [],
    threadId: number | null = null,
    type: string = ConfigType.USER,
  ): Promise<{ turn_id: number; type: string } | null> {
    if (!this.isConnected) {
      onSendFailure('Not connected. Please wait...');
      return Promise.resolve(null);
    }
    return this.postChat(text, files, threadId, onSendFailure, type);
  }

  /** Rich-card control dispatch — POST /action and resolve the card's optimistic
   *  UI from the SYNCHRONOUS HTTP response. No data crosses the WS bus. The
   *  callbacks bag doubles as an abort token: abort()/notifyDisconnect() null it
   *  and a newer sendAction supersedes it. When the request settles superseded we
   *  suppress the stale data callbacks (onMessage/onError) but ALWAYS fire onDone
   *  so the card releases its loading state instead of hanging. */
  sendAction(payload: unknown, callbacks: ActionCallbacks = {}): void {
    this.chatCallbacks = callbacks;
    const start = Date.now();
    fetch(this.buildHttpUrl('/api/action'), {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', ...this.authHeaders() },
      body: JSON.stringify(payload),
    })
      .then(async (resp) => {
        const data = (await resp.json().catch(() => ({}))) as Record<string, unknown>;
        const current = this.chatCallbacks === callbacks;
        if (current) {
          this.chatCallbacks = null;
          if (resp.ok) callbacks.onMessage?.(data as unknown as WsMessageEvent);
          else
            callbacks.onError?.({
              message: typeof data.error === 'string' ? data.error : 'Action failed.',
              recoverable: true,
            });
        }
        callbacks.onDone?.({ duration_ms: Number(data.duration_ms ?? Date.now() - start) });
      })
      .catch(() => {
        if (this.chatCallbacks === callbacks) {
          this.chatCallbacks = null;
          callbacks.onError?.({ message: 'Action request failed.', recoverable: true });
        }
        callbacks.onDone?.({ duration_ms: 0 });
      });
  }

  private postChat(
    text: string,
    files: File[],
    threadId: number | null,
    onSendFailure: (message: string) => void,
    type: string = ConfigType.USER,
  ): Promise<{ turn_id: number; type: string } | null> {
    const form = new FormData();
    form.append('text', text);
    form.append('type', type);
    for (const file of files) form.append('files', file, file.name);
    // POST /api/thread/<turn_id> — -1 creates a new thread, a real id replies
    // into it. Both return HTTP 200 with {turn_id, type} — the caller binds the
    // lane handle from this body the moment the server allocates it, with no WS
    // round-trip. ``type`` is the ProcessorConfig identity the server resolves
    // to a transcript channel.
    const path = `/api/thread/${threadId ?? -1}`;
    return fetch(this.buildHttpUrl(path), {
      method: 'POST',
      credentials: 'same-origin',
      headers: { ...this.authHeaders() },
      body: form,
    })
      .then(async (resp) => {
        if (!resp.ok) {
          onSendFailure('Chat request failed.');
          return null;
        }
        return (await resp.json()) as { turn_id: number; type: string };
      })
      .catch(() => {
        onSendFailure('Chat request failed.');
        return null;
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
    // Receive-only signal bus: every server frame is a turn signal. It carries no
    // content — the drift handler reacts by refetching the turn over REST.
    this.driftHandler?.(data as WsPushEvent);
  }
}

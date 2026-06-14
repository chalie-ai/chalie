/**
 * Session store — WS coordinator, turn state-machine, drift router.
 *
 * Single WS owner rule: ONLY this store may touch WebSocketService handlers
 * (send/sendAction/abort/onDrift/onAny/onConnect/onDisconnect/connect/ensureAlive).
 * Everything else goes through this store or the event bus.
 *
 * Port sources:
 *   - chat.js: sendMessage, _startTurn, _finaliseTurn, requestStop,
 *     _postInterrupt, loadRecentConversation
 *   - event_router.js: _handleEvent, _routeSimpleEvent, _routeThoughtEvent,
 *     _routeNotificationEvent, _renderContentEvent
 *   - app.js: chalie:action / chalie:silent-action wiring, sendAction
 */
import { defineStore } from 'pinia';
import { getWebSocket, useConnectionStore, AuthError } from '@chalie/shared';
import type { WsInboundEvent, WsPushEvent, WsMessageEvent } from '@chalie/shared';
import { on } from '../composables/useEventBus';
import { extractText } from '../composables/useMarkup';
import { conversation } from '../api/conversation';
import { moments } from '../api/moments';
import { getHost } from '../api/index';
import { showToast } from '../utils/toast';
import { useConversationStore } from './conversation';
import type { AttachmentPreview } from './conversation';
import type { ConversationSegment } from '../api/conversation';
import { usePresenceStore } from './presence';
import { useTasksStore } from './tasks';
import { useNotificationsStore } from './notifications';
import type { TipState, UpdateState } from './notifications';
import { usePermissionsStore } from './permissions';
import { useContextUsageStore } from './contextUsage';
import { useAmbientSensor } from '../composables/useAmbientSensor';

// ── Module-level state ────────────────────────────────────────────────────────

/** Guard: init() must be idempotent (HMR / Vue StrictMode). */
let _initialized = false;

/** Last pin-moment timestamp (ms) — 250ms debounce (port of app.js:475-479). */
let _pinDebounce = 0;

/**
 * Unbind functions for event bus listeners registered in init().
 * Stored in an array so the guard can check it and future cleanup can call them.
 */
const _busUnbinds: Array<() => void> = [];

// ── Store ─────────────────────────────────────────────────────────────────────

export const useSessionStore = defineStore('session', {
  state: () => ({
    isSending: false,
    /** Id of the active ACT form while a turn is in-flight. */
    _activeActId: null as number | null,
    /** Id of the last user form (for mid-ACT restore on requestStop). */
    _lastUserFormId: null as number | null,
    /** Captured text from the last user turn (for requestStop restore). */
    _lastUserText: '',

    // History pagination state
    historyOffset: 0,
    historyLoading: false,
    historyExhausted: false,

    /** Registered auth-failure callback (set by App bootstrap). */
    _onAuthFailure: null as (() => void) | null,
  }),

  actions: {
    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /**
     * Wire the WebSocket singleton and start the connection.
     * Idempotent — safe to call multiple times (HMR / StrictMode).
     */
    init(): void {
      if (_initialized) return;
      _initialized = true;

      const ws = getWebSocket();
      const conn = useConnectionStore();
      const contextUsage = useContextUsageStore();

      ws.onConnect(() => {
        conn.setConnected(true);
      });

      ws.onDisconnect(() => {
        conn.setConnected(false);
        this.isSending = false;
      });

      ws.onDrift((data: WsPushEvent) => {
        this.routeDrift(data);
      });

      // onAny feeds the context-usage indicator — must never throw.
      // refresh() is coalesced inside the store, so a per-frame call is safe
      // (port of chat_controls.js `this._ws.onAny(() => this.refreshContext())`).
      ws.onAny((data: WsInboundEvent) => {
        try {
          void data;
          void contextUsage.refresh();
        } catch {
          /* never break the WS pipe */
        }
      });

      // Tab-refocus liveness check (port of app.js visibility listener)
      globalThis.addEventListener('focus', () => ws.ensureAlive());

      // chalie:action — deterministic skill invocations (app.js lines 436-460).
      // WS single-owner: register inside init() so only one listener is ever bound.
      _busUnbinds.push(
        on('chalie:action', (payload) => {
          // app.js reads `payload` from e.detail.payload but the bus emits
          // the full detail; support both shapes.
          const p =
            (payload as { payload?: Record<string, unknown> }).payload ??
            (payload as Record<string, unknown>);
          void this.sendAction(p);
        }),
      );

      // chalie:pin-moment — Remember button: pin plaintext to moments store.
      // Exact port of app.js:474-508: 250ms debounce, single POST /moments,
      // then a "Remembered" / "Already remembered" toast with an Undo action
      // (Undo → POST /moments/<transcript_id>/forget). The Remember flow has NO
      // confirmation dialog — MomentSearchDialog is recall-only, matching
      // legacy moment_search.js.
      _busUnbinds.push(
        on('chalie:pin-moment', (detail) => {
          const text = (detail as { content?: string }).content ?? '';
          if (!text) return;
          const now = Date.now();
          if (now - _pinDebounce < 250) return;
          _pinDebounce = now;
          void moments
            .pin(text)
            .then((res) => {
              const transcriptId = res.item?.transcript_id ?? null;
              const msg = res.duplicate ? 'Already remembered' : 'Remembered';
              showToast(
                msg,
                transcriptId != null ? () => void moments.forget(transcriptId) : null,
              );
            })
            .catch((err: unknown) => {
              console.warn('[Session] pin moment failed:', err);
            });
        }),
      );

      // chalie:silent-action — rich-card interactions, no chat bubble (app.js 462-472).
      // The caller supplies optional onMessage/onError/onDone for optimistic card UI.
      _busUnbinds.push(
        on('chalie:silent-action', (detail) => {
          const d = detail as {
            payload?: Record<string, unknown>;
            onMessage?: (data: WsMessageEvent) => void;
            onError?: (data: { message: string; recoverable?: boolean }) => void;
            onDone?: (data: { duration_ms: number }) => void;
          };
          if (!d.payload) return;
          const wsInner = getWebSocket();
          wsInner.sendAction(d.payload, {
            onMessage: d.onMessage ?? (() => { /* no-op */ }),
            onError: d.onError ?? (() => { /* no-op */ }),
            onDone: d.onDone ?? (() => { /* no-op */ }),
          });
        }),
      );

      ws.connect();
    },

    /** Register an auth-failure callback (called by App bootstrap). */
    onAuthFailure(cb: () => void): void {
      this._onAuthFailure = cb;
    },

    // ── Turn actions ──────────────────────────────────────────────────────────

    /**
     * Main send orchestrator — port of chat.js sendMessage.
     *
     * Mid-ACT path: when a turn is already in-flight, appends the new text
     * to the existing user form, removes the ACT placeholder, and re-starts
     * the turn. The backend cancels the active turn, concatenates messages,
     * and starts a fresh turn.
     */
    async sendMessage(
      text: string,
      source: 'text' | 'voice' = 'text',
      files: File[] = [],
      previews: AttachmentPreview[] = [],
    ): Promise<void> {
      if (!text && !files.length) return;

      const convo = useConversationStore();
      const presence = usePresenceStore();

      if (this.isSending) {
        // Mid-ACT path (chat.js lines 143-153): append to existing user form,
        // remove old ACT cycle, restart turn.
        if (this._lastUserFormId != null) {
          const form = convo.forms.find((f) => f.id === this._lastUserFormId);
          if (form?.kind === 'user') {
            form.text += '\n\n' + text;
          }
        }
        if (this._activeActId != null) {
          const actIdx = convo.forms.findIndex((f) => f.id === this._activeActId);
          if (actIdx !== -1) convo.forms.splice(actIdx, 1);
          this._activeActId = null;
        }
        this._startTurn(text, source, false);
        return;
      }

      // Fresh path
      this.isSending = true;
      presence.setState('processing');

      const userFormId = convo.appendUser(text || '[File attached]', previews, {
        inWorkingMemory: true,
      });
      this._lastUserFormId = userFormId;
      this._lastUserText = text;

      this._startTurn(text || '[File attached]', source, false, files, previews);
    },

    /**
     * Wire and launch a turn — port of chat.js _startTurn.
     *
     * @param showUserBubble - true only for re-entries where sendMessage hasn't
     *   already appended the user form (e.g. mid-ACT path).
     */
    _startTurn(
      text: string,
      source: 'text' | 'voice',
      showUserBubble: boolean,
      files: File[] = [],
      previews: AttachmentPreview[] = [],
    ): void {
      const convo = useConversationStore();
      const presence = usePresenceStore();
      const ws = getWebSocket();

      if (showUserBubble) {
        const uid = convo.appendUser(text || '[File attached]', previews, {
          inWorkingMemory: true,
        });
        this._lastUserFormId = uid;
        this._lastUserText = text;
      }

      const actId = convo.appendAct();
      this._activeActId = actId;

      // Capture response data across onMessage / onDone callbacks
      let responseContent = '';
      let responseMeta: {
        content?: string;
        topic?: string;
        exchange_id?: string;
        mode?: string;
        confidence?: number;
        segments?: ConversationSegment[];
        timestamp?: string;
        duration_ms?: number;
      } = {};

      ws.send(
        text,
        source,
        {
          onStatus: (stage: string) => {
            presence.setState(stage);
          },

          onNarration: (data) => {
            presence.setState('narrating');
            const d = data as { text?: string; step?: number };
            convo.setActNarration(actId, d.text ?? '', d.step);
          },

          // CRITICAL CORRECTION: backend emits act_tool_start = { type, name, id, summary }.
          // Use d.id (NOT d.call_id) — verified against backend abilities/_dispatcher.py.
          onToolStart: (data) => {
            const d = data as { id?: string; name?: string; summary?: string };
            convo.appendToolPill(actId, d.id ?? '', d.name ?? '', d.summary);
          },

          // CRITICAL CORRECTION: backend emits act_tool_end = { type, name, id, ok }.
          // No duration field — pass ms=0 and let resolveToolPill compute client elapsed.
          onToolEnd: (data) => {
            const d = data as { id?: string; ok?: boolean };
            convo.resolveToolPill(d.id ?? '', 0, !!d.ok);
          },

          onMessage: (data) => {
            const d = data as WsMessageEvent & {
              content?: string;
              topic?: string;
              exchange_id?: string;
              mode?: string;
              confidence?: number;
              segments?: ConversationSegment[];
              timestamp?: string;
            };
            responseContent = d.content ?? '';
            responseMeta = {
              topic: d.topic,
              exchange_id: d.exchange_id,
              mode: d.mode ?? '',
              confidence: d.confidence ?? 0,
              segments: d.segments,
              timestamp: d.timestamp ?? '',
            };
            presence.setState('responding');
          },

          onError: (data) => {
            // Port of chat.js onError (lines 290-300):
            // Turn-level errors (provider failure, quota/429) are NOT auth events.
            // Only data.auth_failed triggers the login-redirect callback.
            convo.replaceActWithError(actId, data.message);
            const d = data as { auth_failed?: boolean };
            if (d.auth_failed) this._onAuthFailure?.();
          },

          onDone: (data) => {
            // Port of chat.js _finaliseTurn (lines 313-328)
            responseMeta.duration_ms = data.duration_ms;
            convo.replaceActWithResponse(actId, {
              content: responseContent,
              topic: responseMeta.topic,
              exchange_id: responseMeta.exchange_id,
              mode: responseMeta.mode,
              confidence: responseMeta.confidence,
              segments: responseMeta.segments,
              timestamp: responseMeta.timestamp,
              duration_ms: responseMeta.duration_ms,
            });
            // Port of app.js line 137: record ambient response timestamp.
            useAmbientSensor().recordResponse();
            this._activeActId = null;
            presence.setState('resting');
            this.isSending = false;
            // Signal feed component to force-scroll (A1 listens to this)
            document.dispatchEvent(new CustomEvent('session:turn-done'));

            // Background notification (chat.js _notifyBackgroundIfUnfocused)
            if (responseContent && !document.hasFocus()) {
              this._notifyBackground(responseContent);
            }
          },
        },
        files,
      );
    },

    /**
     * Stop + undo the active turn — port of chat.js requestStop.
     *
     * Removes the ACT placeholder and the last user form. Emits
     * 'session:turn-interrupted' so InputDock (Task A3) can restore
     * the textarea value.
     */
    async requestStop(): Promise<void> {
      const convo = useConversationStore();
      const presence = usePresenceStore();
      const ws = getWebSocket();

      // Remove ACT cycle
      if (this._activeActId != null) {
        const actIdx = convo.forms.findIndex((f) => f.id === this._activeActId);
        if (actIdx !== -1) convo.forms.splice(actIdx, 1);
        this._activeActId = null;
      }

      // Remove last user bubble and capture its LIVE text for restore.
      // Legacy chat.js requestStop reads textContent off the bubble, which in a
      // mid-ACT append holds the concatenated "A\n\nB" — not the original "A".
      // _lastUserText is only a fallback when the form can no longer be found.
      let restoredText = this._lastUserText;
      if (this._lastUserFormId != null) {
        const uidx = convo.forms.findIndex((f) => f.id === this._lastUserFormId);
        if (uidx !== -1) {
          const uform = convo.forms[uidx];
          if (uform.kind === 'user') restoredText = uform.text;
          convo.forms.splice(uidx, 1);
        }
        this._lastUserFormId = null;
      }
      this._lastUserText = '';

      // Abort WS callbacks so stale events are ignored
      ws.abort();

      this.isSending = false;
      presence.setState('resting');

      // Signal InputDock (Task A3) to restore the textarea
      document.dispatchEvent(
        new CustomEvent('session:turn-interrupted', { detail: { text: restoredText } }),
      );

      // Best-effort cancel to backend
      await this._postInterrupt();
    },

    /** POST /chat/interrupt — best-effort, never throws (port of chat.js _postInterrupt). */
    async _postInterrupt(): Promise<void> {
      try {
        const host = getHost();
        const base = host ? host.replace(/\/$/, '') : '';
        await fetch(base + '/chat/interrupt', {
          method: 'POST',
          credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
        });
      } catch {
        // Best-effort — swallow
      }
    },

    /**
     * Send an action payload — port of app.js chalie:action handler (lines 436-460).
     *
     * Creates an ACT cycle, blocks concurrent sends, calls ws.sendAction with
     * callbacks that resolve the ACT into a Chalie form or error.
     */
    async sendAction(payload: Record<string, unknown>): Promise<void> {
      if (this.isSending) return;

      const convo = useConversationStore();
      const presence = usePresenceStore();
      const ws = getWebSocket();

      this.isSending = true;
      const actId = convo.appendAct();
      this._activeActId = actId;

      ws.sendAction(payload, {
        onMessage: (data) => {
          const d = data as WsMessageEvent & {
            content?: string;
            mode?: string;
            confidence?: number;
          };
          convo.replaceActWithResponse(actId, {
            content: d.content ?? '',
            mode: d.mode ?? 'ACT',
            confidence: d.confidence ?? 0.95,
          });
        },
        onError: (data) => {
          convo.replaceActWithError(actId, data.message);
        },
        onDone: () => {
          this._activeActId = null;
          this.isSending = false;
          presence.setState('resting');
        },
      });
    },

    // ── History ───────────────────────────────────────────────────────────────

    /**
     * Load (or paginate) conversation history — port of chat.js loadRecentConversation.
     *
     * Safe to call multiple times; guards against concurrent loads and exhausted
     * history. On initial load (offset=0), appends turns + emits force-scroll signal.
     * On paginated loads (offset>0), prepends turns.
     */
    async loadRecentConversation(): Promise<void> {
      if (this.historyLoading || this.historyExhausted) return;
      this.historyLoading = true;

      const LIMIT = 12;
      const MAX_TURNS = 120;
      const convo = useConversationStore();

      try {
        const data = await conversation.recent(
          LIMIT,
          this.historyOffset > 0 ? this.historyOffset : undefined,
        );
        const messages = data.messages ?? [];

        if (messages.length === 0 && this.historyOffset === 0) {
          this.historyExhausted = true;
          return;
        }

        const isInitialLoad = this.historyOffset === 0;
        if (isInitialLoad) {
          convo.appendTurns(messages);
        } else {
          convo.prependTurns(messages);
        }

        this.historyOffset += messages.length;

        if (!data.has_more || this.historyOffset >= MAX_TURNS) {
          this.historyExhausted = true;
        }

        if (isInitialLoad && messages.length > 0) {
          // Signal feed component (Task A1) to force-scroll to bottom
          document.dispatchEvent(new CustomEvent('session:history-initial-loaded'));
        }
      } catch (err) {
        if (err instanceof AuthError) {
          this._onAuthFailure?.();
        } else {
          console.error('[Session] Failed to load conversation history:', err);
        }
      } finally {
        this.historyLoading = false;
      }
    },

    // ── Drift router ──────────────────────────────────────────────────────────

    /**
     * Route a drift push event — exact port of event_router.js._handleEvent.
     *
     * Routing order MUST match event_router.js:
     *   1. Simple content-free types (app_update / task / capability_alert / etc.)
     *   2. 'thought' (content, bypasses send-guard — always rendered)
     *   3. 'response' while sending → IGNORED
     *   4. Background notify
     *   5. 'notification'
     *   6. 'response' / 'escalation' / 'drift' → appendChalie
     */
    routeDrift(data: WsPushEvent): void {
      // Step 1
      if (this._routeSimpleEvent(data)) return;

      const content = (data as { content?: string }).content ?? '';
      if (!content) return;

      // Step 2: thought bypasses the send-guard
      if ((data.type as string) === 'thought') {
        const convo = useConversationStore();
        const meta = {
          topic: (data as { topic?: string }).topic,
          type: data.type,
          ts: new Date().toISOString(),
          mode: (data as { mode?: string }).mode ?? '',
          confidence: (data as { confidence?: number }).confidence ?? 0,
        };
        convo.appendChalie(content, meta);
        return;
      }

      // Step 3: 'response' while /chat is in-flight → ignore (event_router.js line 78)
      if ((data.type as string) === 'response' && this.isSending) return;

      // Step 4: background notify
      this._notifyBackground(content);

      // Step 5: notification — scheduler fired (reminder/task done). Port of
      // event_router.js._routeNotificationEvent → app.js onNotification (385-388):
      // chime UNCONDITIONALLY (no focus/permission gate) and refresh the task
      // strip. Distinct from the focus-gated background chime in step 4.
      if (data.type === 'notification') {
        const notifications = useNotificationsStore();
        const tasks = useTasksStore();
        notifications.chime();
        void tasks.loadActiveTasks();
        return;
      }

      // Step 6: response / escalation / drift
      this._renderContentEvent(data, content);
    },

    /**
     * Route content-free event types — port of event_router.js._routeSimpleEvent.
     * Returns true when handled.
     */
    _routeSimpleEvent(data: WsPushEvent): boolean {
      const tasks = useTasksStore();
      const permissions = usePermissionsStore();
      const notifications = useNotificationsStore();

      switch (data.type as string) {
        case 'app_update':
          // Update prompt (dormant — backend does not yet emit app_update).
          notifications.handleUpdate(data as unknown as UpdateState);
          return true;
        case 'task':
          tasks.applyDriftEvent(data);
          return true;
        case 'capability_alert':
          // No-op: dormant capability-alert channel (no legacy UI consumer).
          return true;
        case 'permission_request':
          permissions.enqueue(data);
          return true;
        case 'quick_tip':
          // Quick-tip card (dormant — backend does not yet emit quick_tip).
          notifications.handleTip(data as unknown as TipState);
          return true;
        case 'subagent_start':
          // Subagent lifecycle feeds the task drawer's subagent list.
          tasks.applyDriftEvent(data);
          return true;
        case 'subagent_end':
          tasks.applyDriftEvent(data);
          return true;
        default:
          return false;
      }
    },

    /**
     * Render a response/escalation/drift event into the conversation spine.
     * Port of event_router.js._renderContentEvent.
     * Escalation gets an `escalation: true` flag (CSS `--escalation` modifier).
     */
    _renderContentEvent(data: WsPushEvent, content: string): void {
      const convo = useConversationStore();
      const isEscalation = (data.type as string) === 'escalation';
      const meta = {
        topic: (data as { topic?: string }).topic,
        type: data.type,
        ts: new Date().toISOString(),
        mode: (data as { mode?: string }).mode ?? '',
        confidence: (data as { confidence?: number }).confidence ?? 0,
      };
      convo.appendChalie(content, meta, { escalation: isEscalation });
    },

    /**
     * Fire a background notification when the tab is not focused.
     * Port of event_router.js._notifyBackground + chat.js._notifyBackgroundIfUnfocused.
     *
     * Routes through notificationsStore.pushBackground so P1c can add system
     * notifications / sound without touching this store.
     */
    _notifyBackground(content: string): void {
      if (document.hasFocus()) return;
      const notifications = useNotificationsStore();
      // Plain text for the OS notification preview — the faithful markup_extract
      // equivalent (legacy chat.js/event_router.js use extractPlaintext): drops
      // <actions> button labels and substitutes <img alt>.
      const plain = extractText(content);
      if (plain) notifications.pushBackground(plain);
    },
  },
});

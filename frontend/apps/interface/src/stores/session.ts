/**
 * Session store — WS coordinator, turn state-machine, drift router.
 *
 * Single WS owner rule: ONLY this store may touch WebSocketService handlers
 * (send/sendAction/abort/onDrift/onAny/onConnect/onDisconnect/connect/ensureAlive).
 * Everything else goes through this store or the event bus.
 */
import { defineStore } from 'pinia';
import { getWebSocket, useConnectionStore, AuthError, api, HttpError } from '@chalie/shared';
import type { WsInboundEvent, WsPushEvent, WsMessageEvent } from '@chalie/shared';
import { on } from '../composables/useEventBus';
import { extractText } from '../composables/useMarkup';
import { conversation } from '../api/conversation';
import { moments } from '../api/moments';
import { showToast } from '../utils/toast';
import { useConversationStore } from './conversation';
import type { AttachmentPreview } from './conversation';
import type { ConversationSegment } from '../api/conversation';
import { useTasksStore } from './tasks';
import { useNotificationsStore } from './notifications';
import type { TipState, UpdateState } from './notifications';
import { usePermissionsStore } from './permissions';
import { useContextUsageStore } from './contextUsage';
import { useAmbientSensor } from '../composables/useAmbientSensor';

/** Guard: init() must be idempotent (HMR / Vue StrictMode). */
let _initialized = false;

/** Last pin-moment timestamp (ms) — 250ms debounce. */
let _pinDebounce = 0;

/** Unbind fns for event-bus listeners registered in init() (for future cleanup). */
const _busUnbinds: Array<() => void> = [];

export const useSessionStore = defineStore('session', {
  state: () => ({
    isSending: false,
    /** Id of the active ACT form while a turn is in-flight. */
    _activeActId: null as number | null,
    /** Id of the last user form (for mid-ACT restore on requestStop). */
    _lastUserFormId: null as number | null,
    /** Captured text from the last user turn (for requestStop restore). */
    _lastUserText: '',

    /** Turn-level provider/quota error, surfaced as a closable toast above the
     *  input dock. Null when there is nothing to show. */
    errorMessage: null as string | null,

    historyOffset: 0,
    historyLoading: false,
    historyExhausted: false,

    /** Registered auth-failure callback (set by App bootstrap). */
    _onAuthFailure: null as (() => void) | null,
  }),

  actions: {
    /** Wire the WebSocket singleton and connect. Idempotent (HMR / StrictMode). */
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
        // A mid-turn drop strands the live act group: the turn's done now lands
        // on the dead socket and is never resent, so collapse it here or the
        // "thinking…" spinner hangs forever.
        if (this._activeActId != null) {
          useConversationStore().resolveAct(this._activeActId);
          this._activeActId = null;
        }
      });

      ws.onDrift((data: WsPushEvent) => {
        this.routeDrift(data);
      });

      // onAny feeds the context-usage indicator — must never throw. refresh()
      // is coalesced inside the store, so a per-frame call is safe.
      ws.onAny((data: WsInboundEvent) => {
        try {
          void data;
          void contextUsage.refresh();
        } catch {
          /* never break the WS pipe */
        }
      });

      // Tab-refocus liveness check.
      globalThis.addEventListener('focus', () => ws.ensureAlive());

      // chalie:action — deterministic skill invocations. Registered inside init()
      // so the WS single-owner rule holds (only one listener ever bound).
      _busUnbinds.push(
        on('chalie:action', (payload) => {
          // Bus emits the full detail; legacy read e.detail.payload — support both.
          const p =
            (payload as { payload?: Record<string, unknown> }).payload ??
            (payload as Record<string, unknown>);
          void this.sendAction(p);
        }),
      );

      // chalie:pin-moment — Remember button: 250ms debounce, single POST
      // /moments, then a "Remembered" toast with an Undo action (POST
      // /moments/<id>/forget). No confirmation dialog — recall is a separate UI.
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
              showToast(
                res.duplicate ? 'Already remembered' : 'Remembered',
                transcriptId != null ? () => void moments.forget(transcriptId) : null,
              );
            })
            .catch((err: unknown) => {
              console.warn('[Session] pin moment failed:', err);
            });
        }),
      );

      // chalie:silent-action — rich-card interactions, no chat bubble. Caller
      // supplies optional onMessage/onError/onDone for optimistic card UI.
      _busUnbinds.push(
        on('chalie:silent-action', (detail) => {
          const d = detail as {
            payload?: Record<string, unknown>;
            onMessage?: (data: WsMessageEvent) => void;
            onError?: (data: { message: string; recoverable?: boolean }) => void;
            onDone?: (data: { duration_ms: number }) => void;
          };
          if (!d.payload) return;
          getWebSocket().sendAction(d.payload, {
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

    /**
     * Main send orchestrator. Mid-ACT path: when a turn is in-flight, append the
     * new text to the existing user form, drop the partial turn, and restart —
     * the backend cancels the active turn, concatenates, and starts fresh.
     */
    async sendMessage(
      text: string,
      source: 'text' | 'voice' = 'text',
      files: File[] = [],
      previews: AttachmentPreview[] = [],
    ): Promise<void> {
      if (!text && !files.length) return;

      const convo = useConversationStore();

      if (this.isSending) {
        if (this._lastUserFormId != null) {
          const uidx = convo.forms.findIndex((f) => f.id === this._lastUserFormId);
          if (uidx !== -1) {
            const uform = convo.forms[uidx];
            if (uform.kind === 'user') uform.text += '\n\n' + text;
            // Drop the partial turn rendering (interim bubbles + tool groups).
            convo.forms.splice(uidx + 1);
          }
        }
        this._activeActId = null;
        this._startTurn(text, source, false);
        return;
      }

      this.isSending = true;

      const userFormId = convo.appendUser(text || '[File attached]', previews, {
        inWorkingMemory: true,
      });
      this._lastUserFormId = userFormId;
      this._lastUserText = text;

      this._startTurn(text || '[File attached]', source, false, files, previews);
    },

    /**
     * Wire and launch a turn. `showUserBubble` is true only for re-entries where
     * sendMessage hasn't already appended the user form.
     */
    _startTurn(
      text: string,
      source: 'text' | 'voice',
      showUserBubble: boolean,
      files: File[] = [],
      previews: AttachmentPreview[] = [],
    ): void {
      const convo = useConversationStore();
      const ws = getWebSocket();

      if (showUserBubble) {
        const uid = convo.appendUser(text || '[File attached]', previews, {
          inWorkingMemory: true,
        });
        this._lastUserFormId = uid;
        this._lastUserText = text;
      }

      // Open the ACT group up-front: an empty live group is the "thinking…"
      // placeholder (logo + stop affordance). The step's first tool_start lands
      // its pill in place; a tool-free turn evicts the empty group on resolve.
      this._activeActId = convo.appendAct();

      // Capture the FINAL turn result across onMessage(final) / onDone — interim
      // steps render immediately and are never cached.
      let responseContent = '';
      let responseMeta: {
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
          // Use d.id (NOT d.call_id) — frame is act_tool_start { type,name,id,summary }.
          // Lazily open the step's tool group; its prose already landed via the
          // preceding interim message.
          onToolStart: (data) => {
            const d = data as { id?: string; name?: string; summary?: string };
            if (this._activeActId == null) this._activeActId = convo.appendAct();
            convo.appendToolPill(this._activeActId, d.id ?? '', d.name ?? '', d.summary);
          },

          // act_tool_end has no duration field — pass ms=0, resolveToolPill
          // computes client elapsed.
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

            if (d.interim) {
              // Interim prose: supersede the previous step (collapse its tool
              // group), then render this step's bubble; the next onToolStart
              // opens a fresh group beneath it.
              if (this._activeActId != null) {
                convo.resolveAct(this._activeActId);
                this._activeActId = null;
              }
              if (d.content) convo.appendChalie(d.content, { ts: d.timestamp ?? '' });
              return;
            }

            // Final result — cached, rendered on done so duration_ms lands on the
            // bubble (also carries turn-wide rich-media segments).
            responseContent = d.content ?? '';
            responseMeta = {
              topic: d.topic,
              exchange_id: d.exchange_id,
              mode: d.mode ?? '',
              confidence: d.confidence ?? 0,
              segments: d.segments,
              timestamp: d.timestamp ?? '',
            };
          },

          onError: (data) => {
            // Turn-level errors (provider failure, quota/429) are NOT auth events:
            // collapse the in-flight step and surface the error as its own form.
            // Only data.auth_failed redirects to login.
            if (this._activeActId != null) {
              convo.resolveAct(this._activeActId);
              this._activeActId = null;
            }
            this.errorMessage = data.message;
            const d = data as { auth_failed?: boolean };
            if (d.auth_failed) this._onAuthFailure?.();
          },

          onDone: (data) => {
            responseMeta.duration_ms = data.duration_ms;
            // Settle the final step before the reply lands beneath it: collapse
            // its tools, or evict the bare "thinking…" placeholder if none ran.
            if (this._activeActId != null) {
              convo.resolveAct(this._activeActId);
              this._activeActId = null;
            }
            if (responseContent || responseMeta.segments?.length) {
              convo.appendChalie(responseContent, {
                topic: responseMeta.topic,
                exchange_id: responseMeta.exchange_id,
                mode: responseMeta.mode,
                confidence: responseMeta.confidence,
                segments: responseMeta.segments,
                ts: responseMeta.timestamp,
                duration_ms: responseMeta.duration_ms,
              });
            }
            useAmbientSensor().recordResponse();
            this.isSending = false;
            document.dispatchEvent(new CustomEvent('session:turn-done'));

            if (responseContent && !document.hasFocus()) {
              this._notifyBackground(responseContent);
            }
          },
        },
        files,
      );
    },

    /**
     * Stop + undo the whole in-flight turn (user message + everything the chain
     * rendered after it). Emits 'session:turn-interrupted' so InputDock can
     * restore the textarea value.
     */
    async requestStop(): Promise<void> {
      const convo = useConversationStore();
      const ws = getWebSocket();

      // Capture the user bubble's LIVE text: a mid-ACT append holds the
      // concatenated "A\n\nB", not the original "A". _lastUserText is only a
      // fallback when the form is gone.
      let restoredText = this._lastUserText;
      if (this._lastUserFormId != null) {
        const uidx = convo.forms.findIndex((f) => f.id === this._lastUserFormId);
        if (uidx !== -1) {
          const uform = convo.forms[uidx];
          if (uform.kind === 'user') restoredText = uform.text;
          convo.forms.splice(uidx);
        }
        this._lastUserFormId = null;
      }
      this._activeActId = null;
      this._lastUserText = '';

      // Abort WS callbacks so stale events are ignored.
      ws.abort();

      this.isSending = false;

      document.dispatchEvent(
        new CustomEvent('session:turn-interrupted', { detail: { text: restoredText } }),
      );

      await this._postInterrupt();
    },

    /** POST /chat/interrupt via the central ApiClient (carries the bearer). */
    async _postInterrupt(): Promise<void> {
      try {
        await api.post('/chat/interrupt', {});
      } catch (err) {
        if (err instanceof AuthError) {
          this._onAuthFailure?.();
          return;
        }
        const msg = err instanceof HttpError ? err.error ?? `HTTP ${err.status}` : 'Stop signal failed';
        console.error('[Session] /chat/interrupt failed:', err);
        showToast(msg);
      }
    },

    /**
     * Create an ACT cycle, block concurrent sends, and call ws.sendAction with
     * callbacks that resolve the ACT into a Chalie form or error.
     */
    async sendAction(payload: Record<string, unknown>): Promise<void> {
      if (this.isSending) return;

      const convo = useConversationStore();
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
          convo.resolveAct(actId);
          this.errorMessage = data.message;
        },
        onDone: () => {
          this._activeActId = null;
          this.isSending = false;
        },
      });
    },

    /**
     * Load (or paginate) conversation history. Guards against concurrent loads
     * and exhaustion. Initial load (offset=0) appends turns + emits force-scroll;
     * paginated loads (offset>0) prepend.
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

        this.historyOffset += data.turns_returned;

        if (!data.has_more || this.historyOffset >= MAX_TURNS) {
          this.historyExhausted = true;
        }

        if (isInitialLoad && messages.length > 0) {
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

    /**
     * Route a drift push event. Routing order is load-bearing:
     *   1. Simple content-free types  2. 'thought' (bypasses send-guard)
     *   3. 'response' while sending → IGNORED  4. Background notify
     *   5. 'notification'  6. 'response' / 'escalation' / 'drift' → appendChalie
     */
    routeDrift(data: WsPushEvent): void {
      if (this._routeSimpleEvent(data)) return;

      const content = (data as { content?: string }).content ?? '';
      if (!content) return;

      // Multi-surface sync — user echo. Own-surface echoes were dropped in
      // WebSocketService (by echo_id), so this came from a DIFFERENT surface and
      // has no bubble yet; render it so all surfaces match.
      if ((data.type as string) === 'user_message') {
        useConversationStore().appendUser(content, [], { inWorkingMemory: true });
        return;
      }

      // Multi-surface sync — assistant reply (rule 2 of the echo+render model).
      // The sending surface gets this via chat callbacks (never here); every
      // OTHER surface lands here and renders a plain Chalie bubble.
      if ((data.type as string) === 'message') {
        const d = data as {
          topic?: string;
          exchange_id?: string;
          mode?: string;
          confidence?: number;
          segments?: ConversationSegment[];
          timestamp?: string;
        };
        this._notifyBackground(content);
        useConversationStore().appendChalie(content, {
          topic: d.topic,
          exchange_id: d.exchange_id,
          mode: d.mode ?? '',
          confidence: d.confidence ?? 0,
          segments: d.segments,
          ts: d.timestamp || new Date().toISOString(),
          type: 'message',
        });
        return;
      }

      // Step 2: thought bypasses the send-guard.
      if ((data.type as string) === 'thought') {
        useConversationStore().appendChalie(content, {
          topic: (data as { topic?: string }).topic,
          type: data.type,
          ts: new Date().toISOString(),
          mode: (data as { mode?: string }).mode ?? '',
          confidence: (data as { confidence?: number }).confidence ?? 0,
        });
        return;
      }

      // Step 3: 'response' while /chat is in-flight → ignore.
      if ((data.type as string) === 'response' && this.isSending) return;

      // Step 4: background notify.
      this._notifyBackground(content);

      // Step 5: notification — scheduler fired (reminder/task done): chime
      // UNCONDITIONALLY (no focus/permission gate, unlike step 4) and refresh
      // the task strip.
      if (data.type === 'notification') {
        useNotificationsStore().chime();
        void useTasksStore().loadActiveTasks();
        return;
      }

      // Step 6: response / escalation / drift.
      this._renderContentEvent(data, content);
    },

    /** Route content-free event types; returns true when handled. */
    _routeSimpleEvent(data: WsPushEvent): boolean {
      switch (data.type as string) {
        case 'app_update':
          useNotificationsStore().handleUpdate(data as unknown as UpdateState);
          return true;
        // task + subagent lifecycle both feed the task drawer.
        case 'task':
        case 'subagent_start':
        case 'subagent_end':
          useTasksStore().applyDriftEvent(data);
          return true;
        case 'capability_alert':
          // No-op: dormant channel (no UI consumer).
          return true;
        case 'permission_request':
          usePermissionsStore().enqueue(data);
          return true;
        case 'quick_tip':
          useNotificationsStore().handleTip(data as unknown as TipState);
          return true;
        case 'provider_retry':
          showToast((data as { message?: string }).message ?? 'The AI provider had a problem — retrying…');
          return true;
        default:
          return false;
      }
    },

    /** Escalation gets an `escalation: true` flag (CSS `--escalation` modifier). */
    _renderContentEvent(data: WsPushEvent, content: string): void {
      useConversationStore().appendChalie(
        content,
        {
          topic: (data as { topic?: string }).topic,
          type: data.type,
          ts: new Date().toISOString(),
          mode: (data as { mode?: string }).mode ?? '',
          confidence: (data as { confidence?: number }).confidence ?? 0,
        },
        { escalation: (data.type as string) === 'escalation' },
      );
    },

    /** Fire a background notification when the tab is not focused. */
    _notifyBackground(content: string): void {
      if (document.hasFocus()) return;
      // Plain text for the OS preview — drops <actions> labels, substitutes <img alt>.
      const plain = extractText(content);
      if (plain) useNotificationsStore().pushBackground(plain);
    },
  },
});

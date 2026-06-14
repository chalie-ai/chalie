/**
 * Conversation spine store — port of renderer.js into reactive Pinia state.
 *
 * Maintains an ordered list of `ConversationForm` items (the "forms" array).
 * Each item is a discriminated union: user / chalie / act / error.
 * Components render this list; this store owns all mutations.
 *
 * Port sources:
 *   - renderer.js: createActCycle, appendUserForm, appendChalieForm,
 *     replaceActWithResponse, replaceActWithError, appendToolPill,
 *     resolveToolPill (incl. 150ms min-display), setActNarrative,
 *     prependUserForm, prependChalieForm
 *   - chat.js: _appendMessage, _prependMessage, _attachmentsFor, _finaliseTurn
 */
import { defineStore } from 'pinia';
import type { ConversationAttachment, ConversationMessage, ConversationSegment } from '../api/conversation';

// ── Type definitions ──────────────────────────────────────────────────────────

export interface AttachmentPreview {
  filename: string;
  objectUrl: string | null;
  isImage: boolean;
}

export interface ChalieMeta {
  topic?: string;
  exchange_id?: string;
  mode?: string;
  confidence?: number;
  segments?: ConversationSegment[];
  ts?: string;
  duration_ms?: number;
  /** Event type from drift (thought, response, escalation, drift). */
  type?: string;
}

export interface ToolPill {
  id: string;
  name: string;
  summary: string | undefined;
  startedAt: number;
  ms?: number;
  ok?: boolean;
  resolved: boolean;
}

export interface UserForm {
  kind: 'user';
  id: number;
  text: string;
  attachments?: AttachmentPreview[];
  /** Whether the turn is within working memory (faded if false). */
  inWorkingMemory?: boolean;
}

export interface ChalieForm {
  kind: 'chalie';
  id: number;
  text?: string;
  segments?: ConversationSegment[];
  meta: ChalieMeta;
  escalation?: boolean;
  inWorkingMemory?: boolean;
}

export interface ActForm {
  kind: 'act';
  id: number;
  narration: string;
  narrationStep?: number;
  tools: ToolPill[];
}

export interface ErrorForm {
  kind: 'error';
  id: number;
  message: string;
}

export type ConversationForm = UserForm | ChalieForm | ActForm | ErrorForm;

// ── Module-scope monotonic id counter ─────────────────────────────────────────
let _nextId = 1;
function nextId(): number {
  return _nextId++;
}

// ── Store ─────────────────────────────────────────────────────────────────────

export const useConversationStore = defineStore('conversation', {
  state: () => ({
    forms: [] as ConversationForm[],
  }),

  getters: {
    /** Id of the most recent act form, or null. */
    activeActFormId(state): number | null {
      for (let i = state.forms.length - 1; i >= 0; i--) {
        const f = state.forms[i];
        if (f.kind === 'act') return f.id;
      }
      return null;
    },
    /** Id of the most recent chalie form, or null. */
    activeChalieFormId(state): number | null {
      for (let i = state.forms.length - 1; i >= 0; i--) {
        const f = state.forms[i];
        if (f.kind === 'chalie') return f.id;
      }
      return null;
    },
  },

  actions: {
    // ── Append / create ──────────────────────────────────────────────────────

    appendUser(
      text: string,
      attachments?: AttachmentPreview[],
      opts?: { inWorkingMemory?: boolean },
    ): number {
      const id = nextId();
      const form: UserForm = {
        kind: 'user',
        id,
        text,
        attachments: attachments ?? [],
        inWorkingMemory: opts?.inWorkingMemory ?? true,
      };
      this.forms.push(form);
      return id;
    },

    /**
     * Append a Chalie speech form.
     * When `meta.segments` is set, the form carries segments; otherwise `text`.
     * Returns the new form's id.
     */
    appendChalie(
      textOrContent: string,
      meta: ChalieMeta,
      opts?: { escalation?: boolean; inWorkingMemory?: boolean },
    ): number {
      const id = nextId();
      const form: ChalieForm = {
        kind: 'chalie',
        id,
        text: meta.segments ? undefined : textOrContent,
        segments: meta.segments,
        meta,
        escalation: opts?.escalation ?? false,
        inWorkingMemory: opts?.inWorkingMemory ?? true,
      };
      this.forms.push(form);
      return id;
    },

    /** Create a new ACT placeholder and return its id. */
    appendAct(): number {
      const id = nextId();
      const form: ActForm = {
        kind: 'act',
        id,
        narration: '',
        tools: [],
      };
      this.forms.push(form);
      return id;
    },

    appendError(message: string): number {
      const id = nextId();
      this.forms.push({ kind: 'error', id, message });
      return id;
    },

    // ── ACT-cycle mutations ──────────────────────────────────────────────────

    setActNarration(actId: number, text: string, step?: number): void {
      const form = this._findAct(actId);
      if (!form) return;
      form.narration = text;
      if (step != null) form.narrationStep = step;
    },

    appendToolPill(actId: number, id: string, name: string, summary?: string): void {
      const form = this._findAct(actId);
      if (!form) return;
      if (!id) return; // guard matches renderer.js
      const pill: ToolPill = {
        id,
        name,
        summary,
        startedAt: Date.now(),
        resolved: false,
      };
      form.tools.push(pill);
    },

    /**
     * Resolve a tool pill — port of renderer.js resolveToolPill.
     *
     * Enforces a 150ms minimum visible duration. When `ms` is 0, falls back to
     * client-measured elapsed (from pill.startedAt).
     */
    resolveToolPill(pillId: string, ms: number, ok: boolean): void {
      if (!pillId) return;
      // Find the pill across all act forms
      for (const form of this.forms) {
        if (form.kind !== 'act') continue;
        const pill = form.tools.find((t) => t.id === pillId);
        if (!pill) continue;

        const elapsed = pill.startedAt ? Date.now() - pill.startedAt : 200;
        const wait = Math.max(0, 150 - elapsed);
        // Client-measured elapsed fallback: if ms==0, compute from startedAt
        const effectiveMs = ms > 0 ? ms : (pill.startedAt ? Date.now() - pill.startedAt : 0);

        setTimeout(() => {
          // Re-locate: forms array may have shifted
          for (const f of this.forms) {
            if (f.kind !== 'act') continue;
            const p = f.tools.find((t) => t.id === pillId);
            if (p) {
              p.ms = effectiveMs;
              p.ok = ok;
              p.resolved = true;
              break;
            }
          }
        }, wait);
        break;
      }
    },

    // ── ACT → response / error swap ──────────────────────────────────────────

    /**
     * Replace an ACT form with the final Chalie response.
     * Port of renderer.js replaceActWithResponse + chat.js _finaliseTurn.
     *
     * If the WS message carries no content, the ACT form is simply removed
     * (port of _finaliseTurn's no-content branch).
     */
    replaceActWithResponse(
      actId: number,
      data: {
        content?: string;
        topic?: string;
        exchange_id?: string;
        mode?: string;
        confidence?: number;
        segments?: ConversationSegment[];
        timestamp?: string;
        duration_ms?: number;
      },
    ): number | null {
      const idx = this.forms.findIndex((f) => f.id === actId);
      if (idx === -1) return null;

      const content = data.content ?? '';
      if (!content && !data.segments?.length) {
        // No content — remove the act placeholder
        this.forms.splice(idx, 1);
        return null;
      }

      const meta: ChalieMeta = {
        topic: data.topic,
        exchange_id: data.exchange_id,
        mode: data.mode ?? '',
        confidence: data.confidence ?? 0,
        segments: data.segments ?? undefined,
        ts: data.timestamp ?? '',
        duration_ms: data.duration_ms,
      };

      const id = nextId();
      const chalie: ChalieForm = {
        kind: 'chalie',
        id,
        text: meta.segments ? undefined : content,
        segments: meta.segments,
        meta,
        inWorkingMemory: true,
      };
      this.forms.splice(idx, 1, chalie);
      return id;
    },

    /**
     * Replace an ACT form with an error speech-form.
     * Port of renderer.js replaceActWithError.
     */
    replaceActWithError(actId: number, message: string): void {
      const idx = this.forms.findIndex((f) => f.id === actId);
      if (idx === -1) return;
      const id = nextId();
      this.forms.splice(idx, 1, { kind: 'error', id, message });
    },

    // ── History pagination ───────────────────────────────────────────────────

    /**
     * Append history turns to the END of the spine (initial load).
     * Port of chat.js _appendMessage.
     */
    appendTurns(messages: ConversationMessage[]): void {
      for (const msg of messages) {
        this._appendMessage(msg, true);
      }
    },

    /**
     * Prepend history turns to the START of the spine (scroll-up pagination).
     * Oldest-first: messages are iterated in reverse so indices stay stable.
     * Port of chat.js _prependMessage.
     */
    prependTurns(messages: ConversationMessage[]): void {
      for (let i = messages.length - 1; i >= 0; i--) {
        this._prependMessage(messages[i], false);
      }
    },

    // ── Internal helpers ─────────────────────────────────────────────────────

    _findAct(actId: number): ActForm | undefined {
      const form = this.forms.find((f) => f.id === actId);
      return form?.kind === 'act' ? form : undefined;
    },

    _attachmentsFor(msg: ConversationMessage): AttachmentPreview[] {
      return (msg.attachments ?? []).map((a: ConversationAttachment) => ({
        filename: a.filename,
        objectUrl: a.url,
        isImage: a.is_image,
      }));
    },

    _appendMessage(msg: ConversationMessage, inWorkingMemory: boolean): void {
      if (msg.role === 'user') {
        const attachments = this._attachmentsFor(msg);
        this.appendUser(msg.content, attachments, { inWorkingMemory });
      } else if (msg.content || msg.segments?.length) {
        const meta: ChalieMeta = { ts: msg.timestamp };
        if (msg.segments) meta.segments = msg.segments;
        const id = nextId();
        this.forms.push({
          kind: 'chalie',
          id,
          text: msg.segments ? undefined : msg.content || '',
          segments: msg.segments,
          meta,
          inWorkingMemory,
        });
      }
    },

    _prependMessage(msg: ConversationMessage, inWorkingMemory: boolean): void {
      if (msg.role === 'user') {
        const attachments = this._attachmentsFor(msg);
        const id = nextId();
        this.forms.unshift({
          kind: 'user',
          id,
          text: msg.content,
          attachments,
          inWorkingMemory,
        });
      } else if (msg.content || msg.segments?.length) {
        const meta: ChalieMeta = { ts: msg.timestamp };
        if (msg.segments) meta.segments = msg.segments;
        const id = nextId();
        this.forms.unshift({
          kind: 'chalie',
          id,
          text: msg.segments ? undefined : msg.content || '',
          segments: msg.segments,
          meta,
          inWorkingMemory,
        });
      }
    },
  },
});

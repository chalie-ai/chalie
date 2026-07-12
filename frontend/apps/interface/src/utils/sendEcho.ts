/**
 * sendEcho — transient, DOM-only "I just sent this" preview of a user's own
 * submitted text.
 *
 * Today, the user bubble first mounts only after POST → WS `turn_execution`
 * 'working' frame → REST refetch → `upsertTurnToSurfaces`. When that frame is
 * slow, coalesced, or dropped, the first paint after submit is empty. This
 * module mounts a lightweight echo synchronously at send time so the dock
 * never looks like the message vanished — and clears it the moment the REAL
 * content lands (via turnDom's `onTurnLanded` hook), the dispatch that
 * produced it fails, or it's interrupted.
 *
 * Governing constraint (DOM is the render substrate, REST is
 * the sole data source, client state is visual-only): the echo is NEVER
 * registered as a turn (no `data-turn-id`/`data-type`), never read back as
 * data, and invisible to `isSurfaceBusy` (no `data-working`). It reuses
 * `UserBubble.vue` directly for its content so the text/attachment styling
 * (global, unscoped — see conversation.scss) is pixel-identical to the real
 * bubble; the surrounding avatar-gutter row layout lives in TurnView.vue's
 * OWN scoped stylesheet, unreachable from an imperative mount outside its
 * component tree, so that offset is reproduced here via the same global
 * tokens TurnView itself uses.
 */
import { createVNode, render } from 'vue';
import type { ConversationMessage } from '../api/conversation';
import UserBubble from '../components/conversation/UserBubble.vue';
import { onTurnLanded, resolveScopeContainer } from './turnDom';

interface MountedEcho {
  host: HTMLElement;
}

/** One mounted echo per scope key — see `scopeKey`. */
const _echoes = new Map<string, MountedEcho>();

/** Scope identity: the spine (keyed by type alone — there is only ever one
 *  spine) or a specific thread reply (keyed by type + its root turn_id). */
function scopeKey(threadId: number | null, type: string): string {
  return threadId == null ? `spine:${type}` : `${type}:${threadId}`;
}

// Mirrors TurnView's `.msg-row--lead` (new-speaker spacing) plus its
// `.msg-row__gutter` + gap offset (avatar width + 18px) via the SAME global
// tokens TurnView itself reads — see module doc comment for why the scoped
// rule itself can't be reused here.
const ECHO_ROW_STYLE = [
  'width: 100%',
  'max-width: var(--dock-width)',
  'margin: 30px auto 0',
  'padding-left: calc(var(--avatar-size) + 18px)',
].join('; ');

// Registered lazily, on first real use, rather than at this module's own
// top level: turnDom.ts's default surface component (TurnView) transitively
// imports stores/session.ts (TurnView → ActCycle → useSessionStore), which
// imports THIS module — a genuine module cycle. Calling `onTurnLanded` eagerly
// at load time would run turnDom.ts's `_turnLandedHook = hook` assignment
// WHILE turnDom.ts's own top-level `let _turnLandedHook` hasn't executed yet
// (TDZ — `ReferenceError: Cannot access '_turnLandedHook' before
// initialization`). Deferring to first use, well after the whole module graph
// has finished loading, sidesteps that — the same reason session.ts's own
// `registerSessionHooks` call lives inside its `init()` action rather than at
// session.ts's module top level.
let _registered = false;
function _ensureRegistered(): void {
  if (_registered) return;
  _registered = true;
  onTurnLanded((turnId, landedType, isNewTopLevelTurn) => {
    if (isNewTopLevelTurn) clearSendEcho(null, landedType);
    clearSendEcho(turnId, landedType);
  });
}

/**
 * Mount (or replace) the transient echo for a send scope into the surface
 * container that will host the upcoming turn — the spine for `threadId`
 * null, the open thread panel for a reply. No-op when no surface currently
 * claims the scope (e.g. a reply fired while its panel isn't registered).
 * One echo per scope: calling this again for the same scope replaces the
 * previous element rather than stacking.
 */
export function mountSendEcho(text: string, threadId: number | null, type: string): void {
  _ensureRegistered();
  const container = resolveScopeContainer(threadId, type);
  if (!container) return;

  const key = scopeKey(threadId, type);
  clearSendEcho(threadId, type);

  const host = document.createElement('div');
  host.dataset.sendEcho = key;
  host.style.cssText = ECHO_ROW_STYLE;
  container.appendChild(host);

  const message: ConversationMessage = {
    id: `echo-${key}`,
    role: 'user',
    content: text,
    timestamp: new Date().toISOString(),
    turn_id: null,
  };
  render(createVNode(UserBubble, { message }), host);

  _echoes.set(key, { host });
}

/** Unmount and clean up the echo for a send scope. No-op when none is
 *  mounted (including a scope that was never mounted at all). */
export function clearSendEcho(threadId: number | null, type: string): void {
  const key = scopeKey(threadId, type);
  const echo = _echoes.get(key);
  if (!echo) return;
  render(null, echo.host);
  echo.host.remove();
  _echoes.delete(key);
}

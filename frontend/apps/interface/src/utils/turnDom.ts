import { createVNode, render } from 'vue';
import type { App, AppContext, Component } from 'vue';
import TurnView from '../components/conversation/TurnView.vue';
import type { ConversationTurnBlock } from '../api/conversation';
import { clearLiveTurnsForToolCallsResolved } from './liveActTrail';

let appContext: AppContext | null = null;

/**
 * Persist the created app's context so vnodes mounted outside the component
 * tree (e.g. by WS handlers) can resolve Pinia stores and provide/inject.
 */
export function registerAppContext(app: App): void {
  appContext = app._context;
}

/** Look up a turn element by its data-keyed identity, optionally scoped to
 *  one surface's container (multiple surfaces can render the same turn_id —
 *  e.g. the spine AND the thread panel — so an unscoped lookup would be
 *  ambiguous). */
export function getTurnEl(
  turnId: number,
  type: string,
  container?: ParentNode,
): HTMLElement | null {
  const scope: ParentNode = container ?? document;
  return scope.querySelector<HTMLElement>(`[data-turn-id="${turnId}"][data-type="${type}"]`);
}

/** The type stamped on the first rendered copy of a turn, UNSCOPED by type —
 *  deliberately, since resolving the type is exactly the point (a turn_id is
 *  only unique per type, so scanning by type would require already knowing
 *  the answer). Used to recover a WS frame's identity when the frame itself
 *  omits `type` rather than guessing it. Null when no copy is rendered
 *  anywhere yet. */
export function findTurnType(turnId: number): string | null {
  return document.querySelector<HTMLElement>(`[data-turn-id="${turnId}"]`)?.dataset.type ?? null;
}

/** Every rendered copy of a turn, across ALL surfaces — used by effects that
 *  must reach every visible copy of a turn (working/done flags, removal). */
function getAllTurnEls(turnId: number, type: string): HTMLElement[] {
  return Array.from(
    document.querySelectorAll<HTMLElement>(`[data-turn-id="${turnId}"][data-type="${type}"]`),
  );
}

// ── Live-working bookkeeping (mount-time race fix) ───────────────────────────
//
// A WS 'working' frame for a brand-new turn always arrives BEFORE its first
// fetch-driven upsertTurn/mount. setTurnWorking's early-return (preserved
// below for turnDom.spec.ts's asserted "silent no-op" behaviour when no
// element exists yet) would otherwise silently drop that signal, leaving a
// freshly-mounted turn with no data-working attribute until the NEXT signal
// arrives. `_liveWorking` records the signal regardless of whether an
// element exists yet; `upsertTurn` consults it at mount/patch time so a live
// 'true' always wins over a stale fetched snapshot's `false` (a fetch can
// only race BEHIND the WS frame that triggered it, never ahead of it — a
// `false` snapshot never legitimately overrides an already-recorded live
// 'true'). The reverse isn't specially protected: once a settle signal
// clears the set, a very late in-flight fetch resolving afterwards reads the
// already-settled `block.working` correctly regardless.
const _liveWorking = new Set<string>();

function workingKey(turnId: number, type: string): string {
  return `${type}:${turnId}`;
}

/** Snapshot of every `type:turnId` key currently recorded as live-working —
 *  includes turns whose 'working' frame arrived before any element rendered.
 *  Consumed by session's disconnect handler, which must remember what was in
 *  flight BEFORE clearing the visual state it would otherwise scan for on
 *  reconnect. */
export function liveWorkingKeys(): string[] {
  return Array.from(_liveWorking);
}

/** True if any scheduled turn is currently recorded as done (D16 "settled
 *  unseen") — used by the nav to surface unseen scheduled content as a
 *  visual cue. */
export function hasDoneScheduled(): boolean {
  for (const key of _liveDone) {
    if (key.startsWith('scheduled:')) return true;
  }
  return false;
}

/** True if the DOM contract currently considers this turn in flight — a live
 *  'working' signal recorded (even before any element existed) OR an actual
 *  rendered copy still carrying `data-working`. */
export function isTurnWorking(turnId: number, type: string): boolean {
  return (
    _liveWorking.has(workingKey(turnId, type)) ||
    getAllTurnEls(turnId, type).some((el) => el.hasAttribute('data-working'))
  );
}

/** The spine's registered surface id — shared between ConversationFeed.vue
 *  (which registers it) and any code that needs to ask "is the main spine
 *  busy" without a stable turn_id to key off (D3/D5). */
export const SPINE_SURFACE_ID = 'spine';

// ── Lanes — the spine is a thread with a fixed identity ──────────────────────
//
// Every lane runs independently: work in one never blocks a send in another,
// so five threads can be replied to in parallel while the spine stays free. A
// lane is identified by the same `(type, turn_id)` pair a turn is, which makes
// the busy check one keyed lookup for every lane alike. The spine has no
// turn_id of its own — it allocates a NEW one per send — so it takes the
// reserved pair below.
//
// A turn element claims its own thread lane (`data-lane-type` /
// `data-lane-turn-id`) only once the turn is FORKED: a forked turn's work
// belongs to its thread, an unforked one's belongs to the spine. Absence
// therefore means "spine lane", which also covers the ACT-cycle markers that
// carry no turn identity at all. Forked-ness is re-derived from the block on
// every upsert, so the claim survives the re-mount a mid-turn refetch performs
// (anything recorded only in JS, or stamped once at settle time, would not).
export const SPINE_LANE_TYPE = 'main';
export const SPINE_LANE_TURN_ID = -1;

/** True if this lane has work in flight — the single busy authority behind
 *  both the send/queue gate and the dock's busy visual. A thread keys off its
 *  own turn; the spine keys off the reserved pair and matches only working
 *  elements no thread lane has claimed. */
export function isLaneWorking(laneType: string, laneTurnId: number): boolean {
  if (laneType !== SPINE_LANE_TYPE || laneTurnId !== SPINE_LANE_TURN_ID) {
    return isTurnWorking(laneTurnId, laneType);
  }
  const spine = _surfaces.get(SPINE_SURFACE_ID);
  if (!spine) return false;
  return spine.container.querySelector('[data-working]:not([data-lane-turn-id])') != null;
}

/** Claim a turn for its own thread lane on every rendered copy. Called when a
 *  reply is sent into it: the reply proves the fork immediately, whereas
 *  `stampWorking` can only re-derive it once the refetch carrying the new
 *  thread_message lands — a round-trip during which the spine would otherwise
 *  read the reply's work as its own. */
export function markThreadLane(turnId: number, type: string): void {
  for (const el of getAllTurnEls(turnId, type)) {
    el.setAttribute('data-lane-type', type);
    el.setAttribute('data-lane-turn-id', String(turnId));
  }
}

/** `type:turnId` keys marked done (D16, "settled unseen") — the `done`
 *  counterpart to `_liveWorking`, and needed for the same reason: a settle
 *  frame routinely arrives for a turn NO surface renders (a scheduled turn
 *  finishing with the dock closed is the common case, not the edge), where
 *  `setTurnDone` would stamp zero elements and the state would be lost. The
 *  record is authoritative; the DOM attribute is its rendered projection. */
const _liveDone = new Set<string>();

/** True if this turn is marked done (D16) — live record first (covers turns
 *  with no rendered copy), then any rendered copy's `data-done` attribute. */
export function isTurnDone(turnId: number, type: string): boolean {
  return (
    _liveDone.has(workingKey(turnId, type)) ||
    getAllTurnEls(turnId, type).some((el) => el.hasAttribute('data-done'))
  );
}

/** A registered surface's own container element, or null if unregistered —
 *  used by `utils/threadActivity.ts` to scope its DOM scan to the spine only. */
export function getSurfaceContainer(surfaceId: string): HTMLElement | null {
  return _surfaces.get(surfaceId)?.container ?? null;
}

/** The raw text of the LAST user row inside a given turn host — used by
 *  `session.requestStop` (D6) to restore an interrupted turn's draft without
 *  a lane record. Deliberately scoped to a CALLER-SUPPLIED host rather than
 *  looked up document-wide: the same turn_id can render two different row
 *  sets on different surfaces (the spine hides thread-reply rows a
 *  full-thread panel copy would show), so only the copy the stop actually
 *  originated from is unambiguous. */
export function lastUserText(turnHost: ParentNode): string {
  const rows = turnHost.querySelectorAll<HTMLElement>('[data-user-text]');
  const last = rows[rows.length - 1];
  return last?.dataset.userText ?? '';
}

// ── D14 — surface registry ───────────────────────────────────────────────────

/**
 * A registered render target for a ConfigType — the spine mounts SpineTurn,
 * the thread panel mounts TurnView directly. `props` are extra STATIC props
 * merged with `{ block, type }` on every mount (event handlers included —
 * Vue resolves an `onX` prop to the matching `emit('x', ...)`). `accepts`
 * filters which turn_ids render here (default: every turn of this type).
 */
export interface Surface {
  id: string;
  type: string;
  container: HTMLElement;
  component: Component;
  props?: Record<string, unknown>;
  accepts?: (turnId: number) => boolean;
}

const _surfaces = new Map<string, Surface>();

export function registerSurface(surface: Surface): void {
  _surfaces.set(surface.id, surface);
}

export function unregisterSurface(id: string): void {
  _surfaces.delete(id);
}

/** Unmount and remove every rendered turn host from a surface's container —
 *  used when a surface tears down (e.g. the thread panel closes). */
export function clearSurfaceContainer(container: HTMLElement): void {
  for (const host of Array.from(container.children)) {
    render(null, host as HTMLElement);
    host.remove();
  }
}

/** Resolve the surface container that will host the UPCOMING turn for a send
 *  scope — the spine's own container for a new top-level send (`threadId`
 *  null), or the specific non-spine surface currently accepting that turn_id
 *  for a thread reply (the open thread panel). `getSurfaceContainer` above
 *  resolves by surface id; this resolves by SCOPE instead, since callers here
 *  (`utils/sendEcho.ts`) know only the scope, not a surface id.
 *  Null when nothing currently claims the scope — the caller no-ops. */
export function resolveScopeContainer(threadId: number | null, type: string): HTMLElement | null {
  if (threadId == null) {
    const spine = _surfaces.get(SPINE_SURFACE_ID);
    return spine && spine.type === type ? spine.container : null;
  }
  for (const surface of _surfaces.values()) {
    if (surface.id === SPINE_SURFACE_ID) continue;
    if (surface.type !== type) continue;
    if (surface.accepts && !surface.accepts(threadId)) continue;
    return surface.container;
  }
  return null;
}

// ── D13 — version guard ──────────────────────────────────────────────────────

export interface UpsertOptions {
  /** Bypass the monotonic version guard. Two callers, both terminal: the
   *  post-cancel reconcile, whose server-stripped block can legitimately
   *  shrink (see driftDispatcher's cancelled-branch comment), and the
   *  completed/crashed settle, where the turn is immutable and the settled
   *  block must land (see driftDispatcher's terminal-branch comment). */
  force?: boolean;
}

function blockVersion(block: ConversationTurnBlock): number {
  let v = 0;
  for (const m of block.messages) {
    const n = Number.parseInt(m.id, 10);
    if (n > v) v = n;
  }
  return v;
}

/** Insert or replace a turn block inside `container`, preserving numeric
 *  order. Guarded by a monotonic `data-version` stamp on the host element
 *  (the block's highest message id) unless `options.force` is set — an
 *  incoming block whose version is STRICTLY lower than what's already
 *  rendered is dropped; an equal version re-applies (idempotent self-heal). */
export function upsertTurn(
  block: ConversationTurnBlock,
  type: string,
  container: HTMLElement,
  component: Component = TurnView,
  extraProps: Record<string, unknown> = {},
  options: UpsertOptions = {},
): HTMLElement | null {
  const version = blockVersion(block);
  const existing = getTurnEl(block.turn_id, type, container);

  if (existing) {
    const host = existing.parentElement!;
    const currentVersion = Number.parseInt(host.dataset.version ?? '-1', 10);
    if (!options.force && currentVersion > version) return null;
    host.dataset.version = String(version);
    stampDay(host, block);
    mount(host, component, block, type, extraProps);
    stampWorking(host, block, type);
    notifyUpserted(block.turn_id, type);
    return host;
  }

  const host = document.createElement('div');
  host.dataset.version = String(version);
  stampDay(host, block);
  insertInOrder(container, host, block.turn_id);
  mount(host, component, block, type, extraProps);
  stampWorking(host, block, type);
  notifyUpserted(block.turn_id, type);
  return host;
}

/** Stamp the host's local calendar day (backend-supplied `day` key, formatted
 *  YYYY-MM-DD) so the date-divider reconciler (utils/daymarks.ts) can group
 *  turns by day without re-parsing timestamps. Left unset when no message
 *  carries the key — such a turn simply joins the group above it. */
function stampDay(host: HTMLElement, block: ConversationTurnBlock): void {
  const day = block.messages[0]?.day;
  if (day) host.dataset.day = day;
}

/** Stamp `data-working` on a just-(re)mounted turn's own root element
 *  (found via `getTurnEl` scoped to `host` — the component's rendered root
 *  sits one level BELOW the wrapper `host` div, not on it). Combines the
 *  freshly-rendered block's own `working` flag with any live WS signal
 *  recorded before this element existed (see `_liveWorking` above) — the
 *  live signal wins over a stale snapshot's `false`. This is the mount-time
 *  fix for the gap where `setTurnWorking` no-ops on a not-yet-rendered turn. */
function stampWorking(host: HTMLElement, block: ConversationTurnBlock, type: string): void {
  const el = getTurnEl(block.turn_id, type, host);
  if (!el) return;
  // Lane claim (see SPINE_LANE_* above): a forked turn owns its work, so the
  // spine must stop counting it as its own. Re-derived from the block on every
  // upsert rather than recorded once, so a surface mounting this turn for the
  // FIRST time (the thread panel opening, or a turn re-materialising after
  // eviction) gets the claim too, not just the copy `markThreadLane` reached.
  //
  // Additive only, deliberately. Forked-ness is NOT monotonic server-side: a
  // reply cancelled before it produces any output is stripped from the turn
  // again, so a later block can legitimately arrive with no thread_message row.
  // Clearing on that falsy read would still be wrong, because `markThreadLane`
  // claims a turn the moment a reply is SENT and a refetch already in flight
  // behind that send resolves with the pre-reply block — clearing would hand
  // the reply's work straight back to the spine, which is the exact bug this
  // claim exists to prevent. The cost is a claim that outlives the fork; that
  // is safe only because turn_ids are never reissued, so the stale claim can
  // only ever be re-read for this same turn's own future work, which is a
  // further reply and therefore thread-owned anyway. Any future feature that
  // runs SPINE-scoped work under an existing turn_id (a "regenerate the
  // original answer", say) breaks that invariant and must revisit this.
  if (block.messages.some((m) => m.thread_message)) {
    el.setAttribute('data-lane-type', type);
    el.setAttribute('data-lane-turn-id', String(block.turn_id));
  }
  const working = block.working || _liveWorking.has(workingKey(block.turn_id, type));
  if (working) {
    el.setAttribute('data-working', 'true');
  } else {
    el.removeAttribute('data-working');
  }
  // Same catch-up for done (D16): a settle recorded while nothing rendered
  // this turn must surface on its first mount.
  if (_liveDone.has(workingKey(block.turn_id, type))) {
    el.setAttribute('data-done', 'true');
  }
}

/** Registered by `utils/sendEcho.ts`, lazily on its first real
 *  use rather than at that module's own top level (this module's default
 *  surface component, TurnView, transitively imports stores/session.ts,
 *  which imports sendEcho.ts back — an eager top-level registration call
 *  would run mid-cycle, before this file's own `let` below has executed).
 *  Notified after every real-content upsert below so the transient send echo
 *  can clear the moment the content it was standing in for actually lands.
 *  Dependency injection rather than a static import: sendEcho.ts already
 *  imports this module's surface registry (`resolveScopeContainer` above) to
 *  resolve where to mount its echo, so a reverse import here would cycle —
 *  same rationale as driftDispatcher's `registerSessionHooks`/`hooks()`. */
type TurnLandedHook = (turnId: number, type: string, isNewTopLevelTurn: boolean) => void;
let _turnLandedHook: TurnLandedHook | null = null;
export function onTurnLanded(hook: TurnLandedHook): void {
  _turnLandedHook = hook;
}

/**
 * Fan a block out to EVERY registered surface of its type whose `accepts`
 * passes (default: all). The dispatcher fetches a block ONCE per signal and
 * reaches every surface (spine + an open thread panel, say) through this.
 */
export function upsertTurnToSurfaces(
  block: ConversationTurnBlock,
  type: string,
  options: UpsertOptions = {},
): void {
  // §6.5 step 4 — the block's persisted tool_calls supersede its RESOLVED
  // live pills (an unresolved pill, a step still running, is left to finish
  // live). Fired here, the one shared upsert path every fetched block passes
  // through — the DOM-contract port of the retired buffer's `_writeTurn`
  // call, without which a mid-turn refetch renders the frozen pill AND its
  // collapsed chip side by side until the turn settles.
  clearLiveTurnsForToolCallsResolved(
    type,
    block.turn_id,
    block.messages.some((m) => m.tool_calls?.length),
  );

  // Captured BEFORE the loop below mutates the DOM: true only when
  // this block's turn_id has never rendered on the spine before (a genuinely
  // NEW top-level send, not a thread reply's refetch of an already-known
  // turn — see onTurnLanded's doc comment above).
  const spine = _surfaces.get(SPINE_SURFACE_ID);
  const isNewTopLevelTurn =
    spine != null && spine.type === type && !getTurnEl(block.turn_id, type, spine.container);

  let applied = false;
  for (const surface of _surfaces.values()) {
    if (surface.type !== type) continue;
    if (surface.accepts && !surface.accepts(block.turn_id)) continue;
    if (upsertTurn(block, type, surface.container, surface.component, surface.props, options)) {
      applied = true;
    }
  }

  // Only signal a landing that actually rendered. A version-rejected no-op
  // (every surface dropped a strictly-stale block) wrote nothing, so it must
  // not fire the land hook that clears a live send echo — that would blank the
  // echoed text for content the DOM never received.
  if (applied) _turnLandedHook?.(block.turn_id, type, isNewTopLevelTurn);
}

function mount(
  host: HTMLElement,
  component: Component,
  block: ConversationTurnBlock,
  type: string,
  extraProps: Record<string, unknown>,
): void {
  const vnode = createVNode(component, { block, type, ...extraProps });
  vnode.appContext = appContext;
  render(vnode, host);
}

function notifyUpserted(turnId: number, type: string): void {
  document.dispatchEvent(new CustomEvent('turn-upserted', { detail: { turnId, type } }));
}

/** Find the right slot in `container` so children stay sorted by turn_id. */
function insertInOrder(container: HTMLElement, host: HTMLElement, turnId: number): void {
  const children = Array.from(container.children);
  const idx = children.findIndex((child) => {
    const id = readTurnId(child);
    return id !== null && id > turnId;
  });
  if (idx === -1) {
    container.appendChild(host);
  } else {
    container.insertBefore(host, children[idx]);
  }
}

/** Read `data-turn-id` from an element or its first descendant that carries it. */
function readTurnId(el: Element): number | null {
  const attr =
    el.getAttribute('data-turn-id') ??
    el.querySelector('[data-turn-id]')?.getAttribute('data-turn-id') ??
    null;
  if (attr === null) return null;
  const num = Number(attr);
  return Number.isNaN(num) ? null : num;
}

/** Remove a turn's rendered host from EVERY surface (no-op when absent). */
export function removeTurn(turnId: number, type: string): void {
  for (const el of getAllTurnEls(turnId, type)) {
    const host = el.parentElement;
    if (host) {
      render(null, host);
      host.remove();
    }
  }
  // A removed turn cannot be "working" or "settled unseen": drop any live
  // record so `isTurnWorking`/`isTurnDone` (which check the records before
  // the DOM) don't keep answering true for a turn that no longer exists,
  // and tell consumers.
  _liveDone.delete(workingKey(turnId, type));
  if (_liveWorking.delete(workingKey(turnId, type))) {
    document.dispatchEvent(
      new CustomEvent('turn-state-changed', {
        detail: { turnId, type, working: false },
      }),
    );
  }
}

/** Evict the oldest spine hosts beyond the newest `keep`, but only ones
 *  scrolled fully above the viewport — never anything visible. Turns are
 *  DOM-sorted oldest→newest, so the evicted set is always a contiguous oldest
 *  prefix, leaving a contiguous newest block; evicted turns re-materialise from
 *  the history endpoint on scroll-up (D17), keeping the DOM bounded without
 *  losing history. Daymark dividers are skipped via the `[data-version]` filter
 *  (only turn hosts carry it) — the caller reconciles them afterwards. */
export function evictOldestAbove(container: HTMLElement, keep: number): void {
  const hosts = Array.from(container.querySelectorAll<HTMLElement>(':scope > [data-version]'));
  for (const host of hosts.slice(0, Math.max(0, hosts.length - keep))) {
    if (host.getBoundingClientRect().bottom >= 0) break; // reached the viewport — stop
    render(null, host);
    host.remove();
  }
}

/** Toggle the `data-working` attribute on every rendered copy and broadcast
 *  the change. */
export function setTurnWorking(turnId: number, type: string, working: boolean): void {
  const key = workingKey(turnId, type);
  if (working) {
    _liveWorking.add(key);
  } else {
    _liveWorking.delete(key);
  }

  // Dispatch even when no element is rendered yet: `isTurnWorking` derives
  // from the `_liveWorking` record above, so consumers (useDockBusy) must be
  // told the answer changed regardless of whether a DOM copy exists.
  for (const el of getAllTurnEls(turnId, type)) {
    if (working) {
      el.setAttribute('data-working', 'true');
    } else {
      el.removeAttribute('data-working');
    }
  }
  document.dispatchEvent(
    new CustomEvent('turn-state-changed', {
      detail: { turnId, type, working },
    }),
  );
}

/** D16 — record the done (seen/unseen) state, stamp `data-done` on every
 *  rendered copy, and broadcast the change. The `_liveDone` record is kept
 *  even with zero rendered copies (see its comment) so `isTurnDone` and a
 *  later mount both still see the state. */
export function setTurnDone(turnId: number, type: string, done: boolean): void {
  const key = workingKey(turnId, type);
  if (done) {
    _liveDone.add(key);
  } else {
    _liveDone.delete(key);
  }

  const els = getAllTurnEls(turnId, type);
  for (const el of els) {
    if (done) {
      el.setAttribute('data-done', 'true');
    } else {
      el.removeAttribute('data-done');
    }
  }
  document.dispatchEvent(
    new CustomEvent('turn-state-changed', {
      detail: { turnId, type, done },
    }),
  );
}

import { createVNode, render } from 'vue';
import type { App, AppContext, Component } from 'vue';
import TurnView from '../components/conversation/TurnView.vue';
import type { ConversationTurnBlock } from '../api/conversation';

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

/** Every rendered copy of a turn, across ALL surfaces — used by effects that
 *  must reach every visible copy of a turn (working/done flags, removal). */
function getAllTurnEls(turnId: number, type: string): HTMLElement[] {
  return Array.from(
    document.querySelectorAll<HTMLElement>(`[data-turn-id="${turnId}"][data-type="${type}"]`),
  );
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

// ── D13 — version guard ──────────────────────────────────────────────────────

export interface UpsertOptions {
  /** Bypass the monotonic version guard. Used ONLY by the post-cancel
   *  reconcile: a cancelled turn's server-stripped block can legitimately
   *  shrink (see driftDispatcher's cancelled-branch comment). */
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
    mount(host, component, block, type, extraProps);
    notifyUpserted(block.turn_id);
    return host;
  }

  const host = document.createElement('div');
  host.dataset.version = String(version);
  insertInOrder(container, host, block.turn_id);
  mount(host, component, block, type, extraProps);
  notifyUpserted(block.turn_id);
  return host;
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
  for (const surface of _surfaces.values()) {
    if (surface.type !== type) continue;
    if (surface.accepts && !surface.accepts(block.turn_id)) continue;
    upsertTurn(block, type, surface.container, surface.component, surface.props, options);
  }
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

function notifyUpserted(turnId: number): void {
  document.dispatchEvent(new CustomEvent('turn-upserted', { detail: { turnId } }));
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
}

/** Toggle the `data-working` attribute on every rendered copy and broadcast
 *  the change. */
export function setTurnWorking(turnId: number, type: string, working: boolean): void {
  const els = getAllTurnEls(turnId, type);
  if (!els.length) return;
  for (const el of els) {
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

/** D16 — toggle the `data-done` attribute (seen/unseen) on every rendered
 *  copy and broadcast the change. Replaces the buffer's `done` Set for
 *  rendering purposes. */
export function setTurnDone(turnId: number, type: string, done: boolean): void {
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

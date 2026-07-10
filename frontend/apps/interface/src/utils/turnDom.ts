import { createVNode, render } from 'vue';
import type { App, AppContext } from 'vue';
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

/** Look up a turn element by its data-keyed identity. */
export function getTurnEl(turnId: number, type: string): HTMLElement | null {
  return document.querySelector<HTMLElement>(
    `[data-turn-id="${turnId}"][data-type="${type}"]`,
  );
}

/** Insert or replace a turn block inside `container`, preserving numeric order. */
export function upsertTurn(
  block: ConversationTurnBlock,
  type: string,
  container: HTMLElement,
): HTMLElement {
  const existing = getTurnEl(block.turn_id, type);

  if (existing) {
    const host = existing.parentElement!;
    const vnode = createVNode(TurnView, { block, type });
    vnode.appContext = appContext;
    render(vnode, host);
    return host;
  }

  const host = document.createElement('div');
  insertInOrder(container, host, block.turn_id);
  const vnode = createVNode(TurnView, { block, type });
  vnode.appContext = appContext;
  render(vnode, host);
  return host;
}

/** Find the right slot in `container` so children stay sorted by turn_id. */
function insertInOrder(
  container: HTMLElement,
  host: HTMLElement,
  turnId: number,
): void {
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

/** Remove a turn's rendered host from the DOM (no-op when absent). */
export function removeTurn(turnId: number, type: string): void {
  const el = getTurnEl(turnId, type);
  if (!el) return;
  const host = el.parentElement;
  if (host) {
    render(null, host);
    host.remove();
  }
}

/** Toggle the `data-working` attribute and broadcast the change. */
export function setTurnWorking(
  turnId: number,
  type: string,
  working: boolean,
): void {
  const el = getTurnEl(turnId, type);
  if (!el) return;
  if (working) {
    el.setAttribute('data-working', 'true');
  } else {
    el.removeAttribute('data-working');
  }
  document.dispatchEvent(
    new CustomEvent('turn-state-changed', {
      detail: { turnId, type, working },
    }),
  );
}

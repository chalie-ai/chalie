// @vitest-environment happy-dom
/**
 * threadActivity — feature spec for the Activity drawer's derived state
 * (deriveThreadActivity) and the Scheduler dock's per-turn primitive
 * (threadPhase). Both read straight off the real DOM contract (turnDom's
 * registered surface + setTurnWorking/setTurnDone), no mocks.
 */
import { describe, expect, it, vi } from 'vitest';

async function fresh() {
  vi.resetModules();
  const turnDom = await import('./turnDom');
  const threadActivity = await import('./threadActivity');
  return { turnDom, threadActivity };
}

/** A forked-turn host, stamped exactly as TurnView.vue would. */
function forkedHost(opts: {
  turnId: number;
  type?: string;
  gist?: string | null;
  preview?: string;
  lastActivity?: string;
}): HTMLElement {
  const el = document.createElement('div');
  el.setAttribute('data-turn-id', String(opts.turnId));
  el.setAttribute('data-type', opts.type ?? 'user');
  el.setAttribute('data-forked', 'true');
  if (opts.gist) el.setAttribute('data-gist', opts.gist);
  el.setAttribute('data-preview', opts.preview ?? '');
  if (opts.lastActivity) el.setAttribute('data-last-activity', opts.lastActivity);
  return el;
}

describe('deriveThreadActivity', () => {
  it('reads only rows inside the registered spine surface, ignoring turns rendered elsewhere', async () => {
    const { turnDom, threadActivity } = await fresh();
    const spine = document.body.appendChild(document.createElement('div'));
    const other = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({ id: turnDom.SPINE_SURFACE_ID, type: 'user', container: spine, component: {} });

    const spineEl = forkedHost({ turnId: 1, preview: 'in the spine' });
    spineEl.setAttribute('data-done', 'true');
    spine.appendChild(spineEl);

    const otherEl = forkedHost({ turnId: 2, preview: 'not in the spine' });
    otherEl.setAttribute('data-done', 'true');
    other.appendChild(otherEl);

    const items = threadActivity.deriveThreadActivity('user');
    expect(items.map((i) => i.turn_id)).toEqual([1]);
    spine.remove();
    other.remove();
  });

  it('requires data-forked — a non-forked working/done turn on the spine is excluded', async () => {
    const { turnDom, threadActivity } = await fresh();
    const spine = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({ id: turnDom.SPINE_SURFACE_ID, type: 'user', container: spine, component: {} });

    const el = document.createElement('div');
    el.setAttribute('data-turn-id', '9');
    el.setAttribute('data-type', 'user');
    el.setAttribute('data-working', 'true');
    // deliberately no data-forked
    spine.appendChild(el);

    expect(threadActivity.deriveThreadActivity('user')).toEqual([]);
    spine.remove();
  });

  it('surfaces done turns only (kind "done"), excludes still-working turns; falls back gist → preview for the label', async () => {
    const { turnDom, threadActivity } = await fresh();
    const spine = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({ id: turnDom.SPINE_SURFACE_ID, type: 'user', container: spine, component: {} });

    // Still streaming — must NOT surface until the reply settles.
    const working = forkedHost({ turnId: 1, gist: 'a gist', preview: 'a preview', lastActivity: '2026-01-01 00:00:00' });
    working.setAttribute('data-working', 'true');
    spine.appendChild(working);

    // Settled unseen, has a gist — surfaces with kind 'done', gist wins the label.
    const doneWithGist = forkedHost({ turnId: 2, gist: 'has gist', preview: 'a preview', lastActivity: '2026-01-01 00:00:01' });
    doneWithGist.setAttribute('data-done', 'true');
    spine.appendChild(doneWithGist);

    // Settled unseen, no gist — label falls back to the preview.
    const doneNoGist = forkedHost({ turnId: 3, preview: 'no gist here', lastActivity: '2026-01-01 00:00:02' });
    doneNoGist.setAttribute('data-done', 'true');
    spine.appendChild(doneNoGist);

    const items = threadActivity.deriveThreadActivity('user');
    expect(items.map((i) => i.turn_id)).not.toContain(1); // working turn excluded
    const two = items.find((i) => i.turn_id === 2)!;
    const three = items.find((i) => i.turn_id === 3)!;
    expect(two.kind).toBe('done');
    expect(two.label).toBe('has gist');
    expect(three.kind).toBe('done');
    expect(three.label).toBe('no gist here');
    spine.remove();
  });

  it('sorts by data-last-activity descending (most recent first)', async () => {
    const { turnDom, threadActivity } = await fresh();
    const spine = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({ id: turnDom.SPINE_SURFACE_ID, type: 'user', container: spine, component: {} });

    const older = forkedHost({ turnId: 1, preview: 'older', lastActivity: '2026-01-01 00:00:00' });
    older.setAttribute('data-done', 'true');
    const newer = forkedHost({ turnId: 2, preview: 'newer', lastActivity: '2026-01-02 00:00:00' });
    newer.setAttribute('data-done', 'true');
    spine.append(older, newer);

    expect(threadActivity.deriveThreadActivity('user').map((i) => i.turn_id)).toEqual([2, 1]);
    spine.remove();
  });
});

describe('threadPhase', () => {
  it('is "working" while a live working signal is recorded, regardless of any rendered element', async () => {
    const { turnDom, threadActivity } = await fresh();
    turnDom.setTurnWorking(7, 'user', true);
    expect(threadActivity.threadPhase(7, 'user')).toBe('working');
  });

  it('is "done" from a live-only setTurnDone signal, with no rendered element anywhere', async () => {
    const { turnDom, threadActivity } = await fresh();
    turnDom.setTurnDone(42, 'user', true);
    expect(threadActivity.threadPhase(42, 'user')).toBe('done');
  });

  it('is "done" once working clears and a rendered copy carries data-done', async () => {
    const { turnDom, threadActivity } = await fresh();
    const container = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({ id: 'a', type: 'user', container, component: { render: () => null } });
    const el = document.createElement('div');
    el.setAttribute('data-turn-id', '8');
    el.setAttribute('data-type', 'user');
    el.setAttribute('data-done', 'true');
    container.appendChild(el);

    expect(threadActivity.threadPhase(8, 'user')).toBe('done');
    container.remove();
  });

  it('is null when neither working nor done — no activity to report', async () => {
    const { threadActivity } = await fresh();
    expect(threadActivity.threadPhase(999, 'user')).toBeNull();
  });
});

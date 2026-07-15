// @vitest-environment happy-dom
/**
 * sendEcho — feature spec for the optimistic "I just sent this" preview.
 *
 * Real DOM (happy-dom), real turnDom surface registry, and the REAL
 * UserBubble.vue rendered via Vue's actual render()/createVNode — nothing
 * mocked. `turnDom` is imported fresh alongside `sendEcho` on every test
 * (both carry module-level singletons: the surface registry + `_turnLandedHook`
 * in turnDom, the `_echoes` map + `_registered` flag in sendEcho) so no state
 * leaks between tests, mirroring turnDom.spec.ts's and cancelReconcile.spec.ts's
 * own `vi.resetModules()` + fresh-import pattern.
 */
import { describe, expect, it, vi } from 'vitest';
import { h } from 'vue';
import type { Component } from 'vue';
import type { ConversationTurnBlock } from '../api/conversation';

// A minimal render-function stub standing in for a turn's real render
// component (TurnView/SpineTurn) — same pattern as turnDom.spec.ts. Only
// used to simulate turns already landed on a surface; the echo itself always
// renders the REAL UserBubble.vue.
const StubComponent: Component = {
  props: ['block', 'type'],
  render(this: { block: ConversationTurnBlock; type: string }) {
    return h(
      'div',
      { 'data-turn-id': this.block.turn_id, 'data-type': this.type },
      `stub-${this.block.turn_id}`,
    );
  },
};

function block(turnId: number, messageIds: number[]): ConversationTurnBlock {
  return {
    turn_id: turnId,
    gist: null,
    preview: `turn ${turnId}`,
    last_activity_at: null,
    working: false,
    duration_ms: 0,
    type: 'user',
    messages: messageIds.map((id) => ({
      id: String(id),
      role: 'user',
      content: `msg ${id}`,
      timestamp: '2026-01-01 00:00:00',
      turn_id: turnId,
    })),
  };
}

async function fresh() {
  vi.resetModules();
  const turnDom = await import('./turnDom');
  const sendEcho = await import('./sendEcho');
  return { turnDom, sendEcho };
}

describe('mountSendEcho — container resolution', () => {
  it('mounts into the spine container on an immediate send, rendering the text with no turn-identity attributes', async () => {
    const { turnDom, sendEcho } = await fresh();
    const spineContainer = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: turnDom.SPINE_SURFACE_ID,
      type: 'user',
      container: spineContainer,
      component: StubComponent,
    });

    sendEcho.mountSendEcho('hello from me', null, 'user');

    const host = spineContainer.querySelector<HTMLElement>('[data-send-echo]');
    expect(host).not.toBeNull();
    expect(host!.textContent).toContain('hello from me');

    // DOM-contract safety: a stray turn-identity attribute on the echo would
    // make it indistinguishable from a real rendered turn to code that scans
    // the surface (getTurnEl, isSurfaceWorking, etc).
    expect(host!.hasAttribute('data-turn-id')).toBe(false);
    expect(host!.hasAttribute('data-type')).toBe(false);
    expect(host!.hasAttribute('data-working')).toBe(false);
    expect(host!.querySelector('[data-turn-id]')).toBeNull();
    expect(host!.querySelector('[data-type]')).toBeNull();
    expect(host!.querySelector('[data-working]')).toBeNull();

    spineContainer.remove();
  });

  it('mounts into the matching thread-panel surface for a thread-scoped send, not the spine', async () => {
    const { turnDom, sendEcho } = await fresh();
    const spineContainer = document.body.appendChild(document.createElement('div'));
    const panelContainer = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: turnDom.SPINE_SURFACE_ID,
      type: 'user',
      container: spineContainer,
      component: StubComponent,
    });
    turnDom.registerSurface({
      id: 'thread-panel-42',
      type: 'user',
      container: panelContainer,
      component: StubComponent,
      accepts: (turnId) => turnId === 42,
    });

    sendEcho.mountSendEcho('a reply', 42, 'user');

    expect(spineContainer.querySelector('[data-send-echo]')).toBeNull();
    const host = panelContainer.querySelector<HTMLElement>('[data-send-echo]');
    expect(host).not.toBeNull();
    expect(host!.textContent).toContain('a reply');

    spineContainer.remove();
    panelContainer.remove();
  });
});

describe('mountSendEcho — attachments render on the echo itself', () => {
  it('paints the attachment chips at mount, so a sent file is named on the FIRST frame rather than only after the refetch redraws the bubble', async () => {
    const { turnDom, sendEcho } = await fresh();
    const container = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: turnDom.SPINE_SURFACE_ID,
      type: 'user',
      container,
      component: StubComponent,
    });

    sendEcho.mountSendEcho('here are the files', null, 'user', [
      new File(['x'], 'holiday-snap.png', { type: 'image/png' }),
      new File(['y'], 'invoice.pdf', { type: 'application/pdf' }),
    ]);

    const host = container.querySelector<HTMLElement>('[data-send-echo]');
    expect(host!.textContent).toContain('holiday-snap.png');
    expect(host!.textContent).toContain('invoice.pdf');

    container.remove();
  });

  it("renders a file-only send as its chips alone — never the raw '[File attached]' placeholder, which UserBubble only suppresses while attachments are present", async () => {
    const { turnDom, sendEcho } = await fresh();
    const container = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: turnDom.SPINE_SURFACE_ID,
      type: 'user',
      container,
      component: StubComponent,
    });

    // Mirrors session.sendMessage's `text || FILE_PLACEHOLDER` for a send that
    // carries files and no typed text.
    sendEcho.mountSendEcho('[File attached]', null, 'user', [
      new File(['x'], 'scan.jpg', { type: 'image/jpeg' }),
    ]);

    const host = container.querySelector<HTMLElement>('[data-send-echo]');
    expect(host!.textContent).toContain('scan.jpg');
    expect(host!.textContent).not.toContain('[File attached]');

    container.remove();
  });

  it('renders no attachment list for a text-only send', async () => {
    const { turnDom, sendEcho } = await fresh();
    const container = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: turnDom.SPINE_SURFACE_ID,
      type: 'user',
      container,
      component: StubComponent,
    });

    sendEcho.mountSendEcho('just text', null, 'user');

    const host = container.querySelector<HTMLElement>('[data-send-echo]');
    expect(host!.querySelector('.user-attachments')).toBeNull();

    container.remove();
  });
});

describe('mountSendEcho — remount', () => {
  it('replaces the existing echo on the same scope rather than stacking a second one', async () => {
    const { turnDom, sendEcho } = await fresh();
    const container = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: turnDom.SPINE_SURFACE_ID,
      type: 'user',
      container,
      component: StubComponent,
    });

    sendEcho.mountSendEcho('first draft', null, 'user');
    sendEcho.mountSendEcho('second draft', null, 'user');

    const hosts = container.querySelectorAll('[data-send-echo]');
    expect(hosts.length).toBe(1);
    expect(hosts[0].textContent).toContain('second draft');
    expect(hosts[0].textContent).not.toContain('first draft');

    container.remove();
  });
});

describe('mountSendEcho — cleared by a landed turn, not by an unrelated refetch', () => {
  it('a genuinely new top-level turn landing on the spine clears the spine echo; a refetch of an already-known turn does not', async () => {
    const { turnDom, sendEcho } = await fresh();
    const spineContainer = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: turnDom.SPINE_SURFACE_ID,
      type: 'user',
      container: spineContainer,
      component: StubComponent,
    });

    // Turn 5 already lands on the spine BEFORE the echo mounts — later
    // upserts of turn 5 are reply refetches, not new top-level sends.
    turnDom.upsertTurnToSurfaces(block(5, [50]), 'user');

    sendEcho.mountSendEcho('typing…', null, 'user');
    expect(spineContainer.querySelector('[data-send-echo]')).not.toBeNull();

    // Refetch of the SAME turn (e.g. a reply landing inside it) — the spine
    // already had turn 5, so this must NOT be treated as a new top-level send.
    turnDom.upsertTurnToSurfaces(block(5, [50, 51]), 'user');
    expect(spineContainer.querySelector('[data-send-echo]')).not.toBeNull();

    // A genuinely new top-level turn — this is what the echo was standing in for.
    turnDom.upsertTurnToSurfaces(block(6, [60]), 'user');
    expect(spineContainer.querySelector('[data-send-echo]')).toBeNull();

    spineContainer.remove();
  });
});

describe('mountSendEcho — cleared by the reply\'s own turn landing on its thread-panel surface', () => {
  it('a reply landing on the thread panel it was echoed into clears the reply echo (not just a spine echo)', async () => {
    const { turnDom, sendEcho } = await fresh();
    const panelContainer = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: 'thread-panel-77',
      type: 'user',
      container: panelContainer,
      component: StubComponent,
      accepts: (turnId) => turnId === 77,
    });

    sendEcho.mountSendEcho('a reply', 77, 'user');
    expect(panelContainer.querySelector('[data-send-echo="user:77"]')).not.toBeNull();

    // The reply's real content lands — its first-ever mount into this surface.
    turnDom.upsertTurnToSurfaces(block(77, [770]), 'user');

    expect(panelContainer.querySelector('[data-send-echo="user:77"]')).toBeNull();
    panelContainer.remove();
  });
});

describe('mountSendEcho — survives a version-rejected (stale) upsert of its own turn', () => {
  it('a strictly-stale re-upsert of the echoed turn does not clear the echo; an equal-or-higher version does', async () => {
    const { turnDom, sendEcho } = await fresh();
    const panelContainer = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: 'thread-panel-88',
      type: 'user',
      container: panelContainer,
      component: StubComponent,
      accepts: (turnId) => turnId === 88,
    });

    // Turn 88 already rendered at version 880 BEFORE the echo mounts — later
    // upserts of it are refetches, not the send this echo stands in for.
    turnDom.upsertTurnToSurfaces(block(88, [880]), 'user');

    sendEcho.mountSendEcho('another reply', 88, 'user');
    expect(panelContainer.querySelector('[data-send-echo="user:88"]')).not.toBeNull();

    // Strictly-stale (version 5 < already-rendered 880) — every surface
    // rejects it, so `upsertTurnToSurfaces` never fires the land hook, and
    // the echo must survive: clearing it here would blank the echoed text
    // for content the DOM never actually received.
    turnDom.upsertTurnToSurfaces(block(88, [5]), 'user');
    expect(panelContainer.querySelector('[data-send-echo="user:88"]')).not.toBeNull();

    // An equal-or-higher version (881 >= 880) applies — the hook fires and
    // the echo is finally cleared.
    turnDom.upsertTurnToSurfaces(block(88, [880, 881]), 'user');
    expect(panelContainer.querySelector('[data-send-echo="user:88"]')).toBeNull();

    panelContainer.remove();
  });
});

describe('clearSendEcho', () => {
  it('removes the echo node entirely, leaving no orphan host in the container (interrupt/failure path)', async () => {
    const { turnDom, sendEcho } = await fresh();
    const container = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: turnDom.SPINE_SURFACE_ID,
      type: 'user',
      container,
      component: StubComponent,
    });

    sendEcho.mountSendEcho('interrupted send', null, 'user');
    expect(container.children.length).toBe(1);

    sendEcho.clearSendEcho(null, 'user');

    expect(container.children.length).toBe(0);
    expect(container.querySelector('[data-send-echo]')).toBeNull();

    container.remove();
  });
});

describe('mountSendEcho — no registered surface for the scope', () => {
  it('is a silent no-op: no throw, no stray DOM, for both an unregistered spine and an unclaimed thread scope', async () => {
    const { sendEcho } = await fresh();

    expect(() => sendEcho.mountSendEcho('nobody home', null, 'user')).not.toThrow();
    expect(document.querySelector('[data-send-echo]')).toBeNull();

    expect(() => sendEcho.mountSendEcho('nobody home either', 42, 'user')).not.toThrow();
    expect(document.querySelector('[data-send-echo]')).toBeNull();
  });
});

// @vitest-environment happy-dom
/**
 * PermissionStack — feature spec for the lane a permission card renders in:
 * a gate bubbles up to the channel the request came from, for every tool.
 *
 * Real PermissionStack → PermissionCard tree, real Pinia (permissions +
 * session stores), real `@chalie/shared` barrel, real Teleports — into a host
 * that mirrors App.vue: the panel's target (`#permStackPanel`, which
 * ThreadPanel.vue renders above its own dock) exists only while the session
 * says the panel is open, and it sits BEFORE the stack in the tree, the order
 * App.vue mounts them. `#permStack` is the static spine target.
 */
import { beforeEach, describe, expect, it } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { defineComponent, h, nextTick } from 'vue';
import { ConfigType } from '@chalie/shared';
import type { WsPushEvent } from '@chalie/shared';
import PermissionStack from './PermissionStack.vue';
import { usePermissionsStore } from '../../stores/permissions';
import { useSessionStore } from '../../stores/session';

/** App.vue in miniature: ThreadPanel's in-flow target (only while the panel is open), then the stack. */
const Host = defineComponent({
  setup() {
    const session = useSessionStore();
    return () =>
      h('div', [
        session.panelThreadId != null ? h('div', { id: 'permStackPanel' }) : null,
        h(PermissionStack),
      ]);
  },
});

/** A `permission_request` frame as the drift dispatcher hands it to the store. */
function gate(
  requestId: string,
  origin: { type: string; turn_id: number; forked: boolean } | null,
  summary = 'Read the inbox',
): WsPushEvent {
  const data = { type: 'permission_request', request_id: requestId, action_id: 'pim', summary, origin };
  return data as unknown as WsPushEvent;
}

const spineCard = (id: string) =>
  document.querySelector<HTMLElement>(`#permStack .perm-card[data-request-id="${id}"]`);
const panelCard = (id: string) =>
  document.querySelector<HTMLElement>(`#permStackPanel .perm-card[data-request-id="${id}"]`);
const laneText = (card: HTMLElement | null) =>
  card?.querySelector('.perm-card__lane-text')?.textContent?.trim() ?? null;

/** Let the render, the deferred Teleport, and any queued microtasks all land. */
async function settle(): Promise<void> {
  await nextTick();
  await flushPromises();
  await nextTick();
}

beforeEach(() => {
  setActivePinia(createPinia());
  document.body.innerHTML = '<div id="permStack"></div>';
});

describe('PermissionStack — lanes', () => {
  it('a main-spine turn\'s card renders in the spine stack, unlabelled, even with a panel open on something else', async () => {
    const session = useSessionStore();
    const permissions = usePermissionsStore();
    session.openThreadPanel(99, ConfigType.USER);
    const wrapper = mount(Host, { attachTo: document.body });

    permissions.enqueue(gate('r1', { type: ConfigType.USER, turn_id: 5, forked: false }));
    await settle();

    const card = spineCard('r1');
    expect(card).not.toBeNull();
    expect(panelCard('r1')).toBeNull();
    expect(card!.querySelector('.perm-card__lane')).toBeNull();
    expect(card!.querySelector('.perm-card__title')!.textContent!.trim()).toBe('Read the inbox');
    // The gated permission still shows, beneath the summary.
    expect(card!.querySelector('.perm-card__desc')!.textContent!.trim()).toBe('Access Email, Calendar & Contacts');
    wrapper.unmount();
  });

  it('a card without a summary falls back to the permission\'s label as its title', async () => {
    const permissions = usePermissionsStore();
    const wrapper = mount(Host, { attachTo: document.body });

    permissions.enqueue(gate('r1', { type: ConfigType.USER, turn_id: 5, forked: false }, ''));
    await settle();

    const card = spineCard('r1');
    expect(card!.querySelector('.perm-card__title')!.textContent!.trim()).toBe('Access Email, Calendar & Contacts');
    expect(card!.querySelector('.perm-card__desc')).toBeNull();
    wrapper.unmount();
  });

  it('a thread turn\'s card waits on the spine labelled "Thread · <heading>" with an Open button while its panel is closed', async () => {
    const permissions = usePermissionsStore();
    // The thread's root turn as the spine renders it (TurnView host attributes).
    document.body.insertAdjacentHTML(
      'beforeend',
      '<div data-turn-id="7" data-type="user" data-gist="Trip plans" data-preview="plan my trip"></div>',
    );
    const wrapper = mount(Host, { attachTo: document.body });

    permissions.enqueue(gate('r2', { type: ConfigType.USER, turn_id: 7, forked: true }));
    await settle();

    const card = spineCard('r2');
    expect(card).not.toBeNull();
    expect(laneText(card)).toBe('Thread · Trip plans');
    expect(card!.querySelector('.perm-card__lane-open')).not.toBeNull();
    wrapper.unmount();
  });

  it('Open on a labelled card opens that turn\'s panel; the card moves into the panel, unlabelled, and back out when the panel closes', async () => {
    const session = useSessionStore();
    const permissions = usePermissionsStore();
    const wrapper = mount(Host, { attachTo: document.body });

    permissions.enqueue(gate('r2', { type: ConfigType.USER, turn_id: 7, forked: true }));
    await settle();
    // No spine host for turn 7 — the label falls back to the id.
    expect(laneText(spineCard('r2'))).toBe('Thread · #7');

    spineCard('r2')!.querySelector<HTMLButtonElement>('.perm-card__lane-open')!.click();
    await settle();

    expect(session.panelThreadId).toBe(7);
    expect(session.panelType).toBe(ConfigType.USER);
    expect(spineCard('r2')).toBeNull();
    const inPanel = panelCard('r2');
    expect(inPanel).not.toBeNull();
    expect(inPanel!.querySelector('.perm-card__lane')).toBeNull();
    expect(document.querySelectorAll('.perm-card[data-request-id="r2"]')).toHaveLength(1);

    session.closeThreadPanel();
    await settle();

    expect(panelCard('r2')).toBeNull();
    expect(laneText(spineCard('r2'))).toBe('Thread · #7');
    wrapper.unmount();
  });

  it('a scheduled turn\'s card is labelled "Scheduled task · #<id>" and Open opens the scheduled panel', async () => {
    const session = useSessionStore();
    const permissions = usePermissionsStore();
    const wrapper = mount(Host, { attachTo: document.body });

    permissions.enqueue(gate('r3', { type: ConfigType.SCHEDULED, turn_id: 3, forked: false }));
    await settle();
    expect(laneText(spineCard('r3'))).toBe('Scheduled task · #3');

    spineCard('r3')!.querySelector<HTMLButtonElement>('.perm-card__lane-open')!.click();
    await settle();

    expect(session.panelThreadId).toBe(3);
    expect(session.panelType).toBe(ConfigType.SCHEDULED);
    expect(panelCard('r3')).not.toBeNull();
    expect(spineCard('r3')).toBeNull();
    wrapper.unmount();
  });

  it('a panel open on the same id under another type is not the card\'s panel — it stays on the spine, labelled', async () => {
    const session = useSessionStore();
    const permissions = usePermissionsStore();
    session.openThreadPanel(7, ConfigType.SCHEDULED);
    const wrapper = mount(Host, { attachTo: document.body });

    permissions.enqueue(gate('r2', { type: ConfigType.USER, turn_id: 7, forked: true }));
    await settle();

    expect(panelCard('r2')).toBeNull();
    expect(laneText(spineCard('r2'))).toBe('Thread · #7');
    wrapper.unmount();
  });

  it('a card already queued when its panel opens lands in the panel; a resolved gate leaves every lane', async () => {
    const session = useSessionStore();
    const permissions = usePermissionsStore();
    const wrapper = mount(Host, { attachTo: document.body });

    permissions.enqueue(gate('r2', { type: ConfigType.USER, turn_id: 7, forked: true }));
    permissions.enqueue(gate('r1', { type: ConfigType.USER, turn_id: 5, forked: false }));
    await settle();

    session.openThreadPanel(7, ConfigType.USER);
    await settle();
    expect(panelCard('r2')).not.toBeNull();
    expect(spineCard('r1')).not.toBeNull(); // the main-spine card does not follow

    permissions.remove('r2'); // what a permission_resolved frame drives
    permissions.remove('r1');
    await settle();
    expect(document.querySelectorAll('.perm-card')).toHaveLength(0);
    wrapper.unmount();
  });
});

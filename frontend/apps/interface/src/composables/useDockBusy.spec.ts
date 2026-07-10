// @vitest-environment happy-dom
/**
 * useDockBusy — feature spec for the DOM-event-driven busy read (D3/D5).
 *
 * Real DOM (happy-dom, mirroring turnDom.spec.ts / TurnView.spec.ts), real
 * `utils/turnDom.ts` module (surface registry + live-working bookkeeping),
 * real Vue reactivity via a minimal host component mounted with
 * `@vue/test-utils`'s `mount` (already a project dependency — see
 * components/conversation/TurnView.spec.ts for the established pattern).
 * No mocks: this composable's whole job is translating two DOM CustomEvents
 * into a reactive ref, so the DOM IS the thing under test.
 *
 * turnDom's surface registry / live-working set are module-level singletons
 * shared across this file's tests (no `vi.resetModules()`, matching
 * TurnView.spec.ts's convention) — each test uses its own turnId/surface id
 * to avoid cross-test collisions.
 */
import { describe, expect, it, vi } from 'vitest';
import { defineComponent, h, ref } from 'vue';
import { mount } from '@vue/test-utils';
import { useDockBusy } from './useDockBusy';
import { registerSurface, setTurnWorking, SPINE_SURFACE_ID } from '../utils/turnDom';

describe('useDockBusy — a specific turnId', () => {
  it('refreshes on turn-state-changed for its OWN turnId', async () => {
    const turnId = ref<number | null>(201);
    const Host = defineComponent({
      setup() {
        const busy = useDockBusy(() => turnId.value, () => 'user');
        return { busy };
      },
      render() { return h('div'); },
    });
    const wrapper = mount(Host);
    expect(wrapper.vm.busy).toBe(false);

    setTurnWorking(201, 'user', true);
    await wrapper.vm.$nextTick();
    expect(wrapper.vm.busy).toBe(true);

    setTurnWorking(201, 'user', false);
    await wrapper.vm.$nextTick();
    expect(wrapper.vm.busy).toBe(false);

    wrapper.unmount();
  });
});

describe('useDockBusy — turnId() === null', () => {
  it('keys off the registered SPINE_SURFACE_ID container and refreshes on turn-upserted too', async () => {
    const container = document.body.appendChild(document.createElement('div'));
    registerSurface({ id: SPINE_SURFACE_ID, type: 'user', container, component: {} });

    const Host = defineComponent({
      setup() {
        const busy = useDockBusy(() => null, () => 'user');
        return { busy };
      },
      render() { return h('div'); },
    });
    const wrapper = mount(Host);
    expect(wrapper.vm.busy).toBe(false);

    // stampWorking (upsertTurn's own effect) has no dedicated event of its
    // own — mounts fire 'turn-upserted' instead, which useDockBusy must also
    // treat as a refresh trigger.
    container.innerHTML = '<div data-working></div>';
    document.dispatchEvent(new CustomEvent('turn-upserted', { detail: { turnId: 1 } }));
    await wrapper.vm.$nextTick();
    expect(wrapper.vm.busy).toBe(true);

    wrapper.unmount();
    container.remove();
  });
});

describe('useDockBusy — teardown', () => {
  it('unregisters its turn-state-changed and turn-upserted listeners on unmount', () => {
    const removeSpy = vi.spyOn(document, 'removeEventListener');
    const Host = defineComponent({
      setup() {
        useDockBusy(() => 302, () => 'user');
        return () => h('div');
      },
    });
    const wrapper = mount(Host);

    wrapper.unmount();

    const removedTypes = removeSpy.mock.calls.map((call) => call[0]);
    expect(removedTypes).toContain('turn-state-changed');
    expect(removedTypes).toContain('turn-upserted');
    removeSpy.mockRestore();
  });
});

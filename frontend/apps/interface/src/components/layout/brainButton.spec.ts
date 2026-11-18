// @vitest-environment happy-dom
/**
 * The header's Brain button — feature spec.
 *
 * Clicking it takes the current tab to the Brain at /brain/; it never opens a
 * new tab or window. The real header is mounted; the only stand-ins are spies
 * on the browser's navigation calls, which a test page cannot follow.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import PresenceBar from './PresenceBar.vue';

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('header Brain button', () => {
  it('navigates the current tab to /brain/ and opens no new tab', () => {
    setActivePinia(createPinia());
    const open = vi.spyOn(globalThis, 'open').mockImplementation(() => null);
    const assign = vi.spyOn(globalThis.location, 'assign').mockImplementation(() => undefined);

    const header = mount(PresenceBar, { attachTo: document.body });
    header.get('#settingsBtn').trigger('click');

    expect(assign).toHaveBeenCalledExactlyOnceWith('/brain/');
    expect(open).not.toHaveBeenCalled();
    header.unmount();
  });
});

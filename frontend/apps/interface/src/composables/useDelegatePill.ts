/**
 * Turns a tool pill into the way into its delegate's transcript. A pill whose
 * call spawned a delegate turn renders as a real button that opens the panel
 * on that turn; every other pill gets no attributes and stays inert. The
 * button's accessible name is its visible content; `title` only describes it.
 */
import type { DelegateRef } from '@chalie/shared';
import { useSessionStore } from '../stores/session';

export function delegatePillAttrs(delegate: DelegateRef | null): Record<string, unknown> {
  if (!delegate) return {};
  return {
    type: 'button',
    class: 'delegate-pill',
    title: 'Open transcript',
    onClick: () => useSessionStore().openDelegatePanel(delegate),
  };
}

/**
 * isDelegateChannel — the one predicate deciding whether a frame, pill feed or
 * panel target belongs to a delegate (subagent) turn. A wrong `true` hands a
 * user turn's frame to the delegate path (and a wrong `false` hands a
 * delegate's frame to a same-numbered user turn), so the boundary is pinned on
 * both sides with the channel names the backend really emits.
 *
 * Pure function, no collaborators — the one place a plain unit spec is right.
 */
import { describe, expect, it } from 'vitest';
import { isDelegateChannel } from './delegateChannel';

describe('isDelegateChannel', () => {
  it.each(['delegate:web_search', 'delegate:code_agent', 'delegate:web_browse'])(
    'recognises the delegate channel %s',
    (channel) => {
      expect(isDelegateChannel(channel)).toBe(true);
    },
  );

  it.each([
    ['the user channel', 'user'],
    ['a scheduled channel', 'scheduled'],
    ['the discovery channel', 'discovery'],
    ['a differently-cased prefix', 'Delegate:web_search'],
    ['the prefix with no colon', 'delegate'],
    ['a name that merely contains the prefix', 'not-a-delegate:web_search'],
    ['an empty string', ''],
  ])('rejects %s', (_label, channel) => {
    expect(isDelegateChannel(channel)).toBe(false);
  });

  it.each([
    ['undefined', undefined],
    ['null', null],
    ['a number', 7],
    ['an object', { channel: 'delegate:web_search' }],
    ['an array holding a delegate channel', ['delegate:web_search']],
  ])('rejects a non-string (%s) without throwing', (_label, value) => {
    expect(isDelegateChannel(value)).toBe(false);
  });
});

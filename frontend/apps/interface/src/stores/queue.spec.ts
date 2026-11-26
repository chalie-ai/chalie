/**
 * Queue store — feature spec for the just-fixed defect where queued files were
 * dropped on flush: `enqueue` now stores `{text, files}` per entry and `take()`
 * must return every queued entry's text (newline-joined, oldest first) AND
 * every queued entry's files (concatenated in the same order), not text alone.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';

// The only mock in this file: `@chalie/shared` is substituted purely because
// the interface app's vitest config has no Vue SFC plugin, so the real
// barrel's `.vue` re-exports fail to parse — `ConfigType` below carries its
// real production values, nothing under test is faked.
vi.mock('@chalie/shared', () => ({
  ConfigType: { USER: 'user', SCHEDULED: 'scheduled', DISCOVERY: 'discovery' },
}));

import { ConfigType } from '@chalie/shared';
import { useQueueStore } from './queue';

beforeEach(() => {
  setActivePinia(createPinia());
});

describe('enqueue / take — queued files survive the flush', () => {
  it('joins queued texts with a newline and concatenates their files in send order', () => {
    const queue = useQueueStore();
    const fileA = new File(['a'], 'a.txt');
    const fileB = new File(['b'], 'b.txt');

    queue.enqueue(42, 'first queued line', ConfigType.USER, [fileA]);
    queue.enqueue(42, 'second queued line', ConfigType.USER, [fileB]);

    const { text, files } = queue.take(42);

    expect(text).toBe('first queued line\nsecond queued line');
    expect(files).toEqual([fileA, fileB]);
    // take() clears the scope — a second take on the same scope is empty.
    expect(queue.take(42)).toEqual({ text: '', files: [] });
  });
});

describe('take — quote blocks in queued messages', () => {
  it('lifts every queued block to the head of the joined message, oldest first, and joins the texts after them', () => {
    const queue = useQueueStore();
    const block = (quote: string): string => `[quote_block]{"quote": "${quote}", "reply": ""}[/quote_block]`;

    queue.enqueue(42, `${block('one')}\n\nfirst line`, ConfigType.USER);
    queue.enqueue(42, 'plain second line', ConfigType.USER);
    queue.enqueue(42, block('two'), ConfigType.USER);

    expect(queue.take(42).text).toBe(`${block('one')}\n${block('two')}\n\nfirst line\nplain second line`);
  });

  it('joins messages without blocks exactly as before, a files-only entry keeping its empty line', () => {
    const queue = useQueueStore();
    queue.enqueue(42, 'first', ConfigType.USER);
    queue.enqueue(42, '', ConfigType.USER, [new File(['x'], 'a.txt')]);
    queue.enqueue(42, 'third', ConfigType.USER);

    expect(queue.take(42).text).toBe('first\n\nthird');
  });
});

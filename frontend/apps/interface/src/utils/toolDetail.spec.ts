/**
 * lineDiff — feature spec for the line diff an opened file edit shows.
 *
 * Lines both sides share read as context; where they differ, the removed
 * lines come before the added ones. A trailing newline, or Windows line
 * endings, are not a change. Past a million table cells the line match is
 * skipped and every old line reads as removed, then every new one as added.
 */
import { describe, expect, it } from 'vitest';
import { lineDiff, type DiffLine } from './toolDetail';

const same = (text: string): DiffLine => ({ op: 'same', text });
const add = (text: string): DiffLine => ({ op: 'add', text });
const remove = (text: string): DiffLine => ({ op: 'remove', text });

describe('lineDiff', () => {
  it('shows identical text as context only, a pure insert as added lines among it, and a pure delete as removed lines among it', () => {
    expect(lineDiff('a\nb\nc', 'a\nb\nc')).toEqual([same('a'), same('b'), same('c')]);
    expect(lineDiff('a\nc', 'a\nb\nc')).toEqual([same('a'), add('b'), same('c')]);
    expect(lineDiff('a\nb\nc', 'a\nc')).toEqual([same('a'), remove('b'), same('c')]);
  });

  it('shows a changed line among unchanged ones as removed then added, in place', () => {
    expect(lineDiff('def cap(rent):\n    return rent * 2\n    # end', 'def cap(rent):\n    return rent * 1.5\n    # end')).toEqual([
      same('def cap(rent):'), remove('    return rent * 2'), add('    return rent * 1.5'), same('    # end'),
    ]);
    expect(lineDiff('a\nb\nc\nd', 'a\nB\nc\nD')).toEqual([
      same('a'), remove('b'), add('B'), same('c'), remove('d'), add('D'),
    ]);
  });

  it('does not count a trailing newline or Windows line endings as a change, and shows an empty side as all added or all removed', () => {
    expect(lineDiff('a\r\nb\r\n', 'a\nb')).toEqual([same('a'), same('b')]);
    // A second trailing newline is a blank line that was added.
    expect(lineDiff('a\n', 'a\n\n')).toEqual([same('a'), add('')]);
    expect(lineDiff('', 'x\ny')).toEqual([add('x'), add('y')]);
    expect(lineDiff('x\n', '')).toEqual([remove('x')]);
    expect(lineDiff('', '')).toEqual([]);
  });

  it('past a million table cells shows every old line removed then every new one added, and up to a million still matches lines', () => {
    const lines = (n: number, changed: number) => Array.from({ length: n }, (_, i) => (i === changed ? 'changed' : `line ${i}`)).join('\n');

    // 1001 × 1001 cells: past the limit.
    const big = lineDiff(lines(1000, -1), lines(1000, 500));
    expect(big).toHaveLength(2000);
    expect(big.slice(0, 1000).every((l) => l.op === 'remove')).toBe(true);
    expect(big.slice(1000).every((l) => l.op === 'add')).toBe(true);
    expect(big[1500]).toEqual(add('changed'));

    // 1000 × 1000 cells: exactly at the limit, still matched line by line.
    const atLimit = lineDiff(lines(999, -1), lines(999, 500));
    expect(atLimit).toHaveLength(1000);
    expect(atLimit.filter((l) => l.op !== 'same')).toEqual([remove('line 500'), add('changed')]);
  });
});

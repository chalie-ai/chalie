/**
 * Quote blocks — feature spec for the message a quote rides in: the exact text
 * a send carries, reading it back out of a message the chat renders, and the
 * one-line label a thread list shows for one.
 */
import { describe, expect, it } from 'vitest';
import { encodeQuoteBlocks, parseQuoteBlocks, quotePreview } from './quoteBlocks';

const first = { quote: 'the highlighted text', reply: 'my reply' };
const second = { quote: 'the highlighted text 2', reply: 'my reply 2' };

describe('the message a send carries', () => {
  it('puts each block on its own line, then a blank line, then the typed text', () => {
    expect(encodeQuoteBlocks([first, second], 'my regular message')).toBe(
      '[quote_block]{"quote": "the highlighted text", "reply": "my reply"}[/quote_block]\n' +
        '[quote_block]{"quote": "the highlighted text 2", "reply": "my reply 2"}[/quote_block]\n' +
        '\n' +
        'my regular message',
    );
  });

  it('is the typed text alone, untouched, when there are no blocks', () => {
    expect(encodeQuoteBlocks([], '  hello\n\nthere ')).toBe('  hello\n\nthere ');
  });

  it('is the blocks alone when nothing was typed', () => {
    expect(encodeQuoteBlocks([first], '')).toBe(
      '[quote_block]{"quote": "the highlighted text", "reply": "my reply"}[/quote_block]',
    );
  });

  it('keeps a quote that spans lines, or holds quote marks and backslashes, on its one line', () => {
    const block = { quote: 'line one\nline "two" \\ end', reply: '' };
    const sent = encodeQuoteBlocks([block], 'ok');
    expect(sent).toBe(
      '[quote_block]{"quote": "line one\\nline \\"two\\" \\\\ end", "reply": ""}[/quote_block]\n\nok',
    );
    expect(parseQuoteBlocks(sent)).toEqual({ blocks: [block], text: 'ok' });
  });
});

describe('reading a message back', () => {
  it('splits the blocks it opens with from the typed text after them', () => {
    const sent = encodeQuoteBlocks([first, second], 'my regular message\n\nsecond paragraph');
    expect(parseQuoteBlocks(sent)).toEqual({
      blocks: [first, second],
      text: 'my regular message\n\nsecond paragraph',
    });
  });

  it('reads the message as the server stores it, every line break sent as CR LF', () => {
    const sent = encodeQuoteBlocks([first, second], 'my regular message\nsecond line');
    expect(parseQuoteBlocks(sent.replaceAll('\n', '\r\n'))).toEqual({
      blocks: [first, second],
      text: 'my regular message\r\nsecond line',
    });
  });

  it('reads a message of blocks alone as blocks with no text', () => {
    expect(parseQuoteBlocks(encodeQuoteBlocks([first, second], ''))).toEqual({
      blocks: [first, second],
      text: '',
    });
  });

  it('leaves a message with no blocks whole', () => {
    const content = '\nnot a block [quote_block]{"quote": "a", "reply": "b"}[/quote_block]\n';
    expect(parseQuoteBlocks(content)).toEqual({ blocks: [], text: content });
  });

  it('stops at the first line that is not a block; that line and the rest stay text', () => {
    const content =
      '[quote_block]{"quote": "a", "reply": "b"}[/quote_block]\n[quote_block]not json[/quote_block]\nrest';
    expect(parseQuoteBlocks(content)).toEqual({
      blocks: [{ quote: 'a', reply: 'b' }],
      text: '[quote_block]not json[/quote_block]\nrest',
    });
  });

  it('reads a marker without both fields as plain text', () => {
    const content = '[quote_block]{"quote": "a"}[/quote_block]\n\nhi';
    expect(parseQuoteBlocks(content)).toEqual({ blocks: [], text: content });
  });
});

describe('the one-line label of a message that opens with blocks', () => {
  it('is the typed text', () => {
    expect(quotePreview(encodeQuoteBlocks([first], 'what about this?'))).toBe('what about this?');
  });

  it('is the first quote in curly quotes when nothing was typed', () => {
    expect(quotePreview(encodeQuoteBlocks([first, second], ''))).toBe('“the highlighted text”');
  });

  it('cuts a quote the preview was cut inside with an ellipsis, escapes read', () => {
    const sent = encodeQuoteBlocks([{ quote: 'a "long" passage', reply: 'r' }], '');
    // Cut part-way into the quote, after an escaped quote mark …
    expect(quotePreview(sent.slice(0, sent.indexOf('pass') + 2))).toBe('“a "long" pa…”');
    // … and inside an escape, which is dropped rather than misread.
    expect(quotePreview(sent.slice(0, sent.indexOf('\\"long') + 1))).toBe('“a …”');
  });

  it('shows the quote, not the second block, when the cut fell inside that block', () => {
    const sent = encodeQuoteBlocks([first, second], 'typed');
    expect(quotePreview(sent.slice(0, sent.indexOf('text 2')))).toBe('“the highlighted text”');
  });

  it('leaves a message without blocks as it is', () => {
    expect(quotePreview('just a question')).toBe('just a question');
  });
});

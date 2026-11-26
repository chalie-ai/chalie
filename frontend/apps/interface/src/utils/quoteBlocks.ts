/**
 * Quote blocks — a piece of Chalie's reply the user quoted, with their reply
 * to it, carried at the head of an ordinary user message:
 *
 *   [quote_block]{"quote": "<quoted text>", "reply": "<reply>"}[/quote_block]
 *   [quote_block]{"quote": "<quoted text 2>", "reply": "<reply 2>"}[/quote_block]
 *
 *   <the typed message>
 *
 * One block per line (JSON escapes any newline inside a value), then a blank
 * line, then the typed text. With no blocks the message is just its text.
 */

export interface QuoteBlock {
  quote: string;
  reply: string;
}

/** Hands a new block to the composer named by `dockId` — the id of its dock. */
export const QUOTE_BLOCK_EVENT = 'chalie:quote-block';

export interface QuoteBlockEventDetail {
  dockId: string;
  block: QuoteBlock;
}

const OPEN = '[quote_block]';
const CLOSE = '[/quote_block]';
const QUOTE_START = `${OPEN}{"quote": "`;

function encodeBlock(block: QuoteBlock): string {
  return `${OPEN}{"quote": ${JSON.stringify(block.quote)}, "reply": ${JSON.stringify(block.reply)}}${CLOSE}`;
}

/** The message a send carries: the blocks, a blank line, then the text. */
export function encodeQuoteBlocks(blocks: readonly QuoteBlock[], text: string): string {
  if (!blocks.length) return text;
  const head = blocks.map(encodeBlock).join('\n');
  return text ? `${head}\n\n${text}` : head;
}

function decodeLine(line: string): QuoteBlock | null {
  if (!line.startsWith(OPEN) || !line.endsWith(CLOSE)) return null;
  try {
    const value: unknown = JSON.parse(line.slice(OPEN.length, -CLOSE.length));
    if (typeof value !== 'object' || value === null) return null;
    const { quote, reply } = value as Record<string, unknown>;
    return typeof quote === 'string' && typeof reply === 'string' ? { quote, reply } : null;
  } catch {
    // Not a block, only text that starts like one: it stays text.
    return null;
  }
}

/** Split a message into its leading blocks and the text after them. A message
 *  with no blocks comes back whole as its text. A browser sends a form field's
 *  line breaks as CR LF, so a stored message reads back with either. */
export function parseQuoteBlocks(content: string): { blocks: QuoteBlock[]; text: string } {
  const blocks: QuoteBlock[] = [];
  let rest = content;
  while (rest.startsWith(OPEN)) {
    const end = rest.indexOf('\n');
    const line = end === -1 ? rest : rest.slice(0, end);
    const block = decodeLine(line.endsWith('\r') ? line.slice(0, -1) : line);
    if (!block) break;
    blocks.push(block);
    rest = end === -1 ? '' : rest.slice(end + 1);
  }
  if (!blocks.length) return { blocks, text: content };
  const blank = /^\r?\n/.exec(rest)?.[0].length ?? 0;
  return { blocks, text: rest.slice(blank) };
}

/** The quoted text of a block cut off part-way, as a message preview is: the
 *  JSON string read up to where it ends, or up to the cut with an ellipsis. */
function clippedQuote(line: string): string | null {
  if (!line.startsWith(QUOTE_START)) return null;
  const body = line.slice(QUOTE_START.length);
  let raw = '';
  let i = 0;
  while (i < body.length && body[i] !== '"') {
    const size = body[i] !== '\\' ? 1 : body[i + 1] === 'u' ? 6 : 2;
    if (i + size > body.length) break; // the cut fell inside this escape
    raw += body.slice(i, i + size);
    i += size;
  }
  try {
    const quote = JSON.parse(`"${raw}"`) as string;
    return body[i] === '"' ? quote : `${quote}…`;
  } catch {
    return null;
  }
}

/** A one-line label for a message that may open with blocks — a thread's
 *  preview, cut short by the server: the typed text, else the first quote in
 *  curly quotes. Anything else is returned as it is. */
export function quotePreview(preview: string): string {
  if (!preview.startsWith(OPEN)) return preview;
  const { blocks, text } = parseQuoteBlocks(preview);
  // A block the cut fell inside is no part of the typed text.
  if (text && !text.startsWith(OPEN)) return text;
  const quote = blocks[0]?.quote ?? clippedQuote(text);
  return quote ? `“${quote}”` : preview;
}

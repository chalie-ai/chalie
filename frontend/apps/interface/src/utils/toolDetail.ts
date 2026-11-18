/**
 * toolDetail — what an opened tool-call row shows: only what matters for its
 * kind of call, built from the call as read on its own. A command reads as
 * the command, a file edit as a line diff of what it replaced, a delegate as
 * what it was asked, and anything else as its arguments.
 */
import type { ToolCallDetail, ToolCallPage } from '../api/conversation';
import { callKind, type CallKind, type CallView } from './toolCatalog';

export interface DiffLine {
  op: 'same' | 'add' | 'remove';
  text: string;
}

export interface DetailField {
  label: string;
  value: string;
  /** A structured value, shown as pretty JSON. */
  nested: boolean;
}

/** One variant per view. `heading` is the call's `action`; a `write` view's
 *  `diff` lists every line as added when the call created the file, and is
 *  empty when it overwrote one (its `contents` then show as they are). */
export type ToolDetailView =
  | { kind: 'command'; command: string }
  | { kind: 'edit'; path: string; diff: DiffLine[] }
  | { kind: 'write'; path: string; contents: string; diff: DiffLine[] }
  | { kind: 'read'; source: string; lines: string }
  | { kind: 'browser'; heading: string; fields: DetailField[]; pages: ToolCallPage[] }
  | { kind: 'browse'; fields: DetailField[]; pages: ToolCallPage[] }
  | { kind: 'delegate'; fields: DetailField[] }
  | { kind: 'email'; heading: string; fields: DetailField[] }
  | { kind: 'fields'; heading: string; fields: DetailField[] };

type Params = ToolCallDetail['params'];

const EMAIL_FIELDS = [
  'to', 'cc', 'subject', 'body', 'uid', 'operation', 'sender', 'keyword', 'date_from', 'date_to', 'attachments',
];

/** What a delegate was asked: its instructions, or a subtask's message. */
const DELEGATE_FIELDS = ['instructions', 'message'];

/** Past this many table cells a diff skips the line match. */
const MAX_DIFF_CELLS = 1_000_000;

const BUILDERS: Record<CallView, (detail: ToolCallDetail, kind: CallKind) => ToolDetailView> = {
  command: ({ params }) => ({ kind: 'command', command: text(params.command) }),
  edit: ({ params }) => ({ kind: 'edit', path: text(params.path), diff: lineDiff(text(params.search), text(params.replace)) }),
  write: ({ params, result }) => ({
    kind: 'write',
    path: text(params.path),
    contents: text(params.contents),
    diff: createdFile(result) ? splitLines(text(params.contents)).map(added) : [],
  }),
  read: ({ params }) => ({
    kind: 'read',
    source: text(params.source),
    lines: lineRange(text(params.start_line), text(params.end_line)),
  }),
  browser: ({ params, pages }, kind) => ({
    kind: 'browser',
    heading: text(params.action),
    fields: fieldsOf(params, kind.fields ?? []),
    pages,
  }),
  browse: ({ params, pages }) => ({ kind: 'browse', fields: fieldsOf(params, DELEGATE_FIELDS), pages }),
  delegate: ({ params }) => ({ kind: 'delegate', fields: fieldsOf(params, DELEGATE_FIELDS) }),
  email: ({ params }) => ({ kind: 'email', heading: text(params.action), fields: fieldsOf(params, EMAIL_FIELDS) }),
};

export function toolDetailView(detail: ToolCallDetail): ToolDetailView {
  const kind = callKind(detail);
  return kind.view ? BUILDERS[kind.view](detail, kind) : argumentsView(detail.params);
}

/** Every argument as a labelled field, `action` as the heading. */
function argumentsView({ action, ...rest }: Params): ToolDetailView {
  return { kind: 'fields', heading: text(action), fields: fieldsOf(rest, Object.keys(rest)) };
}

/** The named arguments that hold something, in the order named. */
function fieldsOf(params: Params, keys: readonly string[]): DetailField[] {
  return keys.filter((key) => !isEmpty(params[key])).map((key) => ({ label: labelOf(key), ...shown(params[key]) }));
}

function labelOf(key: string): string {
  return key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ');
}

function isEmpty(value: unknown): boolean {
  if (value == null || value === '') return true;
  return typeof value === 'object' && Object.keys(value).length === 0;
}

/** A value as text: a list of strings one per line, any other structure as
 *  pretty JSON. */
function shown(value: unknown): Pick<DetailField, 'value' | 'nested'> {
  if (typeof value !== 'object' || value === null) return { value: text(value), nested: false };
  if (Array.isArray(value) && value.every((v) => typeof v === 'string')) return { value: value.join('\n'), nested: false };
  return { value: JSON.stringify(value, null, 2), nested: true };
}

function text(value: unknown): string {
  if (value == null) return '';
  return typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
}

/** "lines 3–9" when a read named its end, "from line 3" when only its start. */
function lineRange(start: string, end: string): string {
  if (!end) return start ? `from line ${start}` : '';
  return `lines ${start || 1}–${end}`;
}

/** Whether a `write_file` result says the file was new. Its second line is
 *  the JSON body `{path, bytes, created}`; a line that is not JSON says no. */
function createdFile(result: string | null): boolean {
  const body = result?.split('\n')[1];
  if (!body) return false;
  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch {
    return false;
  }
  return typeof parsed === 'object' && parsed !== null && (parsed as { created?: unknown }).created === true;
}

function splitLines(value: string): string[] {
  return value === '' ? [] : value.replace(/\r?\n$/, '').split(/\r?\n/);
}

const added = (line: string): DiffLine => ({ op: 'add', text: line });
const removed = (line: string): DiffLine => ({ op: 'remove', text: line });

/** A line diff of `before` → `after`: shared lines as context, and where they
 *  differ the removed lines then the added ones. Past MAX_DIFF_CELLS every
 *  line of `before` reads as removed, then every line of `after` as added. */
export function lineDiff(before: string, after: string): DiffLine[] {
  const a = splitLines(before);
  const b = splitLines(after);
  if ((a.length + 1) * (b.length + 1) > MAX_DIFF_CELLS) return [...a.map(removed), ...b.map(added)];
  return walkDiff(a, b, lcsTable(a, b));
}

/** The longest-common-subsequence length of every pair of suffixes, row-major:
 *  cell `i * (b.length + 1) + j` holds it for `a[i..]` and `b[j..]`. */
function lcsTable(a: readonly string[], b: readonly string[]): Uint32Array {
  const w = b.length + 1;
  const table = new Uint32Array((a.length + 1) * w);
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      const cell = i * w + j;
      table[cell] = a[i] === b[j] ? table[cell + w + 1] + 1 : Math.max(table[cell + w], table[cell + 1]);
    }
  }
  return table;
}

function walkDiff(a: readonly string[], b: readonly string[], table: Uint32Array): DiffLine[] {
  const w = b.length + 1;
  const diff: DiffLine[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      diff.push({ op: 'same', text: a[i] });
      i += 1;
      j += 1;
    } else if (table[(i + 1) * w + j] >= table[i * w + j + 1]) {
      diff.push(removed(a[i]));
      i += 1;
    } else {
      diff.push(added(b[j]));
      j += 1;
    }
  }
  return [...diff, ...a.slice(i).map(removed), ...b.slice(j).map(added)];
}

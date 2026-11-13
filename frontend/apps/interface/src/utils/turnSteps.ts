/**
 * turnSteps — what a step's work says. A step is one row under what it said:
 * its thinking, folded to "Thought for 9s", then what its tool calls did,
 * "Wrote 2 files and read 1 file", folded over one row per call.
 */
import { reactive } from 'vue';
import type { ConversationMessage, ToolCallChip } from '../api/conversation';
import { callKind, type CallKind } from './toolCatalog';
import { formatElapsed } from './time';

export type StepCall = Pick<ToolCallChip, 'id' | 'tool_name' | 'params' | 'summary' | 'state' | 'delegate'>;

/** One step's work — a stored row's, or the calls running now. */
export interface StepWork {
  thinking?: ConversationMessage['thinking'];
  tool_calls?: StepCall[];
}

/** A call's one-line label: the summary the backend wrote, else the tool's name. */
export function callLabel(call: StepCall): string {
  return call.summary.trim() || call.tool_name;
}

/** What the thinking reads as: "Thought for Xs" when it was timed, "Thought"
 *  when it left traces but no time, else nothing. */
export function thought(thinking: StepWork['thinking']): string {
  if (!thinking) return '';
  if (thinking.duration_ms > 0) return `Thought for ${formatElapsed(thinking.duration_ms)}`;
  return thinking.traces.length ? 'Thought' : '';
}

/** Whether the step thought or called a tool — whether it has work to show. */
export function didWork(work: StepWork): boolean {
  return Boolean(thought(work.thinking) || work.tool_calls?.length);
}

/** The calls a step shows — all but the kinds the catalog hides. */
export function visibleCalls(calls: readonly StepCall[] = []): StepCall[] {
  return calls.filter((c) => !callKind(c).hidden);
}

/** Whether the step has work the person can see: it thought, or made a call
 *  the catalog does not hide. A step that only updated the plan has none. */
export function showsWork(work: StepWork): boolean {
  return Boolean(thought(work.thinking) || visibleCalls(work.tool_calls).length);
}

export interface CallsSummary {
  /** "Wrote 2 files and read 1 file". */
  text: string;
  /** How many of the shown calls failed. */
  failed: number;
}

interface KindCount {
  kind: CallKind;
  targets: Set<string>;
  untargeted: number;
}

const LIST = new Intl.ListFormat('en-GB', { type: 'conjunction' });

/** What a step's shown calls did: one phrase per kind, in the order each kind
 *  first ran. A kind with a target counts the distinct values it acted on; a
 *  call without one counts on its own. Past three kinds, the rest are "n more". */
export function summarizeCalls(shown: readonly StepCall[]): CallsSummary {
  const counts = new Map<string, KindCount>();
  for (const call of shown) {
    const kind = callKind(call);
    const count = counts.get(kind.key) ?? { kind, targets: new Set<string>(), untargeted: 0 };
    counts.set(kind.key, count);
    const targets = targetsOf(kind, call);
    targets.forEach((t) => count.targets.add(t));
    if (!targets.length) count.untargeted += 1;
  }
  const phrases = [...counts.values()].map((c) => countPhrase(c.kind, c.targets.size + c.untargeted));
  const listed = phrases.length > 3 ? [...phrases.slice(0, 3), `${phrases.length - 3} more`] : phrases;
  const text = LIST.format(listed);
  return { text: text.charAt(0).toUpperCase() + text.slice(1), failed: shown.filter((c) => c.state === 'error').length };
}

function targetsOf(kind: CallKind, call: StepCall): string[] {
  const value = kind.target ? call.params?.[kind.target] : undefined;
  return [value].flat().filter((v) => v != null && v !== '').map((v) => JSON.stringify(v));
}

function countPhrase(kind: CallKind, n: number): string {
  return n === 1 ? kind.one : kind.many.replace('{n}', String(n));
}

/** The calls whose list the user unfolded, by id. Module state, so a list
 *  stays open when its running step settles into its stored row and remounts:
 *  the stored row carries the same call ids the live pills did. */
const openCalls = reactive(new Set<number>());

export function callsOpen(calls: readonly StepCall[]): boolean {
  return calls.some((c) => openCalls.has(c.id));
}

export function setCallsOpen(calls: readonly StepCall[], open: boolean): void {
  for (const { id } of calls) {
    if (open) openCalls.add(id);
    else openCalls.delete(id);
  }
}

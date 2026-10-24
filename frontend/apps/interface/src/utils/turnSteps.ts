/**
 * turnSteps — what a step's work says. A step is one row: its thinking, folded
 * to "Thought for 9s", and its tool calls, one row each, above what it said.
 */
import type { ConversationMessage, ToolCallChip } from '../api/conversation';
import { formatElapsed } from './time';

export type StepCall = Pick<ToolCallChip, 'tool_name' | 'summary' | 'state' | 'delegate'>;

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

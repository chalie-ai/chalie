/**
 * toolCallDetails — which tool-call rows the user opened, and each call as
 * read on its own. Module state, so an open row stays open, and keeps what it
 * read, when its running step settles into its stored row and remounts.
 */
import { reactive } from 'vue';
import { conversation, type ToolCallDetail } from '../api/conversation';
import type { StepCall } from './turnSteps';

/** A call as its detail reads it: its own state and, when it started a
 *  subagent, whether that subagent still runs — undefined when it started
 *  none, or its run state is not known yet. */
type CallRef = Pick<StepCall, 'id' | 'state'> & { subagentRunning?: boolean };

const openRows = reactive(new Set<number>());

/** Successful reads only, by call id. */
const details = new Map<number, ToolCallDetail>();

/** The read under way for each call, the state it was made for, and whether
 *  it was made after the call's subagent ended. */
const reads = new Map<number, { state: StepCall['state']; ended: boolean; read: Promise<ToolCallDetail> }>();

export function detailOpen(id: number): boolean {
  return openRows.has(id);
}

export function toggleDetail(id: number): void {
  if (openRows.has(id)) openRows.delete(id);
  else openRows.add(id);
}

/** The call as last read, whatever state it was read in. */
export function lastDetail(id: number): ToolCallDetail | undefined {
  return details.get(id);
}

/** The call as read in its current state: from the cache while the cached
 *  read matches that state, else from the read under way for it, else read
 *  anew. On the chat a subagent outlives the call that started it, so its end
 *  changes what the call shows (a browsing call's pages) while the call's own
 *  state stays put: once the subagent has ended, a read made while it still
 *  worked is stale. A read is kept only if it succeeds and is still the
 *  newest. */
export function readDetail(call: CallRef): Promise<ToolCallDetail> {
  const ended = call.subagentRunning === false;
  const cached = details.get(call.id);
  if (cached?.state === call.state && !(ended && cached.delegate?.working)) return Promise.resolve(cached);
  const pending = reads.get(call.id);
  if (pending?.state === call.state && (pending.ended || !ended)) return pending.read;
  const read = conversation.toolCall(call.id);
  reads.set(call.id, { state: call.state, ended, read });
  const settle = (detail?: ToolCallDetail): void => {
    if (reads.get(call.id)?.read !== read) return;
    reads.delete(call.id);
    if (detail) details.set(call.id, detail);
  };
  // The caller handles the failure; this branch only clears the read.
  read.then(settle, () => settle());
  return read;
}

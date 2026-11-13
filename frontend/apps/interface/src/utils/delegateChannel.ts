/**
 * delegateChannel — the one place that knows how a delegate (subagent) turn's
 * transcript channel is spelled. Every other channel on the wire (`user`, a
 * scheduled channel, …) belongs to a ConfigType feed and is routed by type.
 */

const DELEGATE_CHANNEL_PREFIX = 'delegate:';

/** The channel a `subtask` call's child turn runs on — the only delegate
 *  channel whose calls outlive their caller's turn. */
export const SUBTASK_CHANNEL = `${DELEGATE_CHANNEL_PREFIX}subtask`;

/** True when `channel` is a delegate turn's transcript channel. */
export function isDelegateChannel(channel: unknown): channel is string {
  return typeof channel === 'string' && channel.startsWith(DELEGATE_CHANNEL_PREFIX);
}

/** The subagent a delegate channel belongs to (`delegate:web_search` → `web_search`). */
export function delegateName(channel: string): string {
  return channel.slice(DELEGATE_CHANNEL_PREFIX.length);
}

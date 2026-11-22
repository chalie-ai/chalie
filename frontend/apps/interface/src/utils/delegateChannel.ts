/**
 * delegateChannel — the one place that knows how a delegate (subagent) turn's
 * transcript channel is spelled, and what the person reads for each kind of
 * subagent. Every other channel on the wire (`user`, a scheduled channel, …)
 * belongs to a ConfigType feed and is routed by type.
 */
import { Bot, Globe, Mail, Search, SquareTerminal } from '@lucide/vue';
import type { LucideIcon } from '@lucide/vue';

const DELEGATE_CHANNEL_PREFIX = 'delegate:';

/** True when `channel` is a delegate turn's transcript channel. */
export function isDelegateChannel(channel: unknown): channel is string {
  return typeof channel === 'string' && channel.startsWith(DELEGATE_CHANNEL_PREFIX);
}

/**
 * The kind of subagent a delegate channel belongs to. Each subagent runs on a
 * channel of its own, numbered by the subagent (`delegate:web_search:12` →
 * `web_search`); a bare kind channel (`delegate:web_search`) reads the same.
 */
export function delegateName(channel: string): string {
  return channel.slice(DELEGATE_CHANNEL_PREFIX.length).replace(/:\d+$/, '');
}

/** What the person reads for a kind of subagent: its type in words, and its icon. */
export interface SubagentKind {
  name: string;
  icon: LucideIcon;
}

const SUBAGENT_KINDS = new Map<string, SubagentKind>([
  ['subtask', { name: 'Subtask', icon: Bot }],
  ['web_search', { name: 'Web search', icon: Search }],
  ['web_browse', { name: 'Web browsing', icon: Globe }],
  ['pim', { name: 'Mail & calendar', icon: Mail }],
  ['code_agent', { name: 'Coding', icon: SquareTerminal }],
]);

/**
 * The kind of the subagent `tool` names (`web_search` → Web search, a search
 * icon) — what the person reads in place of the tool name. A kind not listed
 * here yet reads as its tool name in words (`image_gen` → `Image gen`) under
 * the generic subagent icon.
 */
export function subagentKind(tool: string): SubagentKind {
  const known = SUBAGENT_KINDS.get(tool);
  if (known) return known;
  const words = tool.replaceAll('_', ' ');
  return { name: words.charAt(0).toUpperCase() + words.slice(1), icon: Bot };
}

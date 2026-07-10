/**
 * useConversationFeed — unit spec.
 *
 * Exercises the core buffer invariants against hand-built sample blocks.
 * No network: all assertions are on in-memory state driven by direct calls
 * to upsertTurn and setWorking.
 *
 * Pill mechanics themselves are tested in ../utils/liveActTrail.spec.ts;
 * here we only cover the feed→trail handoff (§6.5 step 4 reconcile).
 *
 * Because useConversationFeed is a module-level singleton we re-import a
 * fresh module per test via vi.resetModules() to get clean state — the
 * liveActTrail module must be re-imported alongside it so both sides of a
 * handoff test share the same module instance.
 */
import { describe, expect, it, vi } from 'vitest';
import type { ConversationTurnBlock } from '../api/conversation';

// ── Shared helpers ────────────────────────────────────────────────────────────

function block(
  turnId: number,
  messageIds: number[],
  opts: { working?: boolean; toolCalls?: number[] } = {},
): ConversationTurnBlock {
  return {
    turn_id: turnId,
    gist: null,
    preview: `turn ${turnId}`,
    last_activity_at: null,
    working: opts.working ?? false,
    duration_ms: 0,
    messages: messageIds.map((id, i) => ({
      id: String(id),
      role: i === 0 ? 'user' : 'assistant',
      content: `msg ${id}`,
      timestamp: '2026-01-01 00:00:00',
      turn_id: turnId,
      tool_calls: opts.toolCalls?.includes(id)
        ? [{ tool_name: 'search', summary: 'searched', state: 'done', ended_at: null }]
        : undefined,
    })),
  };
}

// Re-import a fresh singleton for each test so state is isolated. Returns the
// feed plus the SAME-registry liveActTrail module (the feed's default type is
// ConfigType.USER = 'user').
async function freshFeed() {
  vi.resetModules();
  // Stub the api module so upsertTurn calls never hit the network.
  vi.mock('../api/conversation', () => ({
    conversation: {
      thread: vi.fn(),
      threads: vi.fn(),
      batch: vi.fn(),
    },
  }));
  const { useConversationFeed } = await import('./useConversationFeed');
  const trail = await import('../utils/liveActTrail');
  return { feed: useConversationFeed(), trail, type: 'user' };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('upsertTurn — version guard', () => {
  it('accepts a newer block (higher max message id)', async () => {
    const { feed } = await freshFeed();
    feed.upsertTurn(block(1, [10, 11]));
    feed.upsertTurn(block(1, [10, 11, 12]));
    expect(feed.blocks[1]?.messages).toHaveLength(3);
  });

  it('drops a strictly stale block (lower max message id)', async () => {
    const { feed } = await freshFeed();
    feed.upsertTurn(block(1, [10, 11, 12]));
    feed.upsertTurn(block(1, [10, 11])); // older — must be dropped
    expect(feed.blocks[1]?.messages).toHaveLength(3);
  });

  it('re-applies an equal-version block (idempotent self-heal)', async () => {
    const { feed } = await freshFeed();
    feed.upsertTurn(block(1, [10, 11]));
    const same = block(1, [10, 11]);
    same.preview = 'updated preview';
    feed.upsertTurn(same);
    expect(feed.blocks[1]?.preview).toBe('updated preview');
  });
});

describe('upsertTurn — turn_id ordering in sortedBlocks', () => {
  it('sorts turns by ascending turn_id regardless of insertion order', async () => {
    const { feed } = await freshFeed();
    feed.upsertTurn(block(3, [30]));
    feed.upsertTurn(block(1, [10]));
    feed.upsertTurn(block(2, [20]));
    const ids = feed.sortedBlocks.value.map((b) => b.turn_id);
    expect(ids).toEqual([1, 2, 3]);
  });
});

describe('setWorking', () => {
  it('marks a turn working then clears it', async () => {
    const { feed } = await freshFeed();
    feed.setWorking(7, true);
    expect(feed.working[7]).toBe(true);
    feed.setWorking(7, false);
    expect(feed.working[7]).toBeUndefined();
  });
});

describe('feed → liveActTrail handoff (§6.5 step 4)', () => {
  it('upsertTurn with persisted tool_calls clears resolved live pills for that turn', async () => {
    const { feed, trail, type } = await freshFeed();
    trail.startLiveTool(type, 1, 42, 'search');
    trail.finishLiveTool(type, 1, 42, true);
    feed.upsertTurn(block(1, [10, 11], { toolCalls: [11] }));
    expect(trail.liveTrailsFor(type, 1)).toEqual([]);
  });

  it('upsertTurn without tool_calls leaves live pills untouched', async () => {
    const { feed, trail, type } = await freshFeed();
    trail.startLiveTool(type, 1, 42, 'search');
    feed.upsertTurn(block(1, [10, 11]));
    expect(trail.liveTrailsFor(type, 1)[0]?.pills).toHaveLength(1);
  });

  it('setWorking(false) purges the live trail for that turn', async () => {
    const { feed, trail, type } = await freshFeed();
    trail.startLiveTool(type, 7, 42, 'search');
    feed.setWorking(7, true);
    feed.setWorking(7, false);
    expect(trail.liveTrailsFor(type, 7)).toEqual([]);
  });
});

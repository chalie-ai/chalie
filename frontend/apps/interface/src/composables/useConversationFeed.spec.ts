/**
 * useConversationFeed — unit spec.
 *
 * Exercises the core buffer invariants against hand-built sample blocks.
 * No network: all assertions are on in-memory state driven by direct calls
 * to upsertTurn, startLiveTool, finishLiveTool, and setWorking.
 *
 * Because useConversationFeed is a module-level singleton we re-import a
 * fresh module per test via vi.resetModules() to get clean state.
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

// Re-import a fresh singleton for each test so state is isolated.
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
  return useConversationFeed();
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('upsertTurn — version guard', () => {
  it('accepts a newer block (higher max message id)', async () => {
    const feed = await freshFeed();
    feed.upsertTurn(block(1, [10, 11]));
    feed.upsertTurn(block(1, [10, 11, 12]));
    expect(feed.blocks[1]?.messages).toHaveLength(3);
  });

  it('drops a strictly stale block (lower max message id)', async () => {
    const feed = await freshFeed();
    feed.upsertTurn(block(1, [10, 11, 12]));
    feed.upsertTurn(block(1, [10, 11])); // older — must be dropped
    expect(feed.blocks[1]?.messages).toHaveLength(3);
  });

  it('re-applies an equal-version block (idempotent self-heal)', async () => {
    const feed = await freshFeed();
    feed.upsertTurn(block(1, [10, 11]));
    const same = block(1, [10, 11]);
    same.preview = 'updated preview';
    feed.upsertTurn(same);
    expect(feed.blocks[1]?.preview).toBe('updated preview');
  });
});

describe('upsertTurn — turn_id ordering in sortedBlocks', () => {
  it('sorts turns by ascending turn_id regardless of insertion order', async () => {
    const feed = await freshFeed();
    feed.upsertTurn(block(3, [30]));
    feed.upsertTurn(block(1, [10]));
    feed.upsertTurn(block(2, [20]));
    const ids = feed.sortedBlocks.value.map((b) => b.turn_id);
    expect(ids).toEqual([1, 2, 3]);
  });
});

describe('live-pill clear-on-upsert handoff (§6.5 step 4)', () => {
  it('removes a live trail for a turn once that turn has persisted tool_calls', async () => {
    const feed = await freshFeed();
    // Turn 1 has an in-flight live pill.
    feed.startLiveTool(1, 999, 'search', 'searching…');
    expect(feed.liveTools[1]).toBeDefined();

    // Upsert a block where message id 11 now carries tool_calls.
    feed.upsertTurn(block(1, [10, 11], { toolCalls: [11] }));

    // The live trail for turn 1 must be gone — summaries take over.
    expect(feed.liveTools[1]).toBeUndefined();
  });

  it('does not clear trails for a turn that has no persisted tool_calls yet', async () => {
    const feed = await freshFeed();
    feed.startLiveTool(1, 999, 'search');
    feed.upsertTurn(block(1, [10, 11])); // no message carries tool_calls
    expect(feed.liveTools[1]).toBeDefined();
  });
});

describe('startLiveTool / finishLiveTool', () => {
  it('adds an unresolved pill on startLiveTool', async () => {
    const feed = await freshFeed();
    feed.startLiveTool(1, 42, 'calendar', 'checking calendar');
    const pills = feed.liveTools[1];
    expect(pills).toHaveLength(1);
    expect(pills?.[0]).toMatchObject({ id: '42', name: 'calendar', resolved: false });
  });

  it('resolves the correct pill on finishLiveTool, preserving others', async () => {
    const feed = await freshFeed();
    feed.startLiveTool(1, 42, 'calendar');
    feed.startLiveTool(1, 43, 'weather');
    feed.finishLiveTool(1, 42, true);
    const pills = feed.liveTools[1] ?? [];
    expect(pills.find((p) => p.id === '42')?.resolved).toBe(true);
    expect(pills.find((p) => p.id === '43')?.resolved).toBe(false);
  });

  it('is a no-op for null callId', async () => {
    const feed = await freshFeed();
    expect(() => feed.finishLiveTool(1, null, true)).not.toThrow();
  });
});

describe('setWorking', () => {
  it('marks a turn working then clears it', async () => {
    const feed = await freshFeed();
    feed.setWorking(7, true);
    expect(feed.working[7]).toBe(true);
    feed.setWorking(7, false);
    expect(feed.working[7]).toBeUndefined();
  });

  it('clearing working also purges that turn live trails', async () => {
    const feed = await freshFeed();
    feed.startLiveTool(7, 1, 'search');
    feed.setWorking(7, false);
    expect(feed.liveTools[7]).toBeUndefined();
  });
});

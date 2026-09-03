// @vitest-environment happy-dom
/**
 * Permission-card lane resolution — the pure rule behind which surface a
 * pending gate's card renders in: a gate bubbles up to the channel its request
 * came from, for every tool. A main-spine turn's card is home on the spine; a
 * thread (forked) or scheduled turn's card is home inside the panel open on
 * that exact (turn_id, type) turn, and waits on the spine — labelled — until
 * that panel opens.
 *
 * Real `@chalie/shared` barrel for ConfigType (happy-dom because the barrel's
 * imports expect a window — the same way TurnView.spec.ts runs it).
 */
import { describe, expect, it } from 'vitest';
import { ConfigType } from '@chalie/shared';
import { originLabel, resolvePermissionLane } from './permissionLane';

const CLOSED = { panelThreadId: null, panelType: ConfigType.USER };
const openOn = (turnId: number, type: string) => ({ panelThreadId: turnId, panelType: type });

const mainSpine = { type: ConfigType.USER, turn_id: 5, forked: false };
const thread = { type: ConfigType.USER, turn_id: 7, forked: true };
const scheduled = { type: ConfigType.SCHEDULED, turn_id: 7, forked: false };

describe('resolvePermissionLane — main spine', () => {
  it('a main-spine turn is home on the spine, unlabelled, whatever the panel shows', () => {
    expect(resolvePermissionLane(mainSpine, CLOSED)).toEqual({ lane: 'spine', label: null });
    expect(resolvePermissionLane(mainSpine, openOn(5, ConfigType.USER))).toEqual({ lane: 'spine', label: null });
    expect(resolvePermissionLane(mainSpine, openOn(99, ConfigType.USER))).toEqual({ lane: 'spine', label: null });
  });

  it('a frame without an origin keeps today\'s spot: the spine, unlabelled', () => {
    expect(resolvePermissionLane(null, CLOSED)).toEqual({ lane: 'spine', label: null });
    expect(resolvePermissionLane(null, openOn(7, ConfigType.USER))).toEqual({ lane: 'spine', label: null });
  });
});

describe('resolvePermissionLane — thread (forked) turn', () => {
  it('waits on the spine, labelled "Thread", while its panel is closed', () => {
    expect(resolvePermissionLane(thread, CLOSED)).toEqual({ lane: 'spine', label: 'Thread' });
  });

  it('renders inside the panel open on that turn', () => {
    expect(resolvePermissionLane(thread, openOn(7, ConfigType.USER))).toEqual({ lane: 'panel', label: null });
  });

  it('stays on the spine, labelled, while the panel shows a different turn', () => {
    expect(resolvePermissionLane(thread, openOn(8, ConfigType.USER))).toEqual({ lane: 'spine', label: 'Thread' });
  });

  it('panel identity is the (turn_id, type) PAIR — the same id under another type is a different turn', () => {
    expect(resolvePermissionLane(thread, openOn(7, ConfigType.SCHEDULED))).toEqual({ lane: 'spine', label: 'Thread' });
  });
});

describe('resolvePermissionLane — scheduled turn', () => {
  it('waits on the spine, labelled "Scheduled task", while its panel is closed', () => {
    expect(resolvePermissionLane(scheduled, CLOSED)).toEqual({ lane: 'spine', label: 'Scheduled task' });
  });

  it('renders inside the panel open on that scheduled turn', () => {
    expect(resolvePermissionLane(scheduled, openOn(7, ConfigType.SCHEDULED))).toEqual({ lane: 'panel', label: null });
  });

  it('a user thread open under the same id is not its panel', () => {
    expect(resolvePermissionLane(scheduled, openOn(7, ConfigType.USER))).toEqual({ lane: 'spine', label: 'Scheduled task' });
  });
});

describe('originLabel', () => {
  it('names the surface the card belongs to', () => {
    expect(originLabel(thread)).toBe('Thread');
    expect(originLabel(scheduled)).toBe('Scheduled task');
  });
});

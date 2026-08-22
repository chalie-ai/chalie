/**
 * Permission-card lane resolution — which surface a pending permission card
 * renders in, decided from the gate's `origin` and the thread panel's identity.
 *
 * A gate bubbles up to the channel its request came from: a main-spine turn's
 * card sits in the spine stack; a thread (forked) or scheduled turn's card sits
 * inside the ThreadPanel while that panel is open on that exact turn, else in
 * the spine stack labelled with where it belongs (and an Open button, which the
 * caller wires). Pure — no store reads, no DOM — so the callers feed it the
 * reactive inputs and the lane re-resolves as the panel opens and closes.
 */
import { ConfigType } from '@chalie/shared';
import type { PermissionOrigin } from '../api/policies';

/** The thread panel's identity as the session store holds it (`panelThreadId`/`panelType`). */
export interface PanelIdentity {
  /** turn_id open in the panel, or null when the panel is closed. */
  panelThreadId: number | null;
  /** ConfigType of that turn — paired with the id, which is only unique PER TYPE. */
  panelType: string;
}

export interface PermissionLane {
  /** `spine`: the fixed stack above the main dock; `panel`: the stack inside the open ThreadPanel. */
  lane: 'spine' | 'panel';
  /** "Thread" / "Scheduled task" on a spine card whose turn lives elsewhere — the
   *  cue (and Open affordance) that the card is away from home; null otherwise. */
  label: string | null;
}

const SPINE_HOME: PermissionLane = { lane: 'spine', label: null };
const PANEL: PermissionLane = { lane: 'panel', label: null };

/** Human prefix for a card that is away from its turn. */
export function originLabel(origin: PermissionOrigin): string {
  return origin.type === ConfigType.SCHEDULED ? 'Scheduled task' : 'Thread';
}

export function resolvePermissionLane(
  origin: PermissionOrigin | null,
  panel: PanelIdentity,
): PermissionLane {
  // Nothing to route on — a frame without an origin keeps today's spot.
  if (origin == null) return SPINE_HOME;
  // A main-spine turn is home on the spine, whatever the panel shows.
  if (origin.type === ConfigType.USER && !origin.forked) return SPINE_HOME;
  // Thread or scheduled turn: with its panel open on the FULL (turn_id, type)
  // identity the card belongs inside it; otherwise it waits on the spine,
  // labelled, until the user opens (or the backend resolves) it.
  if (panel.panelThreadId === origin.turn_id && panel.panelType === origin.type) return PANEL;
  return { lane: 'spine', label: originLabel(origin) };
}

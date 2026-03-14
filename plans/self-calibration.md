# Self-Calibration — Future Work

**Status:** Not started — placeholder for when the need is clear.

## Context

We removed TopicStabilityRegulator (672 lines) and RoutingStabilityRegulator (596 lines) during the v1.0.1 service audit. Both ran on 24h cycles computing parameter adjustments, but the chain was broken — services only loaded configs at init, so the tuned values were never actually consumed. The `configs/generated/` directory never existed on disk.

The concept is sound: Chalie should be able to calibrate its own parameters over time based on observed outcomes. The implementation was wrong: self-referential loops with no external learning signal and no reload mechanism.

## Requirements for a Future Implementation

1. **Closed-loop verification** — Any self-tuning must measure whether adjustments actually improved the target metric, not just whether the adjustment was computed.

2. **Live reload** — Tuned parameters must be consumable without restarting the process. Either services poll for config changes, or a signal triggers reload.

3. **External learning signal** — Don't grade your own homework. Use signals from outside the tuning loop (user satisfaction proxies, routing decision reversals, memory retrieval relevance scores) rather than internal metrics that the tuner itself influences.

4. **Bounded and reversible** — Keep the good ideas from the old design: max adjustment per cycle, cooldown periods, hard bounds on all parameters, automatic revert if metrics degrade.

5. **Observable** — Expose calibration state via the observability API so we can see what's being tuned and whether it's helping.

## What To Calibrate

Candidates for self-calibration (when the system is mature enough):
- Topic classifier switch threshold and weights
- Mode router base scores and weight parameters
- Memory decay curves per category
- Cognitive triage confidence thresholds
- Autonomous action gate thresholds

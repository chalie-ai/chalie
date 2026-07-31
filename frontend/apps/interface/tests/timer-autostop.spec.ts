import { expect, test } from '@playwright/test';

// Feature test — TimerCard alarm auto-stop after 30s (post-migration behavior
// change). Drives the REAL hot path end-to-end against a real Chalie instance,
// exactly like turn.spec.ts: one entry point (send a message asking for a short
// timer), the configured LLM calls the real `timer` tool, the real backend
// anchors the rich segment on the tool_call row's timestamp and broadcasts the
// rich-media turn, and the real frontend renders it as a TimerCard. No mocks, no route
// interception — every assertion is a downstream effect of the real boot +
// turn + render + the component's own timers.
//
// Change under test: once a timer expires its alarm bleeps, but it now
// auto-silences ALARM_AUTO_STOP_MS (30s) after it starts — the card stays on
// "Done"; only the audio and the ring's attention pulse stop. The silence is
// surfaced as the `.timer-card--silenced` class (TimerCard.vue onExpire →
// silenceAlarm), the observable proxy for the audio having stopped (headless
// Web Audio is a no-op, so the DOM flag is the assertable downstream effect).
//
// Timing: with a 30s timer, onExpire fires either by natural countdown (LLM
// replied in <30s, card rendered while running) or synchronously on mount via
// the <=60s stale-reload grace window (LLM replied 30-90s in, timer already
// expired but recently). Both run the identical alarm + 30s auto-stop logic. A
// timer that went FULLY stale (>60s past expiry) renders "Ended" and never
// rings; asserting the time line reads "Done" (not "Ended") cleanly separates
// the real expiry path from that case, so a too-slow LLM fails fast and
// legibly rather than via a misleading 40s timeout.

test('an expired timer auto-silences its alarm ~30s later (card stays on Done)', async ({
  page,
}) => {
  test.setTimeout(180_000); // real LLM round-trip + up to 30s expiry + 30s auto-stop

  await page.goto('/');
  await expect(page.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });

  // One entry point: ask the real assistant to start a short timer. The LLM
  // calls the real `timer` tool (title + duration_seconds); the backend anchors
  // the segment and broadcasts the rich card on this user-facing turn.
  await page.locator('#chatInput').fill('Start a 30-second timer labelled Pasta.');
  await page.locator('button.btn-action--send').click();

  // The real turn renders a TimerCard. Target the newest card (.last()) so any
  // timer left in conversation history by an earlier run is ignored.
  const card = page.locator('.timer-card').last();
  await expect(card).toBeVisible({ timeout: 90_000 });

  const timeLine = card.locator('.timer-card__time');

  // onExpire ran: the card reaches "Done" (state 'done'). "Done" — not "Ended" —
  // confirms the real alarm path fired (a stale render shows "Ended" and never
  // rings). Covers both natural expiry (<=30s) and the grace-on-mount path.
  await expect(timeLine).toContainText('Done', { timeout: 45_000 });
  await expect(card).toHaveClass(/timer-card--done/);

  // The alarm is ringing: it has NOT auto-silenced yet.
  await expect(card).not.toHaveClass(/timer-card--silenced/);

  // The alarm auto-silences ~30s after onExpire. Measure the real elapsed time
  // to prove it is the deliberate 30s auto-stop, not an instant render.
  const t0 = Date.now();
  await expect(card).toHaveClass(/timer-card--silenced/, { timeout: 40_000 });
  const elapsedMs = Date.now() - t0;
  expect(elapsedMs).toBeGreaterThanOrEqual(27_000);
  expect(elapsedMs).toBeLessThanOrEqual(38_000);

  // The card STAYS on "Done" — auto-silence stops only the audio + ring pulse;
  // it does NOT transition the card to "Stopped".
  await expect(card).toHaveClass(/timer-card--done/);
  await expect(card).not.toHaveClass(/timer-card--stopped/);
  await expect(timeLine).toContainText('Done');
  await expect(timeLine).not.toContainText('Stopped');
});

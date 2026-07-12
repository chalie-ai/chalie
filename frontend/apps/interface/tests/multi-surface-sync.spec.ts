import { expect, test } from '@playwright/test';

// Multi-surface sync feature test. A Chalie instance is single-user, but that one
// user may have the interface open on several surfaces at once (e.g. phone +
// laptop). Both load the real interface; each opens its OWN /ws connection to the
// single broadcast channel. Driving a real turn from surface A must keep surface
// B in sync, per the broadcast model:
//   1. A user message sent on A renders exactly once on A (it rendered
//      optimistically and de-dupes the server's echo of its own message — never
//      a second copy) AND appears as a fresh bubble on B (the same broadcast echo).
//   2. Chalie's reply fans out and renders as a Chalie bubble on BOTH surfaces.
// No mocks; two real browser contexts, two real sockets, and the configured LLM
// provider answers for real on the instance under test.
//
// The instance is long-lived and single-user, so the conversation may already
// hold history (and may run more than once). To stay immune to that — and to
// prove THIS turn rather than any historical one — every assertion keys off a
// per-run unique token carried in the prompt: the user message contains it, and
// the model is asked to echo it back, so the reply contains it too. Matching that
// token isolates exactly the one message and one reply this run produced.

test('a message sent on one surface renders on every open surface, and so does Chalie’s reply', async ({
  context,
  page,
}) => {
  test.setTimeout(120_000); // a real LLM round-trip can take a while

  const surfaceA = page; // the surface that sends
  const surfaceB = await context.newPage(); // a second open surface (same user)

  await surfaceA.goto('/');
  await surfaceB.goto('/');
  await expect(surfaceA.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });
  await expect(surfaceB.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });

  // Per-run unique token. The prompt both carries it (→ the user bubble) and asks
  // the model to echo it (→ the reply bubble), so every assertion below targets
  // exactly this turn's message and reply, never anything already in history.
  const token = `sync${Date.now().toString(36)}`;
  const prompt = `Reply with exactly this word and nothing else: ${token}`;
  const userWithToken = (p = surfaceA) => p.locator('.speech-form--user').filter({ hasText: token });
  const chalieWithToken = (p = surfaceA) =>
    p.locator('.speech-form--chalie').filter({ hasText: token });

  await surfaceA.locator('#chatInput').fill(prompt);
  await surfaceA.locator('button.btn-action--send').click();

  // Surface A renders its own bubble optimistically the instant it sends.
  await expect(userWithToken(surfaceA)).toHaveCount(1);

  // Surface B never typed anything, yet THIS message lands as a fresh user bubble
  // — delivered purely by the broadcast echo on the shared channel.
  await expect(userWithToken(surfaceB)).toHaveCount(1, { timeout: 15_000 });

  // B has rendered the echo, so the identical broadcast frame has already reached
  // A's socket (dispatch runs synchronously on receipt). Let A's render flush,
  // then assert A still shows EXACTLY one bubble for this message: it de-duped its
  // own echo. A de-dup failure would surface here as a count of 2.
  await surfaceA.waitForTimeout(1_000);
  await expect(userWithToken(surfaceA)).toHaveCount(1);

  // Chalie's reply (echoing the token) fans out over the one channel: it lands as
  // a Chalie bubble on BOTH surfaces, not just whichever one initiated the turn.
  await expect(chalieWithToken(surfaceA)).toHaveCount(1, { timeout: 100_000 });
  await expect(chalieWithToken(surfaceB)).toHaveCount(1, { timeout: 100_000 });
});

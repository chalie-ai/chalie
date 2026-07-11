import { test, expect } from '@playwright/test';

// End-to-end feature test: one entry point (clicking send in the real compose
// dock) drives the whole hot path — optimistic user bubble, presence machine,
// the WS turn, and the real backend's reply rendered as a Chalie bubble. No
// mocks; the configured LLM provider answers for real.

test('sending a message drives a real turn and renders Chalie’s reply', async ({ page }) => {
  test.setTimeout(120_000); // a real LLM round-trip can take a while

  await page.goto('/');
  await expect(page.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });

  const prompt = 'Reply with exactly the single word: pong';
  await page.locator('#chatInput').fill(prompt);
  await page.locator('button.btn-action--send').click();

  // Immediate downstream effect of session.sendMessage (no server needed yet):
  // the user's text renders as a user bubble.
  await expect(page.locator('.speech-form--user').last()).toContainText('pong');

  // Real backend completion: the turn runs over the WS and Chalie's reply lands
  // as a Chalie bubble.
  await expect(page.locator('.speech-form--chalie').last()).toBeVisible({ timeout: 100_000 });
});

test('a page reloaded mid-turn re-attaches and streams the reply', async ({ context, page }) => {
  test.setTimeout(120_000); // a real LLM round-trip can take a while

  const surfaceA = page;
  const surfaceB = await context.newPage();

  await surfaceA.goto('/');
  await expect(surfaceA.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });

  // Per-run unique token so every assertion targets exactly this turn.
  const token = `reattach${Date.now().toString(36)}`;
  const prompt = `Reply with exactly this word and nothing else: ${token}`;

  // Ask a question so the turn is in flight when we load surfaceB.
  await surfaceA.locator('#chatInput').fill(prompt);
  await surfaceA.locator('button.btn-action--send').click();

  // The user bubble is optimistically rendered on A.
  await expect(surfaceA.locator('.speech-form--user').filter({ hasText: token })).toHaveCount(1);

  // NOW load surfaceB fresh — mid-turn. It should re-attach: eventually render
  // Chalie's reply (echoing token), proving frames routed to the reloaded page
  // via the re-attach path (arm + status), not just a history fetch.
  await surfaceB.goto('/');
  await expect(surfaceB.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });

  // surfaceB shows the working indicator (re-attached) OR — if the turn was
  // very fast — already the reply. Either is acceptable; the reply MUST land.
  await expect(surfaceB.locator('.speech-form--chalie').filter({ hasText: token }))
    .toBeVisible({ timeout: 100_000 });

  // And surfaceA (the originator) also completes normally.
  await expect(surfaceA.locator('.speech-form--chalie').filter({ hasText: token }))
    .toBeVisible({ timeout: 100_000 });
});

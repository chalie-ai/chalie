import { expect, test } from '@playwright/test';

// End-to-end feature test: one entry point (clicking send in the real compose
// dock) drives the whole hot path — optimistic user bubble, the WS turn, and
// the real backend's reply rendered as a Chalie bubble. No mocks; the
// configured LLM provider answers for real.

test('sending a message drives a real turn and renders Chalie’s reply', async ({ page }) => {
  test.setTimeout(120_000); // a real LLM round-trip can take a while

  await page.goto('/');
  await expect(page.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });

  const prompt = 'Reply with exactly the single word: pong';
  await page.locator('#chatInput').fill(prompt);
  await page.locator('button.btn-action--send').click();

  // Immediate downstream effect of session.sendMessage (no server needed yet):
  // the user's text renders as a user bubble.
  await expect(page.locator('.user-message').last()).toContainText('pong');

  // Real backend completion: the turn runs over the WS and Chalie's reply lands
  // as a Chalie bubble.
  await expect(page.locator('.speech-form--chalie').last()).toBeVisible({ timeout: 100_000 });
});

import { test, expect } from '@playwright/test';

// End-to-end feature test: one entry point (clicking send in the real compose
// dock) drives the whole hot path — optimistic user bubble, presence machine,
// the WS turn, and the real backend's reply rendered as a Chalie bubble. No
// mocks; the configured LLM provider answers for real.

test('sending a message drives a real turn and renders Chalie’s reply', async ({ page }) => {
  test.setTimeout(120_000); // a real LLM round-trip can take a while

  await page.goto('/next/');
  await expect(page.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });

  const prompt = 'Reply with exactly the single word: pong';
  await page.locator('#chatInput').fill(prompt);
  await page.locator('button.btn-action--send').click();

  // Immediate downstream effects of session.sendMessage (no server needed yet):
  //  - the user's text renders as a user bubble, and
  //  - the presence machine leaves its idle "Chalie" label for a working state.
  await expect(page.locator('.speech-form--user').last()).toContainText('pong');
  await expect(page.locator('.presence-label')).not.toHaveText('Chalie', { timeout: 10_000 });

  // Real backend completion: the turn runs over the WS and Chalie's reply lands
  // as a Chalie bubble, then the presence machine returns to rest.
  await expect(page.locator('.speech-form--chalie').last()).toBeVisible({ timeout: 100_000 });
  await expect(page.locator('.presence-label')).toHaveText('Chalie', { timeout: 100_000 });
});

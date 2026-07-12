import { expect, test } from '@playwright/test';

// Feature test — drives the REAL interface served at / by a real Chalie
// instance (authenticated via the global-setup login cookie). No mocks: every
// assertion is a downstream effect of the real boot path.

test('interface boots against the real backend and renders the shell', async ({ page }) => {
  await page.goto('/');

  // The loading overlay polls GET /ready and removes itself once the backend
  // answers ready. Its disappearance is the real downstream signal that the
  // typed ApiClient reached a live backend (LoadingOverlay.vue).
  await expect(page.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });

  // The real app shell is mounted directly by App.vue (no RouterView): the
  // conversation spine, the compose dock, and the presence bar are all present.
  await expect(page.locator('#conversationFeed.conversation-spine')).toBeVisible();
  await expect(page.locator('#chatInput')).toBeVisible();
  await expect(page.locator('.presence-bar')).toBeVisible();

  // Presence rests at the idle label ("Chalie") on a healthy connection — never
  // the "Connection lost" error label (presence.ts LABELS).
  await expect(page.locator('.presence-label')).not.toHaveText('Connection lost');
});

test('compose textarea accepts input', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });

  const input = page.locator('#chatInput');
  await input.fill('hello chalie');
  await expect(input).toHaveValue('hello chalie');
});

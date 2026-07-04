import { expect, test } from '@playwright/test';

test('backend readiness resolves through the typed ApiClient', async ({ page }) => {
  await page.goto('/');
  // LoadingOverlay polls GET /ready via the typed ApiClient and removes itself
  // once the backend reports ready — the real downstream signal of readiness.
  await expect(page.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });
});

test('unauthenticated API call is rejected (401)', async ({ page }) => {
  await page.goto('/');
  const status = await page.evaluate(async () => {
    const r = await fetch('/api/threads?limit=1', { credentials: 'omit' });
    return r.status;
  });
  expect(status).toBe(401);
});

import { test, expect } from '@playwright/test';

test('backend readiness resolves through the typed ApiClient', async ({ page }) => {
  await page.goto('/next/');
  await expect(page.getByTestId('ready')).toHaveText('ready', { timeout: 15_000 });
});

test('unauthenticated API call is rejected (401)', async ({ page }) => {
  await page.goto('/next/');
  const status = await page.evaluate(async () => {
    const r = await fetch('/conversation/recent?limit=1', { credentials: 'omit' });
    return r.status;
  });
  expect(status).toBe(401);
});

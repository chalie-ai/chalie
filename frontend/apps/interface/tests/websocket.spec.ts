import { test, expect } from '@playwright/test';

test('websocket connects and the status reflects it', async ({ page }) => {
  await page.goto('/next/');
  await expect(page.getByTestId('ws-status')).toHaveText('connected', { timeout: 15_000 });
});

test('a websocket handshake to /ws occurs on load', async ({ page }) => {
  const wsPromise = page.waitForEvent('websocket');
  await page.goto('/next/');
  const ws = await wsPromise;
  expect(ws.url()).toContain('/ws');
});

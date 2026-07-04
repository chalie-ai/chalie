import { expect, test } from '@playwright/test';

// The real interface opens a WebSocket to the backend on load (session store is
// the single WS owner). These assert the real handshake + a healthy connection.

test('a websocket handshake to /ws occurs on load', async ({ page }) => {
  const wsPromise = page.waitForEvent('websocket');
  await page.goto('/');
  const ws = await wsPromise;
  expect(ws.url()).toContain('/ws');
});

test('the connection stays healthy (presence never falls into the error state)', async ({
  page,
}) => {
  await page.goto('/');
  await expect(page.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });

  // On a live WS the presence machine never enters its 'error' state, whose
  // label is "Connection lost" (presence.ts). Give the watchdog a moment.
  await page.waitForTimeout(2_000);
  await expect(page.locator('.presence-dot')).not.toHaveAttribute('data-state', 'error');
  await expect(page.locator('.presence-label')).not.toHaveText('Connection lost');
});

import { expect, test } from '@playwright/test';

// Boots the real Brain SPA served by Chalie at /brain/ and proves the
// shell chrome + every top-level nav entry actually renders. Drives the real
// production serve route + Vue Router boot + auth gate (the storageState cookie
// from global-setup carries a live session, and the QA env has a provider, so
// providersOnly is off and panels are reachable). No mocks: the assertions are
// against the DOM the user sees.

const NAV_IDS = [
  'providers',
  'cognition',
  'scheduler',
  'lists',
  'capabilities',
  'policies',
  'skills',
  'mcp',
  'import-export',
] as const;

test.describe('Brain SPA — smoke', () => {
  test('boots at /brain/, redirects to providers, shell + nav render', async ({ page }) => {
    await page.goto('/brain/');

    // Router "/" → "/providers" redirect (default route).
    await expect(page).toHaveURL(/\/brain\/providers(\/|$|\?)/);

    // Shell grid chrome.
    await expect(page.locator('#appShell')).toBeVisible();
    await expect(page.locator('aside.sidebar')).toBeVisible();
    await expect(page.locator('#topbar')).toBeVisible();
    await expect(page.locator('main.main #panelRoot')).toBeVisible();

    // All 10 top-level nav items present (Cognition + System groups).
    for (const id of NAV_IDS) {
      await expect(page.locator(`[data-nav="${id}"]`)).toBeVisible();
    }

    // The Providers panel really rendered (not an empty router outlet).
    await expect(page.getByRole('heading', { name: 'LLM Providers' })).toBeVisible();
  });
});

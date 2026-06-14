import { test, expect } from '@playwright/test';

// Boots the real Brain SPA served by Chalie at /brain/ and proves the
// shell chrome + every top-level nav entry actually renders. Drives the real
// production serve route + Vue Router boot + auth gate (the storageState cookie
// from global-setup carries a live session, and the QA env has a provider, so
// providersOnly is off and panels are reachable). No mocks: the assertions are
// against the DOM the user sees.

const NAV_IDS = [
  'providers',
  'vision',
  'cognition',
  'scheduler',
  'lists',
  'documents',
  'capabilities',
  'policies',
  'skills',
  'mcp',
  'brain',
] as const;

test.describe('Brain SPA — smoke', () => {
  test('boots at /brain/, redirects to providers, shell + nav render', async ({ page }) => {
    await page.goto('/brain/');

    // Router "/" → "/providers" redirect (BrainView default).
    await expect(page).toHaveURL(/\/brain\/providers(\/|$|\?)/);

    // Shell grid chrome.
    await expect(page.locator('#appShell')).toBeVisible();
    await expect(page.locator('aside.sidebar')).toBeVisible();
    await expect(page.locator('#topbar')).toBeVisible();
    await expect(page.locator('main.main #panelRoot')).toBeVisible();

    // Brand wordmark.
    await expect(page.locator('.sidebar-brand .wordmark')).toHaveText('Chalie');
    await expect(page.locator('.sidebar-brand .wordmark-sub')).toHaveText('Brain');

    // All 11 top-level nav items present (Cognition + System groups).
    for (const id of NAV_IDS) {
      await expect(page.locator(`[data-nav="${id}"]`)).toBeVisible();
    }

    // The Providers panel really rendered (not an empty router outlet).
    await expect(page.getByRole('heading', { name: 'LLM Providers' })).toBeVisible();
  });
});

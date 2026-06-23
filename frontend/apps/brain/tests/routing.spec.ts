import { test, expect, type Page } from '@playwright/test';

// Verifies the SPA history-fallback contract end-to-end: a deep link typed into
// the address bar (and a hard reload of it) is served index.html by the backend
// /brain/ route, Vue Router boots, the auth gate passes, and the correct
// panel renders. This is the real production deep-link path — a 404 or a wrong
// panel on reload would fail loudly here.

/** Goto a deep route, assert its anchor, hard-reload, assert it survives. */
async function deepLinkSurvivesReload(
  page: Page,
  path: string,
  assertPanel: () => Promise<void>,
): Promise<void> {
  await page.goto(path);
  await assertPanel();
  await page.reload();
  await assertPanel();
}

test.describe('Brain SPA — routing & deep-link reload', () => {
  test('Lists deep link + reload', async ({ page }) => {
    await deepLinkSurvivesReload(page, '/brain/lists', async () => {
      await expect(page).toHaveURL(/\/brain\/lists(\/|$|\?)/);
      await expect(page.getByRole('heading', { name: 'Lists', exact: true })).toBeVisible();
    });
  });

  test('MCP deep link + reload', async ({ page }) => {
    await deepLinkSurvivesReload(page, '/brain/mcp', async () => {
      await expect(page).toHaveURL(/\/brain\/mcp(\/|$|\?)/);
      await expect(page.getByRole('heading', { name: 'MCP', exact: true })).toBeVisible();
    });
  });

  test('Skills deep link + reload renders skill cards', async ({ page }) => {
    await deepLinkSurvivesReload(page, '/brain/skills', async () => {
      await expect(page).toHaveURL(/\/brain\/skills(\/|$|\?)/);
      await expect(page.getByRole('heading', { name: 'Skills', exact: true })).toBeVisible();
      // Curated skills ship with the install → at least one card must render.
      await expect(page.locator('.skill-card').first()).toBeVisible();
    });
  });

  test('Brain panel deep link + reload', async ({ page }) => {
    await deepLinkSurvivesReload(page, '/brain/brain', async () => {
      await expect(page).toHaveURL(/\/brain\/brain(\/|$|\?)/);
      // Brain backup/restore panel — Import/Export actions are its anchor.
      await expect(page.locator('#panelRoot')).toContainText(/Export|Import|Backup/i);
    });
  });

  test('Nested route (cognition/memory) deep link + reload', async ({ page }) => {
    await deepLinkSurvivesReload(page, '/brain/cognition/memory', async () => {
      await expect(page).toHaveURL(/\/brain\/cognition\/memory(\/|$|\?)/);
      // CognitionView shell heading + the active sub-nav prove the route resolved…
      await expect(page.getByRole('heading', { name: 'Cognition' })).toBeVisible();
      await expect(page.locator('[data-nav="cognition"].active')).toBeVisible();
      await expect(page.locator('[data-sub="memory"].active')).toBeVisible();
      // …and MemorySub's own content proves the nested <RouterView> actually
      // rendered the child (not just that the route matched). The filter tabs
      // render only after a successful load — a silent child crash shows nothing,
      // a failed fetch shows "Failed to load data." instead — so the Episodes tab
      // being present AND active is a real end-to-end signal the panel mounted.
      await expect(page.locator('.records-controls .filter-tab.active')).toHaveText('Episodes');
    });
  });

  test('Unknown route falls through catch-all to providers', async ({ page }) => {
    await page.goto('/brain/this-route-does-not-exist');
    // Catch-all redirect → /providers (does not 404, does not white-screen).
    await expect(page).toHaveURL(/\/brain\/providers(\/|$|\?)/);
    await expect(page.getByRole('heading', { name: 'LLM Providers' })).toBeVisible();
  });
});

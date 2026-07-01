import { expect, test } from '@playwright/test';

// Rule 7: every frontend feature supports dark AND light. The theme store
// (packages/shared/src/stores/theme.ts) flips html[data-theme] and persists the
// choice to localStorage key 'chalie-theme', re-adopted on init(). This drives
// the real toggle button and a real reload — no mocking of storage or the DOM.

test.describe('Brain SPA — theme parity (Rule 7)', () => {
  test('toggle flips html[data-theme], persists across reload, shell renders in both', async ({
    page,
  }) => {
    await page.goto('/brain/providers');
    await expect(page.locator('aside.sidebar')).toBeVisible();

    const html = page.locator('html');
    const initial = await html.getAttribute('data-theme');
    expect(initial === 'dark' || initial === 'light').toBe(true);
    const flipped = initial === 'dark' ? 'light' : 'dark';

    // Toggle via the real sidebar-footer button.
    await page.locator('button.theme-toggle').click();
    await expect(html).toHaveAttribute('data-theme', flipped);
    // Shell still fully rendered in the flipped theme.
    await expect(page.locator('aside.sidebar')).toBeVisible();
    await expect(page.locator('#topbar')).toBeVisible();

    // localStorage caught the choice…
    await expect
      .poll(() => page.evaluate(() => localStorage.getItem('chalie-theme')))
      .toBe(flipped);

    // …and init() re-adopts it on a hard reload (no flash back to default).
    await page.reload();
    await expect(html).toHaveAttribute('data-theme', flipped);
    await expect(page.locator('aside.sidebar')).toBeVisible();

    // Toggle back → returns to the original theme, shell still renders.
    await page.locator('button.theme-toggle').click();
    await expect(html).toHaveAttribute('data-theme', initial!);
    await expect(page.locator('aside.sidebar')).toBeVisible();
  });
});

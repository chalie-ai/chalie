import { test, expect } from '@playwright/test';

// Theme parity (Rule 7) — the real PresenceBar theme toggle drives html
// [data-theme], persists to localStorage (key 'chalie-theme'), and the shell
// renders in BOTH themes. All against the real interface served at /next/.

test('theme toggles dark↔light via the real toggle and persists across reload', async ({
  page,
}) => {
  await page.goto('/next/');
  await expect(page.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });

  // Force a known starting point, then reload so the pre-paint bootstrap + the
  // theme store init() both adopt it from localStorage.
  await page.evaluate(() => localStorage.setItem('chalie-theme', 'dark'));
  await page.reload();
  await expect(page.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  const darkBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);

  // Click the real toggle (PresenceBar) — flips the store + html + storage.
  await page.locator('button.theme-toggle').click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await expect
    .poll(() => page.evaluate(() => localStorage.getItem('chalie-theme')))
    .toBe('light');

  // The shell must remain rendered and usable in light theme (Rule 7).
  await expect(page.locator('#chatInput')).toBeVisible();
  await expect(page.locator('.presence-bar')).toBeVisible();

  const lightBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  // Inequality only — body has a long background-color transition, so this read
  // may land mid-interpolation. Do NOT harden to an exact value or it will flake.
  expect(darkBg).not.toBe(lightBg);

  // Persistence: a reload keeps the chosen theme (storage → pre-paint bootstrap).
  await page.reload();
  await expect(page.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
});

import { test, expect } from '@playwright/test';

test('boots and renders the Vue foundation home', async ({ page }) => {
  await page.goto('/next/');
  const home = page.getByTestId('home');
  await expect(home).toBeVisible();
  await expect(home).toContainText('Vue foundation');
});

test('theme toggles dark↔light and persists across reload', async ({ page }) => {
  await page.goto('/next/');
  await page.evaluate(() => localStorage.setItem('chalie-theme', 'dark'));
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  const darkBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);

  await page.getByTestId('toggle-theme').click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await expect(page.getByTestId('theme')).toHaveText('light');
  const lightBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  // Inequality only — body has a 420ms background-color transition, so this read
  // may land mid-interpolation. Do NOT harden to an exact value or it will flake.
  expect(darkBg).not.toBe(lightBg);

  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
});

test('base components render and accept input', async ({ page }) => {
  await page.goto('/next/');
  await expect(page.getByTestId('toggle-theme')).toBeVisible();
  const note = page.getByTestId('note').locator('input');
  await note.fill('hello');
  await expect(note).toHaveValue('hello');
});

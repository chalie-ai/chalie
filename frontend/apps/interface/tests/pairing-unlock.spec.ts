import { expect, test } from '@playwright/test';

// Feature tests — drive a REAL Chalie instance (CHALIE_BASE_URL). No mocks:
// every assertion is a downstream effect of the real auth gate (router.ts),
// the real /auth/status + /auth/login endpoints, and real localStorage.
//
// The pairing gate only fires on the Tauri runtime, detected via the
// `__TAURI_INTERNALS__` IPC bridge Tauri always injects; in a browser harness we
// honestly emulate that by injecting it before any script runs. The token
// accessors read localStorage key 'chalie_access_token', so we drive the
// "no token" / "has token" states by writing that real key.

const USERNAME = process.env.CHALIE_TEST_USERNAME ?? 'admin';

test.describe('Tauri pairing gate', () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test('Tauri + no token redirects to /pairing/ (not /login/)', async ({ page }) => {
    await page.addInitScript(() => {
      (window as unknown as { __TAURI_INTERNALS__: object }).__TAURI_INTERNALS__ = {};
    });
    await page.goto('/');
    await expect(page).toHaveURL(/\/pairing\//, { timeout: 15_000 });
    await expect(page.locator('.link-device__scan')).toBeVisible();
  });

  test('web (no Tauri runtime) + no session still redirects to /login/', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL(/\/login\//, { timeout: 15_000 });
  });
});

test.describe('UnlockVault overlay', () => {
  // Needs a real account password to submit the unlock form; skipped where that isn't configured.
  test.skip(!process.env.CHALIE_TEST_PASSWORD, 'needs CHALIE_TEST_PASSWORD');

  test('wrong password keeps the overlay and shows an error when the vault is locked', async ({ page }) => {
    await page.addInitScript((u) => {
      localStorage.setItem('chalie_username', u as string);
    }, USERNAME);
    await page.goto('/');

    const status = await page.request.get('/auth/status');
    const locked = (await status.json()).vault_state === 'locked';
    // This case only applies while the vault is locked; skip if this environment's vault is already open.
    test.skip(!locked, 'vault already unlocked in this env');

    await expect(page.locator('.unlock-vault')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByLabel('Username', { exact: true })).toHaveCount(0);
    await page.getByLabel('Password', { exact: true }).fill('definitely-wrong-pw');
    await page.locator('.unlock-vault__submit').click();

    await expect(page.locator('.unlock-vault__error')).toHaveText('Invalid password.', { timeout: 10_000 });
    await expect(page.locator('.unlock-vault')).toBeVisible();
  });
});

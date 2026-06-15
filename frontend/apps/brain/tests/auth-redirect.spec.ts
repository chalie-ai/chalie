import { test, expect } from '@playwright/test';

// Proves the NEW auth contract: the shared ApiClient owns the 401 redirect.
// Previously each Brain call was wrapped in withAuth(); now a 401 on any
// authenticated call hard-navigates to /login/?next=<path> from inside the
// client itself (ApiClient.fail401 → onAuthError → window.location.replace).
//
// No mocks. We boot the real Lists panel with a live session, then drop the
// session cookie so the backend genuinely 401s the next request, then fire a
// real data mutation (POST /api/lists via the create modal). Crucially this is
// an IN-PANEL action, not a route change — the router auth gate only runs on
// navigation, so the redirect we observe can ONLY have come from the shared
// ApiClient. A break in the client's 401 handling fails this loudly: the
// create would error in-place and the URL would stay on /brain/lists.

test.describe('Brain SPA — shared ApiClient owns the 401 redirect', () => {
  test('a data mutation on an expired session redirects to /login/?next=', async ({ page }) => {
    // Live session (storageState from global-setup): the panel loads for real.
    await page.goto('/brain/lists');
    await expect(page.getByRole('heading', { name: 'Lists', exact: true })).toBeVisible();

    // Simulate real session expiry — no mock, the cookie is genuinely gone, so
    // the backend will return 401 for the next authenticated request.
    await page.context().clearCookies();

    // Fire an authenticated mutation WITHOUT navigating: open the local modal
    // (no fetch) and submit it (POST /api/lists). The POST 401s → the shared
    // ApiClient redirects.
    await page.locator('.panel-header').getByRole('button', { name: 'New List' }).click();
    const nameInput = page.locator('#newListName');
    await expect(nameInput).toBeVisible();
    await nameInput.fill(`pw-authredirect-${Date.now()}`);
    await page.locator('.modal').getByRole('button', { name: 'Create' }).click();

    // Downstream signal: the client hard-navigated to /login/, preserving the
    // brain path as ?next=. This is the centralised behaviour the refactor
    // introduced — the only code that performs this redirect is ApiClient.
    await expect(page).toHaveURL(/\/login\/\?next=/, { timeout: 10_000 });
    // And the preserved destination is the brain path we were on.
    expect(new URL(page.url()).searchParams.get('next')).toContain('/brain/');
  });
});

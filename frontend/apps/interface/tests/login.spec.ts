import { test, expect } from '@playwright/test';

// Feature test — drives the REAL login page served at /login/ by a real Chalie
// instance. No mocks: every assertion is a downstream effect of the real
// backend auth endpoints and the Vue SFC's client-side logic.
//
// The global storageState is an authed cookie (set by global-setup.ts), but
// login must start logged-out, so we override per-file with an empty state.
test.use({ storageState: { cookies: [], origins: [] } });

// Username default, shared by every case (matches global-setup.ts:16).
const USERNAME = process.env.CHALIE_TEST_USERNAME ?? 'admin';

test('renders the login card with correct heading and inputs', async ({ page }) => {
  await page.goto('/login/');

  // The heading and subtext are rendered directly from src/login/LoginPage.vue
  // template lines 52-53. Their presence confirms the Vue app mounted.
  await expect(page.locator('.login-card h1')).toHaveText('Welcome back');
  await expect(page.locator('#username')).toBeVisible();
  await expect(page.locator('#password')).toBeVisible();
});

test('wrong password shows "Invalid credentials." and stays on /login/', async ({ page }) => {
  await page.goto('/login/');

  await page.locator('#username').fill(USERNAME);
  await page.locator('#password').fill('definitely-wrong-pw');
  await page.locator('button[type="submit"]').click();

  // The real backend returns HTTP 401; ApiClient throws AuthError; the SFC
  // catches it and sets errorMsg = "Invalid credentials." (src/login/LoginPage.vue:39).
  // .login-error is always in the DOM (min-height ensures it doesn't reflow);
  // we assert its text content, which is the real downstream signal.
  await expect(page.locator('.login-error')).toHaveText('Invalid credentials.', { timeout: 10_000 });

  // The page must NOT have navigated away — the Vue handler did not call
  // window.location.replace() on error.
  await expect(page).toHaveURL(/\/login\//);
});

test('valid credentials redirect to / and the session is established', async ({ page }) => {
  // Guard: this case requires valid credentials from the environment.
  test.skip(!process.env.CHALIE_TEST_PASSWORD, 'needs CHALIE_TEST_PASSWORD');

  await page.goto('/login/');

  await page.locator('#username').fill(USERNAME);
  await page.locator('#password').fill(process.env.CHALIE_TEST_PASSWORD!);
  await page.locator('button[type="submit"]').click();

  // On success the SFC calls window.location.replace(getNextDest()), which
  // returns '/' with no ?next (src/login/LoginPage.vue:35, getNextDest 7-10).
  // Anchor to the ORIGIN root so a stuck /login/ (which also ends in '/')
  // cannot satisfy the match — the redirect must actually have happened.
  await expect(page).toHaveURL(/^https?:\/\/[^/]+\/(\?|$)/, { timeout: 15_000 });

  // Downstream signal: the real session cookie was issued by /auth/login and
  // the browser sent it back on the next request. /auth/status confirms it.
  const status = await page.request.get('/auth/status');
  expect((await status.json()).has_session).toBe(true);
});

test('?next=/brain/ is honoured — valid login redirects to /brain/', async ({ page }) => {
  test.skip(!process.env.CHALIE_TEST_PASSWORD, 'needs CHALIE_TEST_PASSWORD');

  await page.goto('/login/?next=/brain/');

  await page.locator('#username').fill(USERNAME);
  await page.locator('#password').fill(process.env.CHALIE_TEST_PASSWORD!);
  await page.locator('button[type="submit"]').click();

  // getNextDest() returns '/brain/' when ?next=/brain/ is present and starts
  // with '/'; window.location.replace('/brain/') is then called
  // (src/login/LoginPage.vue:7-10,35).
  // The brain router may further redirect to /brain/providers on a bare env —
  // matching /brain/ as a substring is robust against that.
  await expect(page).toHaveURL(/\/brain\//, { timeout: 15_000 });
});

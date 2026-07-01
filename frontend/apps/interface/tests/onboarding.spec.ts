import { expect, request, test } from '@playwright/test';

// Feature test — drives the REAL onboarding page served at /on-boarding/ by a
// real Chalie instance that has NO master account yet (fresh env). No mocks:
// every assertion is a downstream effect of the real backend auth endpoints and
// the Vue SFC's client-side logic.
//
// The global storageState is an authed cookie (set by global-setup.ts), but
// onboarding is pre-account and must start logged-out.
test.use({ storageState: { cookies: [], origins: [] } });

// Credentials for the registration step. Fixed so the serial steps that follow
// can rely on the same account existing.
const REG_USERNAME = 'onboard-e2e';
const REG_PASSWORD = 'onboarding-pw-123';

test.describe.serial('onboarding flow', () => {
  // Capture the account state ONCE, before the suite mutates it. The skip-guard
  // below references this stable initial value — NOT a fresh /auth/status each
  // time — so the post-registration steps (which REQUIRE the account that the
  // "register" step creates) are never skipped by their own side effect.
  let initialHasAccount = false;

  test.beforeAll(async () => {
    const baseURL = process.env.CHALIE_BASE_URL;
    if (!baseURL) throw new Error('CHALIE_BASE_URL must point at a running Chalie instance');
    const ctx = await request.newContext({ baseURL });
    const res = await ctx.get('/auth/status');
    initialHasAccount = (await res.json()).has_master_account === true;
    await ctx.dispose();
  });

  test.beforeEach(() => {
    // Onboarding can only be exercised on a fresh, no-account instance. If the
    // env already had an account when the suite started, skip cleanly — the
    // pre-mount gate would redirect every visit to /login/.
    test.skip(
      initialHasAccount,
      'onboarding requires a fresh no-account instance; this env already has a master account',
    );
  });

  test('renders "Create Master Account" and the critical warning', async ({ page }) => {
    await page.goto('/on-boarding/');

    // src/onboarding/OnboardingPage.vue:94 — h1 "Create Master Account".
    // Presence confirms the Vue app mounted (the pre-mount gate did NOT redirect).
    await expect(page.locator('h1')).toHaveText('Create Master Account');

    // The warning-box h3 opens with "⚠ Critical — Read Before Continuing"
    // (src/onboarding/OnboardingPage.vue:99) — the real signal the account phase rendered.
    await expect(page.locator('.warning-box h3')).toContainText('⚠ Critical — Read Before Continuing');
  });

  test('client validation fires before any network request', async ({ page }) => {
    await page.goto('/on-boarding/');

    // Case 1: password too short (< 8). showToast('Password must be at least 8
    // characters', 'error') fires client-side (src/onboarding/OnboardingPage.vue:45-47)
    // BEFORE any /auth/register request is made.
    await page.locator('#accountUsername').fill('anyuser');
    await page.locator('#accountPassword').fill('short');
    await page.locator('#accountConfirmPassword').fill('short');
    await page.locator('button[type="submit"]').click();
    await expect(
      page.locator('.toast', { hasText: 'Password must be at least 8 characters' }),
    ).toBeVisible({ timeout: 5_000 });

    // Case 2: password mismatch. handleAccountSubmit returns before register
    // (src/onboarding/OnboardingPage.vue:49-51). The text-filtered locator targets
    // this toast specifically, so a lingering first toast does not interfere.
    await page.locator('#accountPassword').fill('longenough1');
    await page.locator('#accountConfirmPassword').fill('different1');
    await page.locator('button[type="submit"]').click();
    await expect(
      page.locator('.toast', { hasText: 'Passwords do not match' }),
    ).toBeVisible({ timeout: 5_000 });
  });

  test('register reveals the voice phase, then Skip lands on /brain/', async ({ page }) => {
    // The voice phase exists only in the SAME page session immediately after a
    // successful registration: once the account exists, /on-boarding/ redirects
    // to /login/. So registration and Skip must be one continuous flow.
    await page.goto('/on-boarding/');

    await page.locator('#accountUsername').fill(REG_USERNAME);
    await page.locator('#accountPassword').fill(REG_PASSWORD);
    await page.locator('#accountConfirmPassword').fill(REG_PASSWORD);
    await page.locator('button[type="submit"]').click();

    // POST /auth/register → 201 (sets the session cookie); the SFC sets
    // phase='voice' (src/onboarding/OnboardingPage.vue:58), which v-if swaps in
    // the voice-phase card. The heading is the real downstream signal.
    await expect(page.locator('h1')).toHaveText('Voice Support', { timeout: 15_000 });

    // skipVoice() → window.location.replace('/brain/') (src/onboarding/OnboardingPage.vue:85).
    // The session cookie from register lets the /brain/ serve-gate pass; a bare
    // env has no LLM provider, so the brain router forces /brain/providers under
    // its providersOnly lock — matching /brain/ as a substring is robust to that.
    await page.locator('button', { hasText: 'Skip' }).click();
    await expect(page).toHaveURL(/\/brain\//, { timeout: 15_000 });
  });

  test('after the account exists, /on-boarding/ redirects to /login/', async ({ page }) => {
    // The previous step registered the account. onboarding/main.ts's pre-mount
    // gate now sees has_master_account=true → window.location.replace('/login/')
    // before Vue mounts. A fresh (cookie-less) context still redirects because
    // the gate keys off the account existing, not the session.
    await page.goto('/on-boarding/');
    await expect(page).toHaveURL(/\/login\//, { timeout: 10_000 });
  });
});

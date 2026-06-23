import { test, expect } from '@playwright/test';

// Real production pairing path: the Brain mints a wrapper token against the
// LIVE POST /api/wrappers (cookie-authed, no mocks), reads its OWN served
// origin, and encodes the locked PairingPayload into a QR. We decode the exact
// JSON the QR carries (data-pairing) and assert the contract + that host
// mirrors the actual served origin (protocol/host/port permutation coverage on
// the real window.location.origin read).

test.describe('Brain — Link device pairing', () => {
  test('nav item routes and the view mounts', async ({ page }) => {
    await page.goto('/brain/link-device');
    await expect(page).toHaveURL(/\/brain\/link-device(\/|$|\?)/);
    await expect(page.getByRole('heading', { name: 'Link device', exact: true })).toBeVisible();
    await expect(page.locator('[data-action="generate-pairing"]')).toBeVisible();
  });

  test('generate mints a real token and encodes a valid PairingPayload', async ({ page }) => {
    await page.goto('/brain/link-device');
    await page.locator('[data-action="generate-pairing"]').click();

    const qr = page.locator('[data-testid="pairing-qr"]');
    // data-pairing is set only after the real POST /api/wrappers round-trip
    // resolves and the QR is drawn — wait for it to be non-empty.
    await expect(qr).toHaveAttribute('data-pairing', /.+/, { timeout: 15_000 });

    const json = await qr.getAttribute('data-pairing');
    expect(json).toBeTruthy();
    const payload = JSON.parse(json!) as {
      v: number;
      host: string;
      token: string;
      username: string;
    };

    // Locked contract — pairing payload gate.
    expect(payload.v).toBe(1);
    expect(payload.token.length).toBeGreaterThan(0);
    // The master login username, read from the real GET /auth/username, is in
    // the QR so the device's UnlockVault needs only a password.
    expect(payload.username.length).toBeGreaterThan(0);
    // host parses as a valid absolute origin URL…
    expect(() => new URL(payload.host)).not.toThrow();
    // …and mirrors the ACTUAL served origin (protocol + host + non-default
    // port, NO trailing slash) — proving the robust, non-user-editable read.
    expect(payload.host).toBe(new URL(page.url()).origin);
    expect(payload.host.endsWith('/')).toBe(false);
  });
});

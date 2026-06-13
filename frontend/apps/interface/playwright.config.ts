import { defineConfig } from '@playwright/test';

const BASE = process.env.CHALIE_BASE_URL;
if (!BASE) throw new Error('CHALIE_BASE_URL must point at a running Chalie instance');

export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  globalSetup: './tests/global-setup.ts',
  // retain-on-failure (not on-first-retry): retries default to 0 — we want the
  // trace on the very first failure, and retries would mask real flakiness.
  use: { baseURL: BASE, storageState: '.auth/state.json', trace: 'retain-on-failure' },
  reporter: [['list'], ['html', { open: 'never' }]],
});

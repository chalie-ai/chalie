import { request } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';

// Two supported preconditions:
//
//   1. Fresh env (no master account): /auth/status returns has_master_account=false.
//      We write an EMPTY storage state so the spec suite can start without any
//      session cookie. Specs like onboarding.spec.ts that need a no-account env
//      rely on this path.
//
//   2. Normal env (account exists): we POST /auth/login and persist the resulting
//      session cookie as .auth/state.json. A genuine login failure in this branch
//      still throws — it is a real error, not an expected precondition.
export default async function globalSetup(): Promise<void> {
  const baseURL = process.env.CHALIE_BASE_URL!;
  const username = process.env.CHALIE_TEST_USERNAME ?? 'admin';
  const password = process.env.CHALIE_TEST_PASSWORD;
  if (!password) throw new Error('CHALIE_TEST_PASSWORD must be set for the auth fixture');

  const ctx = await request.newContext({ baseURL });

  // Probe the instance before attempting login.
  const statusRes = await ctx.get('/auth/status');
  const status = await statusRes.json();

  mkdirSync('.auth', { recursive: true });

  if (status.has_master_account === false) {
    // Fresh env — no account to log into. Write an empty storage state so the
    // default storageState in playwright.config.ts resolves to a valid file
    // with no cookies. Individual specs that need auth will override this.
    writeFileSync('.auth/state.json', JSON.stringify({ cookies: [], origins: [] }));
    await ctx.dispose();
    return;
  }

  // Account exists — log in and persist the session cookie.
  const res = await ctx.post('/auth/login', { data: { username, password } });
  if (!res.ok()) throw new Error(`Login failed: ${res.status()} ${await res.text()}`);
  await ctx.storageState({ path: '.auth/state.json' });
  await ctx.dispose();
}

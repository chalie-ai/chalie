import { request } from '@playwright/test';
import { mkdirSync } from 'node:fs';

export default async function globalSetup(): Promise<void> {
  const baseURL = process.env.CHALIE_BASE_URL!;
  const username = process.env.CHALIE_TEST_USERNAME ?? 'admin';
  const password = process.env.CHALIE_TEST_PASSWORD;
  if (!password) throw new Error('CHALIE_TEST_PASSWORD must be set for the auth fixture');
  const ctx = await request.newContext({ baseURL });
  const res = await ctx.post('/auth/login', { data: { username, password } });
  if (!res.ok()) throw new Error(`Login failed: ${res.status()} ${await res.text()}`);
  mkdirSync('.auth', { recursive: true });
  await ctx.storageState({ path: '.auth/state.json' });
  await ctx.dispose();
}

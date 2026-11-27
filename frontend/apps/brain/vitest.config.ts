import { defineConfig } from 'vitest/config';

export default defineConfig({
  // `tests/` holds the Playwright suite run against a live server; only the
  // specs under `src/` are vitest's.
  test: {
    environment: 'node',
    include: ['src/**/*.spec.ts'],
  },
});

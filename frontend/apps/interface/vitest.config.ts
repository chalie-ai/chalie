import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vitest/config';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  // The vue() plugin only transforms .vue SFC imports — it does not change how
  // any existing plain-.ts spec runs. Default environment stays 'node'; specs
  // that mount a component opt into a DOM via a per-file
  // `// @vitest-environment happy-dom` docblock instead of a global flip, so
  // the existing node-environment specs (which build their own DOM stand-ins,
  // see session.spec.ts) are untouched.
  plugins: [vue()],
  resolve: {
    alias: {
      '@chalie/shared': fileURLToPath(new URL('../../packages/shared/src', import.meta.url)),
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.spec.ts'],
  },
});

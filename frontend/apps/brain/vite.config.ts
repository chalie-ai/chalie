import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

// Served by Flask at /brain/ (canonical, post-cutover).
export default defineConfig({
  base: process.env.VITE_BASE ?? '/brain/',
  plugins: [vue()],
  resolve: {
    alias: {
      // Alias to the package SOURCE DIR (not index.ts): a file-pointing string
      // alias prefix-matches subpaths, so `@chalie/shared/styles/main.scss` would
      // become `…/index.ts/styles/main.scss`. Pointing at the dir lets the bare
      // import resolve via directory-index (→ src/index.ts) while subpath imports
      // (the SCSS theme) resolve as real files (→ src/styles/main.scss).
      '@chalie/shared': fileURLToPath(new URL('../../packages/shared/src', import.meta.url)),
    },
  },
  css: {
    // Modern Sass compiler API — silences the Dart Sass legacy-js-api deprecation
    // (which becomes a hard error at Dart Sass 2.0).
    preprocessorOptions: { scss: { api: 'modern-compiler' } },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});

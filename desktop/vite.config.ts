import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

// The setup wizard is the only page this app bundles; Tauri embeds ./dist as its
// frontendDist. Everything after setup is served by the Chalie server itself.
//
// Its look is the shared foundation in ../frontend/packages/shared — the stylesheet
// and the Chalie mark — imported from source.
export default defineConfig({
  plugins: [vue()],
  clearScreen: false,
  resolve: {
    // Bare imports inside those shared files resolve from this project rather than
    // next to the files: one copy of Vue on the page, and fonts that need nothing
    // installed under ../frontend.
    dedupe: [
      'vue',
      '@fontsource-variable/epilogue',
      '@fontsource-variable/instrument-sans',
      '@fontsource-variable/jetbrains-mono',
    ],
  },
  server: {
    port: 5173,
    strictPort: true,
    // This project, as by default, plus the shared files it imports.
    fs: { allow: ['.', '../frontend'] },
  },
  build: { target: 'safari15', emptyOutDir: true },
});

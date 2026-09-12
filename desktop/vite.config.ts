import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

// The setup wizard is the only page this app bundles; Tauri embeds ./dist as its
// frontendDist. Everything after setup is served by the Chalie server itself.
export default defineConfig({
  plugins: [vue()],
  clearScreen: false,
  server: { port: 5173, strictPort: true },
  build: { target: 'safari15', emptyOutDir: true },
});

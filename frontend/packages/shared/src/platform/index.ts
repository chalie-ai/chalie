import type { PlatformAdapter } from './PlatformAdapter';
import { webPlatformAdapter } from './webPlatformAdapter';
import { tauriPlatformAdapter } from './tauriPlatformAdapter';

/**
 * The active PlatformAdapter for this runtime. Tauri injects `window.__TAURI__`
 * into its webview; everything else is the web adapter. Resolved once at module
 * load — the runtime does not change mid-session.
 */
export const platform: PlatformAdapter =
  (globalThis as { __TAURI__?: unknown }).__TAURI__ ? tauriPlatformAdapter : webPlatformAdapter;

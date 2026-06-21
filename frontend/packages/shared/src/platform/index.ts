import type { PlatformAdapter } from './PlatformAdapter';
import { webPlatformAdapter } from './webPlatformAdapter';
import { tauriPlatformAdapter } from './tauriPlatformAdapter';

/**
 * True inside a Tauri webview. Tauri always injects the `__TAURI_INTERNALS__`
 * IPC bridge regardless of the `withGlobalTauri` config (the `__TAURI__` global
 * API surface is NOT injected unless that flag is set), so this is the only
 * reliable native-runtime signal. The single source of truth for "am I native".
 */
export const isTauri: boolean = !!(globalThis as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;

/**
 * The active PlatformAdapter for this runtime: the Tauri adapter natively, the
 * web adapter everywhere else. Resolved once at module load — the runtime does
 * not change mid-session.
 */
export const platform: PlatformAdapter = isTauri ? tauriPlatformAdapter : webPlatformAdapter;

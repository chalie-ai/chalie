/**
 * Vision API — re-exports from providers (same endpoints).
 * Derived from frontend/brain/vision.js:
 *   GET  /providers           → list all providers (with supports_vision flag)
 *   GET  /providers/vision    → get current vision provider
 *   PUT  /providers/vision    → set vision provider { provider_id }
 */
export { providers as vision } from './providers';

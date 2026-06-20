/**
 * Client Context Heartbeat
 *
 * Port of frontend/interface/heartbeat.js.
 *
 * Periodically sends the user's timezone, location, system info, device context,
 * battery state, network quality, user preferences, and behavioral signals to
 * the backend. This provides a single source of truth for services to read
 * client context without per-request overhead.
 *
 * Module-level singleton: shared across the app; safe to call useHeartbeat()
 * outside setup().
 */

import { system } from '../api';
import { webPlatformAdapter } from '@chalie/shared';
import { emit } from './useEventBus';
import { useAmbientSensor } from './useAmbientSensor';
import type { AmbientSensorApi } from './useAmbientSensor';

// ---------------------------------------------------------------------------
// Constants — port verbatim from heartbeat.js
// ---------------------------------------------------------------------------

const HEARTBEAT_INTERVAL = 5 * 60 * 1000;   // 5 min
const GEO_TIMEOUT        = 10_000;           // 10 s for requestLocationPermission
const GEO_MAX_AGE        = 0;                // always fresh when prompting

// ---------------------------------------------------------------------------
// Locale → ISO-4217 currency map — port verbatim from heartbeat.js
// ---------------------------------------------------------------------------

const REGION_CURRENCY_MAP: Record<string, string> = {
  US: 'USD', GB: 'GBP', MT: 'EUR', DE: 'EUR', FR: 'EUR', IT: 'EUR',
  ES: 'EUR', NL: 'EUR', BE: 'EUR', AT: 'EUR', IE: 'EUR', PT: 'EUR',
  FI: 'EUR', GR: 'EUR', LU: 'EUR', CY: 'EUR', SK: 'EUR', SI: 'EUR',
  EE: 'EUR', LV: 'EUR', LT: 'EUR', HR: 'EUR',
  JP: 'JPY', CN: 'CNY', KR: 'KRW', IN: 'INR', AU: 'AUD', CA: 'CAD',
  CH: 'CHF', SE: 'SEK', NO: 'NOK', DK: 'DKK', PL: 'PLN', CZ: 'CZK',
  HU: 'HUF', RO: 'RON', BG: 'BGN', RU: 'RUB', BR: 'BRL', MX: 'MXN',
  ZA: 'ZAR', TR: 'TRY', NZ: 'NZD', SG: 'SGD', HK: 'HKD', TW: 'TWD',
  TH: 'THB', MY: 'MYR', PH: 'PHP', ID: 'IDR', VN: 'VND', AE: 'AED',
  SA: 'SAR', IL: 'ILS', EG: 'EGP', NG: 'NGN', KE: 'KES', AR: 'ARS',
  CL: 'CLP', CO: 'COP', PE: 'PEN',
};

/** Derive ISO 4217 currency code from the user's locale. */
function _detectCurrency(locale: string): string {
  try {
    const region = locale.match(/[-_]([A-Z]{2})$/i)?.[1].toUpperCase();
    if (region) return REGION_CURRENCY_MAP[region] ?? 'USD';
  } catch { /* fall through */ }
  return 'USD';
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface BatteryManager {
  level: number;
  charging: boolean;
}

interface NetworkInformation {
  effectiveType?: string;
  saveData?: boolean;
  downlink?: number;
}

/** Public API returned by useHeartbeat(). */
export interface HeartbeatApi {
  start(): void;
  stop(): void;
  setAmbientSensor(sensor: AmbientSensorApi): void;
  onAuthFailure(cb: () => void): void;
  requestLocationPermission(): Promise<void>;
}

// ---------------------------------------------------------------------------
// Singleton state
// ---------------------------------------------------------------------------

let _interval: ReturnType<typeof setInterval> | null = null;
let _ambientSensor: AmbientSensorApi | null = null;
let _authFailureCb: (() => void) | null = null;
let _authFailureFired = false;
let _onVisibilityChange: (() => void) | null = null;

// ---------------------------------------------------------------------------
// Device detection — port verbatim from heartbeat.js
// ---------------------------------------------------------------------------

function _classifyDeviceClass(coarse: boolean, minDim: number): 'phone' | 'tablet' | 'desktop' {
  if (coarse && minDim < 600) return 'phone';
  if (coarse) return 'tablet';
  return 'desktop';
}

function _detectPlatform(): string {
  const uad = (navigator as Navigator & { userAgentData?: { platform: string } }).userAgentData;
  if (uad?.platform) return uad.platform;
  if (navigator.platform) {
    const p = navigator.platform.toLowerCase();
    if (p.includes('mac')) return 'macOS';
    if (p.includes('win')) return 'Windows';
    if (p.includes('linux')) return 'Linux';
    if (p.includes('iphone') || p.includes('ipad')) return 'iOS';
  }
  return 'unknown';
}

function _detectDevice() {
  const sw = screen.width;
  const sh = screen.height;
  const coarse = matchMedia('(pointer: coarse)').matches;

  return {
    class: _classifyDeviceClass(coarse, Math.min(sw, sh)),
    platform: _detectPlatform(),
    screen_w: sw,
    screen_h: sh,
    pixel_ratio: window.devicePixelRatio || 1,
    orientation: sw > sh ? 'landscape' : 'portrait',
    input: coarse ? 'coarse' : 'fine',
  } as const;
}

// ---------------------------------------------------------------------------
// Battery
// ---------------------------------------------------------------------------

async function _getBattery(): Promise<{ level: number; charging: boolean } | null> {
  const nav = navigator as Navigator & { getBattery?: () => Promise<BatteryManager> };
  if (!nav.getBattery) return null;
  try {
    const batt = await nav.getBattery();
    return {
      level: Math.round(batt.level * 100) / 100,
      charging: batt.charging,
    };
  } catch (e) {
    console.warn('[heartbeat] getBattery failed:', e);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

interface NetworkFields {
  effective_type?: string;
  save_data?: boolean;
  downlink?: number;
}

function _getNetwork(): NetworkFields | null {
  const conn = (navigator as Navigator & { connection?: NetworkInformation }).connection;
  if (!conn) return null;
  const net: NetworkFields = {};
  if (conn.effectiveType) net.effective_type = conn.effectiveType;
  if (conn.saveData !== undefined) net.save_data = conn.saveData;
  if (conn.downlink !== undefined) net.downlink = conn.downlink;
  return Object.keys(net).length ? net : null;
}

// ---------------------------------------------------------------------------
// Preferences
// ---------------------------------------------------------------------------

function _getPreferences() {
  return {
    color_scheme: matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light',
    reduced_motion: matchMedia('(prefers-reduced-motion: reduce)').matches,
  } as const;
}

// ---------------------------------------------------------------------------
// Geolocation (granted-only, no prompt)
// ---------------------------------------------------------------------------

async function _getGrantedLocation(): Promise<{ lat: number; lon: number } | null> {
  if (!navigator.geolocation || !navigator.permissions) return null;
  try {
    if ((await navigator.permissions.query({ name: 'geolocation' })).state !== 'granted') return null;
    const pos = await webPlatformAdapter.getCurrentPosition({
      timeout: 3_000,
      maximumAge: 300_000,   // accept 5-min cached position
    });
    return { lat: pos.coords.latitude, lon: pos.coords.longitude };
  } catch (e) {
    console.warn('[heartbeat] permissions API not supported:', e);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Payload builder
// ---------------------------------------------------------------------------

async function _buildContextPayload(): Promise<Record<string, unknown>> {
  const resolvedLocale = new Intl.NumberFormat().resolvedOptions().locale;

  const ctx: Record<string, unknown> = {
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    locale: resolvedLocale,
    language: navigator.language,
    currency: _detectCurrency(resolvedLocale),
    local_time: new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }),
  };

  ctx['device'] = _detectDevice();

  // Network + legacy connection field
  const network = _getNetwork();
  if (network) {
    ctx['network'] = network;
    if (network.effective_type) ctx['connection'] = network.effective_type;
  } else {
    const conn = (navigator as Navigator & { connection?: NetworkInformation }).connection;
    if (conn?.effectiveType) ctx['connection'] = conn.effectiveType;
  }

  const battery = await _getBattery();
  if (battery) ctx['battery'] = battery;

  ctx['preferences'] = _getPreferences();

  ctx['behavioral'] = (_ambientSensor ?? useAmbientSensor()).snapshot();

  ctx['location'] = await _getGrantedLocation();

  return ctx;
}

// ---------------------------------------------------------------------------
// Auth check
// ---------------------------------------------------------------------------

async function _checkAuth(): Promise<void> {
  if (_authFailureFired) return;
  try {
    const data = await system.authStatus();
    if (data.has_master_account && !data.has_session) {
      _authFailureFired = true;
      stop();
      _authFailureCb?.();
    }
  } catch (e) {
    console.warn('[heartbeat] auth check failed:', e);
  }
}

// ---------------------------------------------------------------------------
// Send context
// ---------------------------------------------------------------------------

async function _sendContext(): Promise<void> {
  try {
    const result = await system.heartbeat(await _buildContextPayload()) as Record<string, unknown>;
    if (result['attention']) emit('chalie:attention', { attention: result['attention'] as string });
  } catch (err) {
    console.warn('[CLIENT HEARTBEAT] Error sending context:', err instanceof Error ? err.message : String(err));
  }

  // Always verify the session is still valid.
  await _checkAuth();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

function start(): void {
  if (_interval !== null) return;

  void _sendContext();

  _interval = setInterval(() => { void _sendContext(); }, HEARTBEAT_INTERVAL);

  _onVisibilityChange = () => {
    if (document.visibilityState === 'visible' && !_authFailureFired) {
      void _sendContext();
    }
  };
  document.addEventListener('visibilitychange', _onVisibilityChange);
}

function stop(): void {
  if (_interval !== null) {
    clearInterval(_interval);
    _interval = null;
  }
  if (_onVisibilityChange !== null) {
    document.removeEventListener('visibilitychange', _onVisibilityChange);
    _onVisibilityChange = null;
  }
}

function setAmbientSensor(sensor: AmbientSensorApi): void {
  _ambientSensor = sensor;
}

function onAuthFailure(cb: () => void): void {
  _authFailureCb = cb;
}

/**
 * Request geolocation permission from the browser (shows browser prompt if needed).
 * Call once after login to prime the permission so subsequent heartbeats capture coords.
 * Immediately resends a fresh heartbeat if permission is granted.
 */
async function requestLocationPermission(): Promise<void> {
  if (!navigator.geolocation) return;
  try {
    await webPlatformAdapter.getCurrentPosition({
      timeout: GEO_TIMEOUT,
      maximumAge: GEO_MAX_AGE,
    });
    // Permission was granted — resend context immediately with coordinates.
    await _sendContext();
  } catch (e) {
    console.warn('[heartbeat] geolocation request failed:', e);
  }
}

const _heartbeatApi: HeartbeatApi = {
  start,
  stop,
  setAmbientSensor,
  onAuthFailure,
  requestLocationPermission,
};

/**
 * Returns the module-level singleton heartbeat controller.
 * Safe to call outside setup() — the singleton is created at module load.
 */
export function useHeartbeat(): HeartbeatApi {
  return _heartbeatApi;
}

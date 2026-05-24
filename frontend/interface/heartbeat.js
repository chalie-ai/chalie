/**
 * Client Context Heartbeat
 *
 * Periodically sends the user's timezone, location, system info, device context,
 * battery state, network quality, user preferences, and behavioral signals to
 * the backend. This provides a single source of truth for services to read
 * client context without per-request overhead.
 */

/**
 * Derive ISO 4217 currency code from the user's locale using Intl.NumberFormat.
 */
function _detectCurrency(locale) {
  try {
    const parts = new Intl.NumberFormat(locale, { style: 'currency', currency: 'USD' })
      .resolvedOptions();
    // Try to get the actual locale's default currency via a fresh formatter
    const fmt = new Intl.NumberFormat(locale, { style: 'currency', currencyDisplay: 'code', currency: 'USD' });
    // The reliable way: use locale-to-currency mapping from the browser
    const regionMatch = locale.match(/[-_]([A-Z]{2})$/i);
    if (regionMatch) {
      const region = regionMatch[1].toUpperCase();
      // Use Intl to format with the locale's natural currency
      try {
        const test = new Intl.NumberFormat(locale, { style: 'currency', currency: 'XTS' });
      } catch (e) {
        // Expected — XTS may not be supported
      }
      // Map common regions to currencies
      const regionCurrencyMap = {
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
      if (regionCurrencyMap[region]) return regionCurrencyMap[region];
    }
  } catch (e) { /* fall through */ }
  return 'USD';
}

export class ClientHeartbeat {
  constructor(getHost) {
    this._getHost = getHost || (() => '');
    this._interval = null;
    this._lastSentAt = 0;
    this._ambientSensor = null;
    this._authFailureCb = null;
    this._authFailureFired = false;
    this._onVisibilityChange = null;
  }

  /**
   * Register an AmbientSensor instance whose snapshot() will be included
   * in every heartbeat payload.
   */
  setAmbientSensor(sensor) {
    this._ambientSensor = sensor;
  }

  /**
   * Register a callback invoked when /auth/status reports the user is no
   * longer authenticated (e.g., vault locked after server restart).
   * Called at most once per heartbeat lifetime.
   */
  onAuthFailure(cb) {
    this._authFailureCb = cb;
  }

  /**
   * Build full URL with optional backend host.
   */
  _buildUrl(path) {
    const host = this._getHost();
    return host ? host.replace(/\/$/, '') + path : path;
  }

  /**
   * Start the heartbeat (send on page load + every 5 minutes).
   */
  start() {
    // Send immediately on startup
    this._sendContext();

    // Then every 5 minutes
    this._interval = setInterval(() => this._sendContext(), 5 * 60 * 1000);

    // Also re-check auth (+ resend context) whenever the tab becomes visible
    // again — catches "left the page open overnight, server was restarted"
    // without waiting up to 5 minutes for the next interval tick.
    this._onVisibilityChange = () => {
      if (document.visibilityState === 'visible' && !this._authFailureFired) {
        this._sendContext();
      }
    };
    document.addEventListener('visibilitychange', this._onVisibilityChange);
  }

  /**
   * Stop the heartbeat.
   */
  stop() {
    if (this._interval) {
      clearInterval(this._interval);
      this._interval = null;
    }
    if (this._onVisibilityChange) {
      document.removeEventListener('visibilitychange', this._onVisibilityChange);
      this._onVisibilityChange = null;
    }
  }

  /**
   * Poll /auth/status. If the session is no longer valid (vault locked,
   * session expired, etc.), fire the onAuthFailure callback and stop
   * further polling.
   */
  async _checkAuth() {
    if (this._authFailureFired) return;
    try {
      const res = await fetch(this._buildUrl('/auth/status'), {
        credentials: 'same-origin',
      });
      if (!res.ok) return; // transient — don't trigger redirect on network glitch
      const data = await res.json();
      // /auth/status already forces has_session=false when vault_state==='locked',
      // so this single check covers session expiry AND vault-locked-after-restart.
      // Only redirect if the account still exists — a fresh install with no
      // master account is handled by the on-boarding redirect elsewhere.
      if (data.has_master_account && !data.has_session) {
        this._authFailureFired = true;
        this.stop();
        if (typeof this._authFailureCb === 'function') {
          this._authFailureCb();
        }
      }
    } catch (e) { console.warn('[heartbeat] auth check failed:', e); }
  }

  /**
   * Request geolocation permission from the browser (shows browser prompt if needed).
   * Call once after login to prime the permission so subsequent heartbeats capture coords.
   * Immediately sends a fresh heartbeat if permission is granted.
   */
  async requestLocationPermission() {
    if (!navigator.geolocation) return;
    try {
      await new Promise((res) => {
        navigator.geolocation.getCurrentPosition(
          () => res(true),
          () => res(false),
          { timeout: 10000, maximumAge: 0 }
        );
      });
      // Permission was granted — resend context immediately with coordinates
      this._sendContext();
    } catch (e) { console.warn('[heartbeat] geolocation request failed:', e); }
  }

  /**
   * Detect device class from screen size + pointer type (no user-agent parsing).
   */
  _detectDevice() {
    const sw = screen.width;
    const sh = screen.height;
    const coarse = matchMedia('(pointer: coarse)').matches;

    return {
      class: this._classifyDeviceClass(coarse, Math.min(sw, sh)),
      platform: this._detectPlatform(),
      screen_w: sw,
      screen_h: sh,
      pixel_ratio: window.devicePixelRatio || 1,
      orientation: sw > sh ? 'landscape' : 'portrait',
      input: coarse ? 'coarse' : 'fine',
    };
  }

  /** Classify phone/tablet/desktop from pointer type and minimum screen dimension. */
  _classifyDeviceClass(coarse, minDim) {
    if (coarse && minDim < 600) return 'phone';
    if (coarse) return 'tablet';
    return 'desktop';
  }

  /** Detect OS platform from userAgentData or legacy navigator.platform. */
  _detectPlatform() {
    if (navigator.userAgentData?.platform) {
      return navigator.userAgentData.platform;
    }
    if (navigator.platform) {
      const p = navigator.platform.toLowerCase();
      if (p.includes('mac')) return 'macOS';
      if (p.includes('win')) return 'Windows';
      if (p.includes('linux')) return 'Linux';
      if (p.includes('iphone') || p.includes('ipad')) return 'iOS';
    }
    return 'unknown';
  }

  /**
   * Get battery info (Chrome/Edge only — getBattery is not available everywhere).
   */
  async _getBattery() {
    if (!navigator.getBattery) return null;
    try {
      const batt = await navigator.getBattery();
      return {
        level: Math.round(batt.level * 100) / 100,
        charging: batt.charging,
      };
    } catch (e) {
      console.warn('[heartbeat] getBattery failed:', e);
      return null;
    }
  }

  /**
   * Get enhanced network info from Network Information API.
   */
  _getNetwork() {
    const conn = navigator.connection;
    if (!conn) return null;
    const net = {};
    if (conn.effectiveType) net.effective_type = conn.effectiveType;
    if (conn.saveData !== undefined) net.save_data = conn.saveData;
    if (conn.downlink !== undefined) net.downlink = conn.downlink;
    return Object.keys(net).length ? net : null;
  }

  /**
   * Get user preferences (color scheme, motion).
   */
  _getPreferences() {
    return {
      color_scheme: matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light',
      reduced_motion: matchMedia('(prefers-reduced-motion: reduce)').matches,
    };
  }

  /**
   * Collect and send client context to /health endpoint.
   */
  async _sendContext() {
    try {
      const ctx = await this._buildContextPayload();

      const response = await fetch(this._buildUrl('/health'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify(ctx),
      });

      await this._handleHealthResponse(response);
      this._lastSentAt = Date.now();
    } catch (err) {
      console.warn('[CLIENT HEARTBEAT] Error sending context:', err.message);
    }

    // Always verify the session is still valid — /health is public, so the
    // POST above succeeds even after the vault locks. /auth/status is the
    // source of truth for auth state.
    await this._checkAuth();
  }

  /**
   * Build the full context payload object, collecting device, network,
   * battery, preferences, behavioral, and geolocation data.
   */
  async _buildContextPayload() {
    const resolvedLocale = Intl.NumberFormat().resolvedOptions().locale;
    const ctx = {
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      locale: resolvedLocale,
      language: navigator.language,
      currency: _detectCurrency(resolvedLocale),
      local_time: new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }),
    };

    ctx.device = this._detectDevice();
    this._applyNetworkFields(ctx);

    const battery = await this._getBattery();
    if (battery) ctx.battery = battery;

    ctx.preferences = this._getPreferences();

    if (this._ambientSensor) {
      ctx.behavioral = this._ambientSensor.snapshot();
    }

    ctx.location = await this._getGrantedLocation();

    return ctx;
  }

  /**
   * Populate network and legacy connection fields on the context object.
   */
  _applyNetworkFields(ctx) {
    const network = this._getNetwork();
    if (network) {
      ctx.network = network;
      // Keep legacy field for backward compatibility
      if (network.effective_type) ctx.connection = network.effective_type;
    } else if (navigator.connection?.effectiveType) {
      ctx.connection = navigator.connection.effectiveType;
    }
  }

  /**
   * Return the current coordinates if geolocation permission is already
   * granted, or null otherwise. Never triggers a permission prompt — that
   * is handled by requestLocationPermission().
   */
  async _getGrantedLocation() {
    if (!navigator.geolocation || !navigator.permissions) return null;
    try {
      const perm = await navigator.permissions.query({ name: 'geolocation' });
      if (perm.state !== 'granted') return null;
      return await new Promise((res) => {
        navigator.geolocation.getCurrentPosition(
          (p) => res({ lat: p.coords.latitude, lon: p.coords.longitude }),
          () => res(null),
          { timeout: 3000, maximumAge: 300000 }  // accept 5-min cached position
        );
      });
    } catch (e) {
      console.warn('[heartbeat] permissions API not supported:', e);
      return null;
    }
  }

  /**
   * Handle the /health response: warn on failure, dispatch attention event
   * on success if the server signals one.
   */
  async _handleHealthResponse(response) {
    if (!response.ok) {
      console.warn('[CLIENT HEARTBEAT] Failed to send context:', response.status);
      return;
    }
    try {
      const result = await response.json();
      if (result.attention) {
        document.dispatchEvent(new CustomEvent('chalie:attention', {
          detail: { attention: result.attention }
        }));
      }
    } catch (e) {
      console.warn('[heartbeat] failed to parse health response:', e);
    }
  }
}

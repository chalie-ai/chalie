/**
 * Client Context Heartbeat
 *
 * Periodically sends the user's timezone, location, system info, device context,
 * battery state, network quality, user preferences, and behavioral signals to
 * the backend. This provides a single source of truth for services to read
 * client context without per-request overhead.
 */

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
    } catch (_) { /* network error — ignore, will retry next beat */ }
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
    } catch (_) { /* geolocation not supported */ }
  }

  /**
   * Detect device class from screen size + pointer type (no user-agent parsing).
   */
  _detectDevice() {
    const sw = screen.width;
    const sh = screen.height;
    const minDim = Math.min(sw, sh);
    const coarse = matchMedia('(pointer: coarse)').matches;

    let deviceClass;
    if (coarse && minDim < 600) deviceClass = 'phone';
    else if (coarse) deviceClass = 'tablet';
    else deviceClass = 'desktop';

    // Platform detection from navigator.platform / userAgentData
    let platform = 'unknown';
    if (navigator.userAgentData?.platform) {
      platform = navigator.userAgentData.platform;
    } else if (navigator.platform) {
      const p = navigator.platform.toLowerCase();
      if (p.includes('mac')) platform = 'macOS';
      else if (p.includes('win')) platform = 'Windows';
      else if (p.includes('linux')) platform = 'Linux';
      else if (p.includes('iphone') || p.includes('ipad')) platform = 'iOS';
    }

    // PWA detection
    const pwa = matchMedia('(display-mode: standalone)').matches ||
                matchMedia('(display-mode: fullscreen)').matches ||
                navigator.standalone === true;

    return {
      class: deviceClass,
      platform,
      screen_w: sw,
      screen_h: sh,
      pixel_ratio: window.devicePixelRatio || 1,
      orientation: sw > sh ? 'landscape' : 'portrait',
      input: coarse ? 'coarse' : 'fine',
      pwa,
    };
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
    } catch (_) {
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
      const ctx = {
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        locale: Intl.NumberFormat().resolvedOptions().locale,     // e.g., "en-MT"
        language: navigator.language,                              // e.g., "en-GB"
        local_time: new Date().toISOString(),
      };

      // Device info
      ctx.device = this._detectDevice();

      // Network (enhanced — replaces the old single connection field)
      const network = this._getNetwork();
      if (network) {
        ctx.network = network;
        // Keep legacy field for backward compatibility
        if (network.effective_type) ctx.connection = network.effective_type;
      } else if (navigator.connection?.effectiveType) {
        ctx.connection = navigator.connection.effectiveType;
      }

      // Battery (async — Chrome/Edge only)
      const battery = await this._getBattery();
      if (battery) ctx.battery = battery;

      // User preferences
      ctx.preferences = this._getPreferences();

      // Behavioral snapshot from AmbientSensor
      if (this._ambientSensor) {
        ctx.behavioral = this._ambientSensor.snapshot();
      }

      // Best-effort geolocation (no prompt — requestLocationPermission handles that)
      if (navigator.geolocation && navigator.permissions) {
        try {
          const perm = await navigator.permissions.query({ name: 'geolocation' });
          if (perm.state === 'granted') {
            ctx.location = await new Promise((res) => {
              navigator.geolocation.getCurrentPosition(
                (p) => res({ lat: p.coords.latitude, lon: p.coords.longitude }),
                () => res(null),
                { timeout: 3000, maximumAge: 300000 }  // accept 5-min cached position
              );
            });
          }
        } catch (_) { /* permissions API not supported — skip */ }
      }

      // Send to backend
      const response = await fetch(this._buildUrl('/health'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify(ctx),
      });

      if (!response.ok) {
        console.warn('[CLIENT HEARTBEAT] Failed to send context:', response.status);
      } else {
        try {
          const result = await response.json();
          if (result.attention) {
            document.dispatchEvent(new CustomEvent('chalie:attention', {
              detail: { attention: result.attention }
            }));
          }
        } catch (_) { /* ignore parse failures */ }
      }

      this._lastSentAt = Date.now();
    } catch (err) {
      console.warn('[CLIENT HEARTBEAT] Error sending context:', err.message);
    }

    // Always verify the session is still valid — /health is public, so the
    // POST above succeeds even after the vault locks. /auth/status is the
    // source of truth for auth state.
    await this._checkAuth();
  }
}

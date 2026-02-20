/**
 * Client Context Heartbeat
 *
 * Periodically sends the user's timezone, location, and system info to the backend.
 * This provides a single source of truth for services to read client context without
 * per-request overhead.
 */

export class ClientHeartbeat {
  constructor(getHost) {
    this._getHost = getHost || (() => '');
    this._interval = null;
    this._lastSentAt = 0;
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
  }

  /**
   * Stop the heartbeat.
   */
  stop() {
    if (this._interval) {
      clearInterval(this._interval);
      this._interval = null;
    }
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

      // Connection quality (Network Information API — not available in all browsers)
      if (navigator.connection?.effectiveType) {
        ctx.connection = navigator.connection.effectiveType;       // "4g", "3g", etc.
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
      }

      this._lastSentAt = Date.now();
    } catch (err) {
      console.warn('[CLIENT HEARTBEAT] Error sending context:', err.message);
    }
  }
}

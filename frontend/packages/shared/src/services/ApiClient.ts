import type { GetHost } from './types';

export class AuthError extends Error {
  constructor() {
    super('AUTH');
    this.name = 'AuthError';
  }
}

export class HttpError extends Error {
  constructor(
    public readonly status: number,
    public readonly error?: string,   // server-provided message parsed from a JSON {error} body, if any
    public readonly body?: unknown,    // the full parsed JSON error body, if any
  ) {
    super(`HTTP ${status}`);
    this.name = 'HttpError';
  }
}

/** Per-call options accepted by every request method. */
export interface RequestOpts {
  /**
   * On HTTP 401, invoke the auth-error handler (by default a redirect to the
   * login page) before throwing AuthError. Set `false` for the few callers that
   * must read the 401 as *data* rather than treat it as session expiry — the
   * login form (401 = bad credentials) and the auth-status gate probe (401 →
   * routing decision). Defaults to `true`.
   */
  redirectOnAuthError?: boolean;
}

/** Handler invoked when a request gets a 401 and the caller didn't opt out. */
export type AuthErrorHandler = () => void;

/**
 * Default auth-error handler: hard-navigate to /login/, preserving the current
 * path + query string as `next`. Idempotency is owned by the client (see the
 * `_authErrorFired` latch in `fail401`), so this handler is pure — it just
 * navigates. Ports the `_authRedirected` guard the apps previously kept
 * individually.
 */
function redirectToLogin(): void {
  const next = window.location.pathname + window.location.search;
  window.location.replace('/login/?next=' + encodeURIComponent(next));
}

/**
 * Centralised Chalie REST client. Configurable host; sends the same-origin
 * session cookie on every request. On 401 it runs the auth-error handler
 * (default: redirect to /login/) unless the caller passes
 * `{ redirectOnAuthError: false }`, then always throws AuthError so callers can
 * still react. Non-2xx → HttpError carrying the server's {error} message.
 */
export class ApiClient {
  /**
   * Idempotency latch: once a 401 has triggered the auth-error handler, later
   * 401s on this client are no-ops (we're already navigating away). Scoped to
   * the client instance, not the module, so test/HMR lifetimes stay isolated.
   */
  private _authErrorFired = false;

  constructor(
    private readonly getHost: GetHost,
    private readonly onAuthError: AuthErrorHandler = redirectToLogin,
  ) {}

  private buildUrl(path: string): string {
    const host = this.getHost();
    return host ? host.replace(/\/$/, '') + path : path;
  }

  /** React to a 401: redirect (unless opted out), then always throw AuthError. */
  private fail401(opts?: RequestOpts): never {
    if ((opts?.redirectOnAuthError ?? true) && !this._authErrorFired) {
      this._authErrorFired = true;
      this.onAuthError();
    }
    throw new AuthError();
  }

  /** Parse the response body and throw an HttpError carrying the server message, if any. */
  private async throwHttp(res: Response): Promise<never> {
    const body = await res.json().catch(() => null);
    const msg =
      body && typeof body === 'object' && 'error' in body && typeof (body as { error?: unknown }).error === 'string'
        ? (body as { error?: string }).error
        : undefined;
    throw new HttpError(res.status, msg, body ?? undefined);
  }

  private async request<T>(path: string, init?: RequestInit, opts?: RequestOpts): Promise<T> {
    const res = await fetch(this.buildUrl(path), {
      credentials: 'same-origin',
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    });
    if (res.status === 401) this.fail401(opts);
    if (!res.ok) return this.throwHttp(res);
    return (await res.json()) as T;
  }

  get<T>(path: string, opts?: RequestOpts): Promise<T> {
    return this.request<T>(path, undefined, opts);
  }
  post<T>(path: string, body?: unknown, opts?: RequestOpts): Promise<T> {
    return this.request<T>(path, { method: 'POST', body: JSON.stringify(body ?? {}) }, opts);
  }
  put<T>(path: string, body?: unknown, opts?: RequestOpts): Promise<T> {
    return this.request<T>(path, { method: 'PUT', body: JSON.stringify(body ?? {}) }, opts);
  }
  async del<T>(path: string, opts?: RequestOpts): Promise<T> {
    const res = await fetch(this.buildUrl(path), {
      method: 'DELETE',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
    });
    if (res.status === 401) this.fail401(opts);
    if (!res.ok) return this.throwHttp(res);
    return (await res.json().catch(() => ({}))) as T;
  }

  /** Multipart upload — no JSON Content-Type (browser sets the boundary). */
  async upload<T>(path: string, formData: FormData, opts?: RequestOpts): Promise<T> {
    const res = await fetch(this.buildUrl(path), {
      method: 'POST',
      credentials: 'same-origin',
      body: formData,
    });
    if (res.status === 401) this.fail401(opts);
    if (!res.ok) return this.throwHttp(res);
    return (await res.json()) as T;
  }

  /**
   * POST returning the raw Response (for binary/blob downloads).
   * Throws AuthError on 401 (redirecting unless opted out); does NOT throw on
   * other non-ok statuses — the caller inspects res.ok and reads
   * .blob()/.json() itself.
   */
  async download(path: string, body?: unknown, opts?: RequestOpts): Promise<Response> {
    const res = await fetch(this.buildUrl(path), {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body ?? {}),
    });
    if (res.status === 401) this.fail401(opts);
    return res;
  }

  // ── Foundation endpoints (feature endpoints added in P1/P2) ──────────────
  health(): Promise<{ status: string } | null> {
    return fetch(this.buildUrl('/health'), { credentials: 'same-origin' })
      .then((r) => r.json())
      .catch(() => null);
  }
  /** Never rejects. */
  ready(): Promise<{ ready: boolean }> {
    return fetch(this.buildUrl('/ready'), { credentials: 'same-origin' })
      .then((r) => (r.ok ? r.json() : { ready: false }))
      .catch(() => ({ ready: false }));
  }
  getSetting(key: string): Promise<{ key: string; value: string | null }> {
    return this.get(`/system/settings/${encodeURIComponent(key)}`);
  }
}

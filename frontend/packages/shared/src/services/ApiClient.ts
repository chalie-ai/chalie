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
    public readonly error?: string, // server-provided message parsed from a JSON {error} body, if any
    public readonly body?: unknown, // the full parsed JSON error body, if any
  ) {
    super(`HTTP ${status}`);
    this.name = 'HttpError';
  }
}

export interface RequestOpts {
  /**
   * Defaults to `true` (401 → run auth-error handler, then throw AuthError). Set
   * `false` for callers that must read the 401 as *data*, not session expiry —
   * the login form (401 = bad credentials) and the auth-status gate probe.
   */
  redirectOnAuthError?: boolean;
}

export type AuthErrorHandler = () => void;

/**
 * Default auth-error handler: hard-navigate to /login/ with the current
 * path+query as `next`. Idempotency lives in the client's `_authErrorFired`
 * latch, so this handler is pure navigation.
 */
function redirectToLogin(): void {
  const next = window.location.pathname + window.location.search;
  window.location.replace('/login/?next=' + encodeURIComponent(next));
}

/**
 * Centralised Chalie REST client (configurable host, same-origin session
 * cookie). On 401: runs the auth-error handler unless `redirectOnAuthError:
 * false`, then always throws AuthError. Non-2xx → HttpError with the server's
 * {error} message.
 */
export class ApiClient {
  // Idempotency latch: after the first 401 fires the handler, later 401s no-op
  // (already navigating away). Instance-scoped so test/HMR lifetimes stay isolated.
  private _authErrorFired = false;

  constructor(
    private readonly getHost: GetHost,
    private readonly getToken: GetHost,
    private readonly onAuthError: AuthErrorHandler = redirectToLogin,
  ) {}

  /**
   * Bearer header when a token is configured, else `{}`. Spreading the empty
   * object is a no-op, so the web (cookie) path is unchanged — one common path.
   */
  private authHeaders(): Record<string, string> {
    const token = this.getToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  private buildUrl(path: string): string {
    const host = this.getHost();
    return host ? host.replace(/\/$/, '') + path : path;
  }

  private fail401(opts?: RequestOpts): never {
    if ((opts?.redirectOnAuthError ?? true) && !this._authErrorFired) {
      this._authErrorFired = true;
      this.onAuthError();
    }
    throw new AuthError();
  }

  private async throwHttp(res: Response): Promise<never> {
    const body = await res.json().catch(() => null);
    const msg =
      body &&
      typeof body === 'object' &&
      'error' in body &&
      typeof (body as { error?: unknown }).error === 'string'
        ? (body as { error?: string }).error
        : undefined;
    throw new HttpError(res.status, msg, body ?? undefined);
  }

  private async request<T>(path: string, init?: RequestInit, opts?: RequestOpts): Promise<T> {
    const res = await fetch(this.buildUrl(path), {
      credentials: 'same-origin',
      ...init,
      headers: { 'Content-Type': 'application/json', ...this.authHeaders(), ...init?.headers },
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
      headers: { 'Content-Type': 'application/json', ...this.authHeaders() },
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
      headers: { ...this.authHeaders() },
      body: formData,
    });
    if (res.status === 401) this.fail401(opts);
    if (!res.ok) return this.throwHttp(res);
    return (await res.json()) as T;
  }

  /**
   * POST returning the raw Response (binary/blob downloads). Throws AuthError on
   * 401, but does NOT throw on other non-ok statuses — the caller inspects res.ok
   * and reads .blob()/.json() itself.
   */
  async download(path: string, body?: unknown, opts?: RequestOpts): Promise<Response> {
    const res = await fetch(this.buildUrl(path), {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', ...this.authHeaders() },
      body: JSON.stringify(body ?? {}),
    });
    if (res.status === 401) this.fail401(opts);
    return res;
  }

  health(): Promise<{ status: string } | null> {
    return fetch(this.buildUrl('/health'), {
      credentials: 'same-origin',
      headers: { ...this.authHeaders() },
    })
      .then((r) => r.json())
      .catch(() => null);
  }
  /** Never rejects. */
  ready(): Promise<{ ready: boolean }> {
    return fetch(this.buildUrl('/ready'), {
      credentials: 'same-origin',
      headers: { ...this.authHeaders() },
    })
      .then((r) => (r.ok ? r.json() : { ready: false }))
      .catch(() => ({ ready: false }));
  }
  getSetting(key: string): Promise<{ key: string; value: string | null }> {
    return this.get(`/api/system/settings/${encodeURIComponent(key)}`);
  }
}

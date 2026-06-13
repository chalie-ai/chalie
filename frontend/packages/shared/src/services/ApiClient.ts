import type { GetHost } from './types';

export class AuthError extends Error {
  constructor() {
    super('AUTH');
    this.name = 'AuthError';
  }
}

export class HttpError extends Error {
  constructor(public readonly status: number) {
    super(`HTTP ${status}`);
    this.name = 'HttpError';
  }
}

/** Centralised Chalie REST client. Configurable host; 401 → AuthError. */
export class ApiClient {
  constructor(private readonly getHost: GetHost) {}

  private buildUrl(path: string): string {
    const host = this.getHost();
    return host ? host.replace(/\/$/, '') + path : path;
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(this.buildUrl(path), {
      credentials: 'same-origin',
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    });
    if (res.status === 401) throw new AuthError();
    if (!res.ok) throw new HttpError(res.status);
    return (await res.json()) as T;
  }

  get<T>(path: string): Promise<T> {
    return this.request<T>(path);
  }
  post<T>(path: string, body?: unknown): Promise<T> {
    return this.request<T>(path, { method: 'POST', body: JSON.stringify(body ?? {}) });
  }
  put<T>(path: string, body?: unknown): Promise<T> {
    return this.request<T>(path, { method: 'PUT', body: JSON.stringify(body ?? {}) });
  }
  async del<T>(path: string): Promise<T> {
    const res = await fetch(this.buildUrl(path), {
      method: 'DELETE',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
    });
    if (res.status === 401) throw new AuthError();
    if (!res.ok) throw new HttpError(res.status);
    return (await res.json().catch(() => ({}))) as T;
  }

  /** Multipart upload — no JSON Content-Type (browser sets the boundary). */
  async upload<T>(path: string, formData: FormData): Promise<T> {
    const res = await fetch(this.buildUrl(path), {
      method: 'POST',
      credentials: 'same-origin',
      body: formData,
    });
    if (res.status === 401) throw new AuthError();
    if (!res.ok) throw new HttpError(res.status);
    return (await res.json()) as T;
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

/**
 * Chalie REST API client.
 */
export class ApiClient {
  /**
   * @param {() => string} getHost — returns the current backend host
   */
  constructor(getHost) {
    this._getHost = getHost;
  }

  _buildUrl(path) {
    const host = this._getHost?.();
    if (host) {
      return host.replace(/\/$/, '') + path;
    }
    return path;
  }

  async _get(path) {
    const res = await fetch(this._buildUrl(path), {
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
    });
    if (res.status === 401) throw new Error('AUTH');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async _put(path, body) {
    const res = await fetch(this._buildUrl(path), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(body),
    });
    if (res.status === 401) throw new Error('AUTH');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async _delete(path) {
    const res = await fetch(this._buildUrl(path), {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
    });
    if (res.status === 401) throw new Error('AUTH');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json().catch(() => ({}));
  }

  async _post(path, body) {
    const res = await fetch(this._buildUrl(path), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(body),
    });
    if (res.status === 401) throw new Error('AUTH');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  /** Public wrappers for use outside of this class */
  get(path) { return this._get(path); }
  post(path, body) { return this._post(path, body); }
  del(path) { return this._delete(path); }

  /** Upload a FormData payload (multipart, no JSON Content-Type) */
  async upload(path, formData) {
    const res = await fetch(this._buildUrl(path), {
      method: 'POST',
      credentials: 'same-origin',
      body: formData,
    });
    if (res.status === 401) throw new Error('AUTH');
    return res.json();
  }

  /** @returns {Promise<{status: string}>} */
  healthCheck() {
    return fetch(this._buildUrl('/health'), { credentials: 'same-origin' }).then(r => r.json()).catch(() => null);
  }

  /** @returns {Promise<{ready: boolean}>} — never rejects */
  readyCheck() {
    return fetch(this._buildUrl('/ready'), { credentials: 'same-origin' })
      .then(r => r.ok ? r.json() : { ready: false })
      .catch(() => ({ ready: false }));
  }

  /** @returns {Promise<{thread_id: string, exchanges: Array}>} */
  getRecentConversation() {
    return this._get('/conversation/recent');
  }

  /** @returns {Promise<{today: Array, this_week: Array, older_highlights: Array}>} */
  getConversationSummary() {
    return this._get('/conversation/summary');
  }

  /** @returns {Promise<{traits_summary: string, facts: Array, significant_episodes: Array, concepts: Array}>} */
  getMemoryContext() {
    return this._get('/memory/context');
  }

  /** @returns {Promise<{tools: Array}>} */
  getTools() {
    return this._get('/tools');
  }

  /** @returns {Promise<object>} */
  getToolConfig(name) {
    return this._get(`/tools/${encodeURIComponent(name)}/config`);
  }

  /** @returns {Promise<object>} */
  saveToolConfig(name, config) {
    return this._put(`/tools/${encodeURIComponent(name)}/config`, config);
  }

  /** @returns {Promise<object>} */
  deleteToolConfigKey(name, key) {
    return this._delete(`/tools/${encodeURIComponent(name)}/config/${encodeURIComponent(key)}`);
  }

  /** @returns {Promise<{ok: boolean, message: string}>} */
  testTool(name) {
    return this._post(`/tools/${encodeURIComponent(name)}/test`, {});
  }

}

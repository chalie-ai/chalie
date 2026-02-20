/**
 * POST-based SSE client for the /chat endpoint.
 *
 * Unlike native EventSource, this uses fetch + ReadableStream so we can
 * send a POST body and custom headers.
 */
export class SSEClient {
  /**
   * @param {() => string} getHost — returns the current backend host
   */
  constructor(getHost) {
    this._getHost = getHost;
    this._abortController = null;
  }

  _buildUrl(path) {
    const host = this._getHost?.();
    if (host) {
      return host.replace(/\/$/, '') + path;
    }
    return path;
  }

  /** Abort any in-flight request. */
  abort() {
    if (this._abortController) {
      this._abortController.abort();
      this._abortController = null;
    }
  }

  /**
   * Send a chat message and stream SSE responses.
   *
   * @param {string} text
   * @param {"text"|"voice"} source
   * @param {{
   *   onStatus?:  (stage: string) => void,
   *   onMessage?: (data: {text: string, topic?: string, mode?: string, confidence?: number}) => void,
   *   onCard?:    (data: {html, css, scope_id, title, accent_color, background_color, tool_name}) => void,
   *   onError?:   (data: {message: string, recoverable: boolean}) => void,
   *   onDone?:    (data: {duration_ms: number}) => void,
   * }} callbacks
   */
  async send(text, source, callbacks = {}) {
    this.abort();
    this._abortController = new AbortController();

    let response;
    try {
      response = await fetch(this._buildUrl('/chat'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ text, source }),
        signal: this._abortController.signal,
      });
    } catch (err) {
      if (err.name === 'AbortError') return;
      callbacks.onError?.({ message: 'Network error. Please try again.', recoverable: true });
      callbacks.onDone?.({ duration_ms: 0 });
      return;
    }

    if (response.status === 401) {
      callbacks.onError?.({ message: 'Session expired. Please log in again.', recoverable: false });
      callbacks.onDone?.({ duration_ms: 0 });
      return;
    }

    if (!response.ok) {
      callbacks.onError?.({ message: 'Server error. Please try again.', recoverable: true });
      callbacks.onDone?.({ duration_ms: 0 });
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep incomplete line

        let eventType = null;
        let eventData = null;

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            eventData = line.slice(6);
          } else if (line.startsWith(':')) {
            // Comment / keepalive — ignore
            continue;
          } else if (line === '' && eventType && eventData) {
            this._dispatch(eventType, eventData, callbacks);
            eventType = null;
            eventData = null;
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        callbacks.onError?.({ message: 'Connection lost.', recoverable: true });
        callbacks.onDone?.({ duration_ms: 0 });
      }
    } finally {
      this._abortController = null;
    }
  }

  _dispatch(eventType, rawData, callbacks) {
    let data;
    try {
      data = JSON.parse(rawData);
    } catch {
      return;
    }

    switch (eventType) {
      case 'status':
        callbacks.onStatus?.(data.stage);
        break;
      case 'message':
        callbacks.onMessage?.(data);
        break;
      case 'card':
        callbacks.onCard?.(data);
        break;
      case 'error':
        callbacks.onError?.(data);
        break;
      case 'done':
        callbacks.onDone?.(data);
        break;
    }
  }
}

/**
 * WebSocket drift/push event dispatcher.
 * Thin routing layer — fans out events to module callbacks.
 */
export class EventRouter {
  constructor({ ws, renderer }) {
    this._ws = ws;
    this._renderer = renderer;

    this._onUpdate = null;
    this._onTask = null;
    this._onImageReady = null;
    this._onNotification = null;
    this._onBackgroundContent = null;
    this._onCapabilityAlert = null;
    this._onPermissionRequest = null;
    this._onQuickTip = null;
    this._isSendingGetter = () => false;
  }

  /** Register handler for 'app_update' events. */
  onUpdateEvent(cb) { this._onUpdate = cb; }

  /** Register handler for 'task' events. */
  onTaskEvent(cb) { this._onTask = cb; }

  /** Register handler for 'image_ready' events. */
  onImageReady(cb) { this._onImageReady = cb; }

  /** Register handler for 'notification' events. */
  onNotification(cb) { this._onNotification = cb; }

  /** Register handler for 'capability_alert' events. */
  onCapabilityAlert(cb) { this._onCapabilityAlert = cb; }

  /** Register handler for 'permission_request' events. */
  onPermissionRequest(cb) { this._onPermissionRequest = cb; }

  /** Register handler for 'quick_tip' events. */
  onQuickTip(cb) { this._onQuickTip = cb; }

  /**
   * Register handler for background drift/response/escalation content.
   * Called when the tab is not focused and a renderable event arrives.
   */
  onBackgroundContent(cb) { this._onBackgroundContent = cb; }

  /**
   * Provide a getter that returns true while a sync /chat send is in-flight.
   * The router uses this to skip duplicate rendering of 'response' events.
   */
  setIsSendingGetter(fn) { this._isSendingGetter = fn; }

  /** Wire the drift handler and open the WebSocket connection. */
  init() {
    this._ws.onDrift((data) => this._handleEvent(data));
    this._ws.connect();
  }

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  _handleEvent(data) {
    // App update notification
    if (data.type === 'app_update') {
      if (this._onUpdate) this._onUpdate(data);
      return;
    }

    // Task progress/completion — refresh the task strip
    if (data.type === 'task') {
      if (this._onTask) this._onTask(data);
      return;
    }

    if (data.type === 'image_ready') {
      if (this._onImageReady) this._onImageReady(data);
      return;
    }

    if (data.type === 'capability_alert') {
      if (this._onCapabilityAlert) this._onCapabilityAlert(data);
      return;
    }

    if (data.type === 'permission_request') {
      if (this._onPermissionRequest) this._onPermissionRequest(data);
      return;
    }

    if (data.type === 'quick_tip') {
      if (this._onQuickTip) this._onQuickTip(data);
      return;
    }

    const content = data.content || '';
    if (!content) return;

    // Proactive thought card — render directly in the conversation spine.
    // Handled before the 'response' sending-guard so thought cards always
    // appear regardless of whether a sync /chat request is in flight.
    if (data.type === 'thought') {
      const meta = {
        topic: data.topic,
        type: data.type,
        ts: new Date(),
        mode: data.mode || '',
        confidence: data.confidence || 0,
      };
      this._renderer.appendChalieForm(content, meta);
      return;
    }

    // Ignore 'response' events from the drift stream while a /chat SSE request
    // is in flight — the chat SSE already renders the reply via replaceActWithResponse.
    if (data.type === 'response' && this._isSendingGetter()) return;

    // System notification + sound when tab is not focused
    if (!document.hasFocus()) {
      // Extract plaintext from XML for background notification
      import('./markup_extract.js').then(({ extractPlaintext }) => {
        const notifText = extractPlaintext(content);
        if (notifText && this._onBackgroundContent) this._onBackgroundContent(notifText);
      }).catch(() => {});
    }

    // Scheduler trigger events — refresh task strip, play sound
    if (data.type === 'notification') {
      if (this._onNotification) this._onNotification(data);
      return;
    }

    const meta = {
      topic: data.topic,
      type: data.type,
      ts: new Date(),
      mode: data.mode || '',
      confidence: data.confidence || 0,
    };

    // Render in conversation spine as a Chalie message
    const formEl = this._renderer.appendChalieForm(content, meta);

    // Critic escalation — amber border to signal "needs your attention"
    if (data.type === 'escalation') {
      formEl.classList.add('speech-form--escalation');
    }
  }
}

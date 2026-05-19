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
    if (this._routeSimpleEvent(data)) return;

    const content = data.content || '';
    if (!content) return;

    if (this._routeThoughtEvent(data, content)) return;

    // Ignore 'response' events from the drift stream while a /chat SSE request
    // is in flight — the chat SSE already renders the reply via replaceActWithResponse.
    if (data.type === 'response' && this._isSendingGetter()) return;

    this._notifyBackground(content);

    if (this._routeNotificationEvent(data)) return;

    this._renderContentEvent(data, content);
  }

  /**
   * Route event types that require no content — app_update, task,
   * capability_alert, permission_request, quick_tip.
   * Returns true when the event was handled.
   */
  _routeSimpleEvent(data) {
    switch (data.type) {
      case 'app_update':
        if (this._onUpdate) this._onUpdate(data);
        return true;
      case 'task':
        if (this._onTask) this._onTask(data);
        return true;
      case 'capability_alert':
        if (this._onCapabilityAlert) this._onCapabilityAlert(data);
        return true;
      case 'permission_request':
        if (this._onPermissionRequest) this._onPermissionRequest(data);
        return true;
      case 'quick_tip':
        if (this._onQuickTip) this._onQuickTip(data);
        return true;
      default:
        return false;
    }
  }

  /**
   * Handle proactive thought cards — rendered before the sending-guard so
   * they always appear regardless of whether a /chat request is in flight.
   * Returns true when the event was handled.
   */
  _routeThoughtEvent(data, content) {
    if (data.type !== 'thought') return false;
    const meta = {
      topic: data.topic,
      type: data.type,
      ts: new Date(),
      mode: data.mode || '',
      confidence: data.confidence || 0,
    };
    this._renderer.appendChalieForm(content, meta);
    return true;
  }

  /**
   * Fire the background-content callback when the tab is not focused so the
   * host page can show a system notification / play a sound.
   */
  _notifyBackground(content) {
    if (document.hasFocus()) return;
    import('./markup_extract.js').then(({ extractPlaintext }) => {
      const notifText = extractPlaintext(content);
      if (notifText && this._onBackgroundContent) this._onBackgroundContent(notifText);
    }).catch(() => {});
  }

  /**
   * Handle scheduler notification events (refresh task strip, play sound).
   * Returns true when the event was handled.
   */
  _routeNotificationEvent(data) {
    if (data.type !== 'notification') return false;
    if (this._onNotification) this._onNotification(data);
    return true;
  }

  /**
   * Render a response/escalation/drift event into the conversation spine.
   */
  _renderContentEvent(data, content) {
    const meta = {
      topic: data.topic,
      type: data.type,
      ts: new Date(),
      mode: data.mode || '',
      confidence: data.confidence || 0,
    };
    const formEl = this._renderer.appendChalieForm(content, meta);

    // Critic escalation — amber border to signal "needs your attention"
    if (data.type === 'escalation') {
      formEl.classList.add('speech-form--escalation');
    }
  }
}

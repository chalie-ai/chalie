/**
 * Reminders card — TODO placeholder.
 * Will render a calendar-style card when GET /reminders endpoint is available.
 */
import { BaseCard } from './base.js';

export class RemindersCard extends BaseCard {
  constructor(api, renderer) {
    super(api, renderer);
  }

  async fetch() {
    // TODO: Implement when GET /reminders endpoint is available
    console.debug('RemindersCard: endpoint not yet available');
  }

  template(data) {
    // TODO: Calendar-style card layout
    return '<div>Reminders coming soon.</div>';
  }
}

/**
 * Digest card — TODO placeholder.
 * Will render an inbox-style message digest with sender hierarchy and
 * preview snippets when GET /digest endpoint is available.
 */
import { BaseCard } from './base.js';

export class DigestCard extends BaseCard {
  constructor(api, renderer) {
    super(api, renderer);
  }

  async fetch() {
    // TODO: Implement when GET /digest endpoint is available
    console.debug('DigestCard: endpoint not yet available');
  }

  template(data) {
    // TODO: Inbox-style message digest layout
    return '<div>Digest coming soon.</div>';
  }
}

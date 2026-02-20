/**
 * Weather card — TODO placeholder.
 * Will render an atmospheric card with large temp display and sky gradient
 * when GET /weather endpoint is available.
 */
import { BaseCard } from './base.js';

export class WeatherCard extends BaseCard {
  constructor(api, renderer) {
    super(api, renderer);
  }

  async fetch() {
    // TODO: Implement when GET /weather endpoint is available
    console.debug('WeatherCard: endpoint not yet available');
  }

  template(data) {
    // TODO: Atmospheric card with sky gradient background
    return '<div>Weather coming soon.</div>';
  }
}

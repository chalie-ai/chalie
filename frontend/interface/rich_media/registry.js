/**
 * Rich-media card registry.
 *
 * Maps tag prefix (e.g. "weather") to the module that renders the card.
 * Each module exports render(payload, synthesis, root).
 *
 * To add a new card type: import the module and add a key below.
 * No framework changes required.
 */

import * as weatherModule from './weather.js';

const REGISTRY = {
  weather: weatherModule,
  // search: searchModule,   — future
  // browser: browserModule, — future
};

/**
 * Render a rich-media card or fall back to a plain text bubble.
 *
 * @param {string}      tag       - Full tag string, e.g. "weather_1"
 * @param {object}      payload   - Parsed tool payload dict
 * @param {string}      synthesis - LLM synthesis text
 * @param {HTMLElement} root      - Container the card DOM is mounted into
 * @param {Function}    [appendTextBubble] - Fallback when prefix is not registered;
 *                                           called with (synthesis) when present.
 */
export function renderCard(tag, payload, synthesis, root, appendTextBubble) {
  const prefix = tag.split('_')[0];
  const mod = REGISTRY[prefix];
  if (!mod) {
    if (appendTextBubble && synthesis) {
      appendTextBubble(synthesis);
    }
    return;
  }
  mod.render(payload, synthesis, root);
}

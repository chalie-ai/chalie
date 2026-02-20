/**
 * Presence dot state machine.
 * CSS handles all visual animation via [data-state] selectors.
 */

const LABELS = {
  processing:       'Working...',
  thinking:         'Thinking...',
  retrieving_memory:'Remembering...',
  planning:         'Planning...',
  responding:       'Speaking...',
  still_working:    'Still working...',
  resting:          'Chalie',
  error:            'Connection lost',
};

export class Presence {
  /**
   * @param {HTMLElement} dotEl   — the .presence-dot element
   * @param {HTMLElement} labelEl — the .presence-label element
   */
  constructor(dotEl, labelEl) {
    this._dot = dotEl;
    this._label = labelEl;
    this._state = 'resting';
  }

  /** Current state name. */
  get state() {
    return this._state;
  }

  /**
   * Transition to a new state.
   * @param {string} state — one of the keys in LABELS
   */
  setState(state) {
    if (!LABELS[state]) return;
    this._state = state;
    this._dot.setAttribute('data-state', state);
    this._label.textContent = LABELS[state];
  }
}

/**
 * Transient toast notification.
 * Ported from frontend/interface/utils.js — exact DOM structure and CSS class
 * names preserved so the existing toast CSS applies.
 */

/**
 * Show a transient toast notification with an optional undo action.
 *
 * @param message  Text to display.
 * @param onUndo   Optional callback; if provided, an "Undo" button is rendered.
 * @param duration Visible duration in ms (default 4000). After this the toast
 *                 fades out and is removed from the DOM.
 */
export function showToast(
  message: string,
  onUndo?: (() => void) | null,
  duration = 4_000,
): void {
  // Remove any existing toast so only one is visible at a time.
  document.querySelector('.chalie-toast')?.remove();

  const toast = document.createElement('div');
  toast.className = 'chalie-toast';

  const span = document.createElement('span');
  span.textContent = message;
  toast.appendChild(span);

  if (onUndo) {
    const undoBtn = document.createElement('button');
    undoBtn.className = 'chalie-toast__undo';
    undoBtn.textContent = 'Undo';
    undoBtn.addEventListener('click', () => {
      onUndo();
      toast.remove();
    });
    toast.appendChild(undoBtn);
  }

  document.body.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('chalie-toast--visible'));

  setTimeout(() => {
    toast.classList.remove('chalie-toast--visible');
    setTimeout(() => toast.remove(), 250);
  }, duration);
}

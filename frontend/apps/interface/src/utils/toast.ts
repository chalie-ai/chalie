// Transient toast notification with an optional undo button. DOM structure and
// CSS class names match the existing toast CSS.
export function showToast(
  message: string,
  onUndo?: (() => void) | null,
  duration = 4_000,
): void {
  // Keep only one toast visible at a time.
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

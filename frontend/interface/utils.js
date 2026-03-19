/**
 * Shared utilities for the Chalie interface.
 */

// Safe localStorage wrappers — private browsing on iOS Safari / Firefox throws SecurityError.
export function lsGet(key) { try { return localStorage.getItem(key); } catch { return null; } }
export function lsSet(key, val) { try { localStorage.setItem(key, val); } catch { /* ignore */ } }

/** Escape HTML entities. */
export function escHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/** Convert an ISO date string to a short relative label ("in 5m", "in 2h", "tomorrow"). */
export function relativeTime(isoStr) {
  try {
    const due = new Date(isoStr);
    const now = Date.now();
    const diffMs = due.getTime() - now;
    if (diffMs < 0) return 'overdue';
    const mins = Math.round(diffMs / 60000);
    if (mins < 1) return 'now';
    if (mins < 60) return `in ${mins}m`;
    const hrs = Math.round(mins / 60);
    if (hrs < 24) return `in ${hrs}h`;
    const days = Math.round(hrs / 24);
    if (days === 1) return 'tomorrow';
    return `in ${days}d`;
  } catch {
    return '';
  }
}

/** Show a transient toast notification with an optional undo action. */
export function showToast(message, onUndo, duration = 4000) {
  // Remove existing toast
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

/**
 * Toast notification composable.
 *
 * Port of legacy BrainApp.showToast() (app.js:73-91).
 * Reactive list consumed by ToastHost.vue.
 */
import { ref } from 'vue';

export type ToastType = 'info' | 'success' | 'error' | 'warning';

export interface Toast {
  id: number;
  message: string;
  type: ToastType;
  action?: string;
  onAction?: () => void;
  duration: number;
}

let _nextId = 0;
const toasts = ref<Toast[]>([]);

export function useToast() {
  function show(
    message: string,
    type: ToastType = 'info',
    opts: { action?: string; onAction?: () => void; duration?: number } = {},
  ): void {
    const id = ++_nextId;
    const duration = opts.duration ?? 3200;
    toasts.value.push({
      id,
      message,
      type,
      action: opts.action,
      onAction: opts.onAction,
      duration,
    });
    setTimeout(() => dismiss(id), duration + 300); // +300 for fade-out
  }

  function dismiss(id: number): void {
    const idx = toasts.value.findIndex((t) => t.id === id);
    if (idx !== -1) toasts.value.splice(idx, 1);
  }

  return { toasts, show, dismiss };
}

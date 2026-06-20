/** Promise-based confirm dialog; ConfirmDialog.vue renders the reactive state. */
import { ref, shallowRef } from 'vue';

export interface ConfirmOptions {
  title: string;
  desc: string;
  confirmLabel?: string;
  confirmClass?: string;
}

interface PendingConfirm extends ConfirmOptions {
  resolve: (value: boolean) => void;
}

const pending = shallowRef<PendingConfirm | null>(null);
const visible = ref(false);

export function useConfirm() {
  function confirm(opts: ConfirmOptions): Promise<boolean> {
    return new Promise((resolve) => {
      pending.value = { ...opts, resolve };
      visible.value = true;
    });
  }

  function accept(): void {
    pending.value?.resolve(true);
    close();
  }

  function cancel(): void {
    pending.value?.resolve(false);
    close();
  }

  function close(): void {
    visible.value = false;
    pending.value = null;
  }

  return { visible, pending, confirm, accept, cancel };
}

import { onMounted, ref, type Ref } from 'vue';

export interface AsyncResource<T> {
  data: Ref<T>;
  loading: Ref<boolean>;
  /** Error from the most recent fetch, or `null` if it succeeded. */
  error: Ref<unknown>;
  reload: () => Promise<void>;
}

export interface AsyncResourceOptions<T> {
  /** Seed value for `data` before the first fetch resolves. */
  initial: T;
  /** Fetch once on mount. Default `true`; `false` = manual via `reload()`. */
  immediate?: boolean;
  /** Imperative side-effect for a failed fetch; the error is also on `error`. */
  onError?: (e: unknown) => void;
}

/**
 * The loading/error/data + try/catch/finally `reload()` lifecycle every list/read
 * panel hand-rolls, so panels declare *what* they fetch, not the bookkeeping.
 * Must be called from `setup()` (registers `onMounted` when `immediate`).
 */
export function useAsyncResource<T>(
  fetcher: () => Promise<T>,
  opts: AsyncResourceOptions<T>,
): AsyncResource<T> {
  const data = ref(opts.initial) as Ref<T>;
  // Seed `true` for immediate fetches so the first render shows loading, not a
  // flash of empty data before onMounted fires.
  const loading = ref(opts.immediate !== false);
  const error = ref<unknown>(null);

  async function reload(): Promise<void> {
    loading.value = true;
    error.value = null;
    try {
      data.value = await fetcher();
    } catch (e) {
      error.value = e;
      opts.onError?.(e);
    } finally {
      loading.value = false;
    }
  }

  if (opts.immediate !== false) onMounted(reload);

  return { data, loading, error, reload };
}

import { onMounted, ref, type Ref } from 'vue';

export interface AsyncResource<T> {
  /** The last successfully fetched value (starts at `initial`). */
  data: Ref<T>;
  /** True while a fetch is in flight. */
  loading: Ref<boolean>;
  /** The error thrown by the most recent fetch, or `null` if the last fetch succeeded. */
  error: Ref<unknown>;
  /** Re-run the fetcher. Resolves once `loading` flips back to false. */
  reload: () => Promise<void>;
}

export interface AsyncResourceOptions<T> {
  /** Seed value for `data` before the first fetch resolves (e.g. `[]`). */
  initial: T;
  /** Fetch once on mount. Default `true`. Set `false` to fetch manually via `reload()`. */
  immediate?: boolean;
  /**
   * Side-effect for a failed fetch (e.g. show a toast). The error is also exposed
   * on `error` regardless — `onError` is purely for imperative reactions.
   */
  onError?: (e: unknown) => void;
}

/**
 * The async-resource lifecycle every list/read panel hand-rolls: a `loading`
 * flag, an `error` slot, a `data` ref seeded with `initial`, and a `reload()`
 * that wraps the fetch in try/catch/finally. Extracted so panels declare *what*
 * they fetch, not the bookkeeping around it.
 *
 * ```ts
 * const { data: tools, loading, error } = useAsyncResource(
 *   async () => (await cognition.tools()).tools ?? [],
 *   { initial: [] as Tool[] },
 * );
 * ```
 *
 * Must be called from a component `setup()` (it registers `onMounted` when
 * `immediate`). It is deliberately UI-agnostic: brain panels wire a toast via
 * `onError` (see `useBrainResource`), read-only sub-views read `error` directly
 * to render an empty/failed state.
 */
export function useAsyncResource<T>(
  fetcher: () => Promise<T>,
  opts: AsyncResourceOptions<T>,
): AsyncResource<T> {
  const data = ref(opts.initial) as Ref<T>;
  // Seed `true` for immediate fetches so the first render shows the loading
  // state (matching the hand-rolled `ref(true)`), not a flash of empty data
  // before onMounted fires.
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

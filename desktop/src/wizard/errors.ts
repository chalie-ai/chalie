// One sentence per failure the Rust side can report, so every dead end says what happened
// and what to do about it instead of showing a raw error.

export type AppErrorKind =
  | 'invalid_address'
  | 'unreachable'
  | 'not_chalie'
  | 'invalid_credentials'
  | 'rate_limited'
  | 'server_failed'
  | 'config'
  | 'webview'
  | 'install_failed'
  | 'install_running'
  | 'not_ready'
  | 'account_exists'
  | 'invalid_account';

export interface AppError {
  kind: AppErrorKind;
  message?: string;
  retry_after_seconds?: number | null;
  status?: number;
}

const SENTENCES: Record<AppErrorKind, string> = {
  invalid_address: 'The host field takes only a host name or IP address — no scheme, path or port.',
  unreachable: 'Nothing answered at that address. Check that Chalie is running and that the host and port are right.',
  not_chalie: 'Something answered, but it is not a Chalie server.',
  invalid_credentials: 'That username and password were refused.',
  rate_limited: 'Too many login attempts. Wait a moment and try again.',
  server_failed: 'Chalie answered with an error of its own.',
  config: "The app's settings file could not be read or written.",
  webview: 'The window would not take the session, so the Chalie interface was not opened.',
  install_failed: 'The install did not finish. The detail below says which command stopped it and where to read everything it printed.',
  install_running: 'An install is already running. Wait for it to finish.',
  not_ready: 'Chalie was installed and started, but never began answering.',
  account_exists: 'That Chalie already has an account, so there is nothing to create. Sign in to it instead.',
  invalid_account: 'Those account details were refused.',
};

const UNKNOWN = 'Something went wrong.';

// The same two kinds read differently when they come out of starting this Mac's own Chalie
// instead of installing it — nothing was being installed, so the install-flavoured sentence
// would be false. describeError's context argument swaps these in without touching the kinds
// or the detail, which the Rust side attaches the same way either time.
const START_SENTENCES: Partial<Record<AppErrorKind, string>> = {
  install_failed:
    'Chalie did not start. The detail below says which command stopped it and where to read everything it printed.',
  not_ready: 'Chalie was started, but never began answering.',
};

export function asAppError(error: unknown): AppError | null {
  if (typeof error !== 'object' || error === null) return null;
  const kind = (error as { kind?: unknown }).kind;
  return typeof kind === 'string' && kind in SENTENCES ? (error as AppError) : null;
}

/** The sentence for this failure, followed by whatever detail the Rust side attached.
 * `context` picks the wording for the handful of kinds a start and an install both raise;
 * everything else reads the same either way. */
export function describeError(
  error: unknown,
  context: 'install' | 'start' = 'install',
): { text: string; detail: string } {
  const appError = asAppError(error);
  if (!appError) return { text: UNKNOWN, detail: String(error) };

  let text =
    (context === 'start' && START_SENTENCES[appError.kind]) || SENTENCES[appError.kind];
  if (appError.kind === 'rate_limited' && appError.retry_after_seconds != null) {
    text = `Too many login attempts. Try again in ${appError.retry_after_seconds} seconds.`;
  }
  const detail =
    appError.message ?? (appError.status === undefined ? '' : `HTTP ${appError.status}`);
  return { text, detail };
}

export function isKind(error: unknown, kind: AppErrorKind): boolean {
  return asAppError(error)?.kind === kind;
}

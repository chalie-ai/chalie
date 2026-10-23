// Everything the wizard is allowed to ask the Rust side, and the shapes it answers with.
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import type { UnlistenFn } from '@tauri-apps/api/event';

// Passed straight back out by the listen helpers below, so a component that keeps one never
// has to reach past this file for its type.
export type { UnlistenFn };

/** The port a Chalie server listens on unless its owner changed it. */
export const DEFAULT_PORT = 31025;

export type VaultState = 'unlocked' | 'locked' | 'uninitialized' | 'unknown';

/** The saved setup. The server's fields are absent until this Mac has been connected once. */
export interface SetupState {
  host?: string;
  port?: number;
}

/** `found` says a Chalie answered on this Mac; `has_master_account` says it can be signed
 * in to, which the one-click local path has to tell apart. */
export interface LocalInstance {
  found: boolean;
  has_master_account: boolean;
  host: string;
  port: number;
}

export interface ServerStatus {
  has_master_account: boolean;
  vault_state: VaultState;
}

/** Which part of an install is happening now, in the order they are reached. */
export type InstallPhase =
  | 'installing'
  | 'starting'
  | 'waiting'
  | 'creating_account'
  | 'connecting';

export interface AutoConnectResult {
  needs_credentials: boolean;
  username: string | null;
}

export function getSetupState(): Promise<SetupState> {
  return invoke<SetupState>('get_setup_state');
}

export function detectLocal(): Promise<LocalInstance> {
  return invoke<LocalInstance>('detect_local');
}

export function probeServer(host: string, port: number): Promise<ServerStatus> {
  return invoke<ServerStatus>('probe_server', { host, port });
}

export function connect(
  host: string,
  port: number,
  username: string,
  password: string,
): Promise<void> {
  return invoke<void>('connect', { host, port, username, password });
}

export function autoConnect(): Promise<AutoConnectResult> {
  return invoke<AutoConnectResult>('auto_connect');
}

/** Install a Chalie on this Mac and sign in to it. Resolves only once the window has been
 * handed the new session, by which time this page is on its way out. */
export function installLocal(username: string, password: string): Promise<void> {
  return invoke<void>('install_local', { username, password });
}

/** Give a Chalie that is already running its first account, and sign in to it. */
export function createAccount(
  host: string,
  port: number,
  username: string,
  password: string,
): Promise<void> {
  return invoke<void>('create_account', { host, port, username, password });
}

/** Every line the install's commands print, as each one arrives. */
export function onInstallOutput(show: (line: string) => void): Promise<UnlistenFn> {
  return listen<string>('install-output', (event) => show(event.payload));
}

export function onInstallPhase(show: (phase: InstallPhase) => void): Promise<UnlistenFn> {
  return listen<InstallPhase>('install-phase', (event) => show(event.payload));
}

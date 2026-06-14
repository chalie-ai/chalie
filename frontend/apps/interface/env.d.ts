/// <reference types="vite/client" />
declare module '*.vue' {
  import type { DefineComponent } from 'vue';
  const component: DefineComponent<Record<string, never>, Record<string, never>, unknown>;
  export default component;
}

/**
 * Type shim for the vendored linkifyjs ESM bundle (linkifyjs@4.3.2).
 * The vendor file is a pre-built minified ESM module with no bundled types.
 */
declare module '*/vendor/linkify.es.mjs' {
  export interface LinkMatch {
    type: string;
    value: string;
    isLink: boolean;
    href: string;
    start: number;
    end: number;
  }
  /** Find all links in a string. */
  export function find(text: string, type?: string | null): LinkMatch[];
  export function test(text: string, type?: string | null): boolean;
}

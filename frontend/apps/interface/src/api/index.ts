/**
 * Typed endpoint wrappers for the Chalie interface app.
 *
 * The generic ApiClient lives in @chalie/shared. This module composes
 * domain-namespaced wrappers over it so callers get typed request/response
 * shapes for every REST endpoint the interface uses.
 *
 * Note: /chat, /chat/interrupt, and /action are owned by WebSocketService /
 * stores/session.ts and are NOT exposed here.
 */

export { conversation } from './conversation';
export { moments } from './moments';
export { voice } from './voice';
export { documents } from './documents';
export { scheduler } from './scheduler';
export { system } from './system';
export { tips } from './tips';
export { policies } from './policies';

export { getHost } from '@chalie/shared';

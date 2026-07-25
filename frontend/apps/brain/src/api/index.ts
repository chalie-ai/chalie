/**
 * Brain API barrel — domain endpoint wrappers over the shared ApiClient.
 *
 * Each module wraps a domain's endpoints with typed request/response shapes.
 * 401 → /login/ redirect is handled centrally by the shared ApiClient.
 */
export { system } from './system';
export { providers } from './providers';
export { cognition } from './cognition';
export { scheduler } from './scheduler';
export { lists } from './lists';
export { capabilities } from './capabilities';
export { policies } from './policies';
export { skills } from './skills';
export { mcp } from './mcp';
export { snapshot } from './snapshot';

export { getHost } from '@chalie/shared';

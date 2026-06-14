/**
 * Brain API barrel — domain endpoint wrappers over the shared ApiClient.
 *
 * Each module wraps a domain's endpoints with typed request/response shapes.
 * All modules use withAuth() from http.ts to handle 401 → /login/ redirect.
 */
export { system } from './system';
export { providers } from './providers';
export { cognition } from './cognition';
export { scheduler } from './scheduler';
export { lists } from './lists';
export { documents } from './documents';
export { capabilities } from './capabilities';
export { policies } from './policies';
export { skills } from './skills';
export { mcp } from './mcp';
export { brain } from './brain';

export { getHost } from '@chalie/shared';

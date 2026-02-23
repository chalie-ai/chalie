/**
 * Chalie Service Worker — App Shell caching + Web Push notifications.
 */

// Cache version: use deploy date for easy debugging (bump on each release)
const CACHE_VERSION = '2026.02.23';
const SHELL_CACHE = `chalie-shell-${CACHE_VERSION}`;
const CDN_CACHE = `chalie-cdn-${CACHE_VERSION}`;

const SHELL_ASSETS = [
  '/', '/index.html', '/manifest.json',
  '/app.js', '/api.js', '/sse.js', '/renderer.js',
  '/presence.js', '/voice.js', '/tools.js', '/style.css',
  '/markdown.js', '/lib/marked.esm.js',
  '/icons/icon.png',
  '/cards/base.js', '/cards/memory.js', '/cards/timeline.js',
  '/cards/reminders.js', '/cards/weather.js', '/cards/digest.js',
];

const NETWORK_ONLY_PATHS = [
  '/chat', '/conversation', '/memory', '/proactive',
  '/health', '/system', '/privacy', '/push',
  '/tools/', '/events', '/metrics',
];
// Note: '/tools/' (with slash) avoids matching '/tools.js'

const NETWORK_ONLY_HOSTS = ['stt.grck.org', 'tts.grck.org'];

const CDN_HOSTS = ['fonts.googleapis.com', 'fonts.gstatic.com', 'cdnjs.cloudflare.com'];

// ============================================================================
// Install: Populate shell cache
// ============================================================================

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => {
      return cache.addAll(SHELL_ASSETS).then(() => {
        // Skip waiting to activate immediately
        return self.skipWaiting();
      });
    })
  );
});

// ============================================================================
// Activate: Clean old caches + enable navigation preload + claim clients
// ============================================================================

self.addEventListener('activate', (event) => {
  event.waitUntil(
    // 1. Delete caches not in [SHELL_CACHE, CDN_CACHE]
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => ![SHELL_CACHE, CDN_CACHE].includes(name))
          .map((name) => caches.delete(name))
      );
    }).then(() => {
      // 2. Enable navigation preload (speeds cold-start navigations)
      if (self.registration.navigationPreload) {
        return self.registration.navigationPreload.enable();
      }
    }).then(() => {
      // 3. Claim all clients immediately
      return self.clients.claim();
    })
  );
});

// ============================================================================
// Fetch: Routing logic
// ============================================================================

self.addEventListener('fetch', (event) => {
  // Only handle GET requests
  if (event.request.method !== 'GET') {
    return;
  }

  const { origin, pathname, host } = new URL(event.request.url);

  // 1. Network-only hosts: pass through (never use preload)
  if (NETWORK_ONLY_HOSTS.some((h) => host === h)) {
    return;
  }

  // 2. Same-origin network-only paths: pass through (never use preload)
  if (
    origin === self.location.origin &&
    NETWORK_ONLY_PATHS.some((p) => pathname.startsWith(p))
  ) {
    return;
  }

  // 3. Same-origin non-API GET: cache-first with stale-while-revalidate
  if (origin === self.location.origin) {
    return event.respondWith(
      cacheFirst(event.request, SHELL_CACHE, event.preloadResponse)
    );
  }

  // 4. CDN hosts: cache-first
  if (CDN_HOSTS.some((h) => host === h)) {
    return event.respondWith(cacheFirst(event.request, CDN_CACHE, null));
  }

  // 5. Everything else: pass through (network)
});

// ============================================================================
// Cache-first strategy with stale-while-revalidate + navigation preload
// ============================================================================

/**
 * @param {Request} request
 * @param {string} cacheName
 * @param {Promise<Response>|null} preloadResponse
 * @returns {Promise<Response>}
 */
async function cacheFirst(request, cacheName, preloadResponse) {
  const cache = await caches.open(cacheName);

  // 1. Check cache first
  let response = await cache.match(request);
  if (response) {
    // Cache hit: return immediately, but fire a background update
    backgroundFetch(request, cacheName);
    return response;
  }

  // 2. Cache miss: try preload response (if available)
  if (preloadResponse) {
    try {
      response = await preloadResponse;
      if (response && response.status === 200 && canCacheResponse(response)) {
        // Store in cache before returning
        cache.put(request, response.clone());
      }
      if (response) {
        return response;
      }
    } catch {
      // Preload failed; fall through to fetch
    }
  }

  // 3. Network fetch
  try {
    response = await fetch(request);

    // Cache successful same-origin responses (status 200, type 'basic')
    // and CDN responses (type 'cors')
    if (response && response.status === 200 && canCacheResponse(response)) {
      cache.put(request, response.clone());
    }

    return response;
  } catch (err) {
    // Network failed. If this is a navigation request, return cached /index.html
    // to prevent browser error page.
    if (request.mode === 'navigate') {
      const fallback = await cache.match('/index.html');
      if (fallback) {
        return fallback;
      }
    }

    throw err;
  }
}

/**
 * Background fetch: update cache without blocking the current response
 */
function backgroundFetch(request, cacheName) {
  fetch(request).then((response) => {
    if (response && response.status === 200 && canCacheResponse(response)) {
      const cache = caches.open(cacheName);
      cache.then((c) => c.put(request, response));
    }
  }).catch(() => {
    // Ignore network errors in background
  });
}

/**
 * Guard against cache poisoning: only cache responses that are safe.
 * Returns true if response should be cached.
 */
function canCacheResponse(response) {
  // Never cache error codes
  if (response.status !== 200) {
    return false;
  }

  // Only cache same-origin (type 'basic') or cross-origin CDN (type 'cors')
  // Never cache opaque responses (type 'opaque')
  const type = response.type;
  return type === 'basic' || type === 'cors';
}

// ============================================================================
// Push notifications (existing handlers)
// ============================================================================

self.addEventListener('push', (event) => {
  if (!event.data) return;

  let payload;
  try {
    payload = event.data.json();
  } catch {
    payload = { title: 'Chalie', body: event.data.text() };
  }

  const title = payload.title || 'Chalie';
  const options = {
    body: payload.body || '',
    icon: payload.icon || undefined,
    tag: payload.tag || 'chalie-drift',
    data: { url: payload.url || '/' },
  };

  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  const url = event.notification.data?.url || '/';

  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then((windowClients) => {
      // Focus existing tab if open
      for (const client of windowClients) {
        if (client.url.includes(self.location.origin) && 'focus' in client) {
          return client.focus();
        }
      }
      // Otherwise open a new tab
      return clients.openWindow(url);
    })
  );
});

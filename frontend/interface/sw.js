/**
 * Chalie Service Worker — Web Push + Share Target only.
 *
 * Intentionally does NOT cache assets. Every request goes to the network,
 * so a simple refresh always reflects the latest deploy. Conditional
 * requests (ETag / Last-Modified) handled by the browser's HTTP cache
 * against Flask's static handler are sufficient to avoid wasted bytes.
 *
 * On activation, any caches left behind by earlier SW versions are deleted
 * so returning users immediately drop stale precached shells.
 */

globalThis.addEventListener('install', (event) => {
  event.waitUntil(globalThis.skipWaiting());
});

globalThis.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((names) => Promise.all(names.map((n) => caches.delete(n))))
      .then(() => globalThis.clients.claim())
  );
});

globalThis.addEventListener('fetch', (event) => {
  // Share Target: intercept POST to /share-target and redirect with data
  if (event.request.method === 'POST' && new URL(event.request.url).pathname === '/share-target') {
    event.respondWith(
      event.request.formData().then((formData) => {
        const title = formData.get('title') || '';
        const text = formData.get('text') || '';
        const url = formData.get('url') || '';
        const parts = [title, text, url].filter(Boolean);
        const shared = encodeURIComponent(parts.join(' '));
        return Response.redirect(`/?shared=${shared}`, 303);
      }).catch(() => Response.redirect('/', 303))
    );
  }
  // All other requests fall through to the network — no SW caching.
});

// ============================================================================
// Push notifications
// ============================================================================

globalThis.addEventListener('push', (event) => {
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
    tag: payload.tag || 'chalie-message',
    data: { url: payload.url || '/' },
  };

  event.waitUntil(globalThis.registration.showNotification(title, options));
});

globalThis.addEventListener('notificationclick', (event) => {
  event.notification.close();

  const url = event.notification.data?.url || '/';

  event.waitUntil(
    globalThis.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((windowClients) => {
      for (const client of windowClients) {
        if (client.url.includes(globalThis.location.origin) && 'focus' in client) {
          return client.focus();
        }
      }
      return globalThis.clients.openWindow(url);
    })
  );
});

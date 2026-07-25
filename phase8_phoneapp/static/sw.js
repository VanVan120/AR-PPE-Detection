/* Service worker — the small amount of it that makes this installable.
 *
 * Its job is NOT offline use: the app is useless without the laptop that runs the model.
 * It exists so the browser will offer "Install app" at all (an installable PWA must have
 * a fetch handler), and so the shell still paints something explanatory when the laptop
 * is unreachable instead of the browser's dinosaur.
 *
 * Network-FIRST, deliberately. A cache-first shell would pin whatever token and code were
 * current at install time, and the app would then keep 403ing after the laptop's key was
 * rotated, with no way for the user to tell why. Fresh-when-reachable, cached-only-as-a-
 * fallback has none of that failure mode.
 */
"use strict";

const CACHE = "ar-safety-shell-v1";

self.addEventListener("install", () => self.skipWaiting());

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names.filter((n) => n !== CACHE).map((n) => caches.delete(n)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;                 // frames and uploads are POSTs
  let url;
  try { url = new URL(req.url); } catch (err) { return; }
  if (url.origin !== self.location.origin) return;
  // Live data must never be served from a cache: a cached status or report would show a
  // stale scene as though it were current, which is worse than showing nothing.
  if (url.pathname.startsWith("/api/")) return;

  e.respondWith((async () => {
    try {
      const res = await fetch(req);
      if (res && res.ok) {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
      }
      return res;
    } catch (err) {
      // `ignoreSearch` because the access key lives in the query string: the cached copy
      // was stored under the key that was current then, and it is still the right shell.
      const hit = await caches.match(req, { ignoreSearch: true });
      if (hit) return hit;
      const shell = await caches.match("/", { ignoreSearch: true });
      if (shell && req.mode === "navigate") return shell;
      return new Response("Cannot reach the laptop running the monitor.",
                          { status: 503, headers: { "Content-Type": "text/plain" } });
    }
  })());
});

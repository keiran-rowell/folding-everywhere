/*! coi-serviceworker v0.1.7 - Guido Zuidhof, licensed under MIT */
let coepCredentialless = false;
if (typeof window === 'undefined') {
    self.addEventListener('install', () => self.skipWaiting());
    self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

    self.addEventListener('fetch', (e) => {
        const r = e.request;
        if (r.cache === 'only-if-cached' && r.mode !== 'same-origin') return;

        const request = (coepCredentialless && r.mode === 'no-cors')
            ? new Request(r, { credentials: 'omit' })
            : r;

        e.respondWith(
            fetch(request).then((res) => {
                if (res.status === 0) return res;

                const headers = new Headers(res.headers);
                headers.set('Cross-Origin-Embedder-Policy', coepCredentialless ? 'credentialless' : 'require-corp');
                headers.set('Cross-Origin-Opener-Policy', 'same-origin');

                return new Response(res.body, {
                    status: res.status,
                    statusText: res.statusText,
                    headers
                });
            }).catch((e) => console.error(e))
        );
    });
} else {
    (() => {
        const reloadedBySelf = window.sessionStorage.getItem('coiReloadedBySelf');
        window.sessionStorage.removeItem('coiReloadedBySelf');

        const coi = {
            shouldRegister: () => true,
            shouldDeregister: () => false,
            coepCredentialless: () => false,
            doReload: () => window.location.reload(),
            quiet: false,
            ...window.coi
        };

        if (coi.shouldDeregister()) {
            navigator.serviceWorker && navigator.serviceWorker.getRegistrations().then((registrations) => {
                for (let registration of registrations) {
                    registration.unregister();
                }
            });

            return;
        }

        if (window.crossOriginIsolated) return;

        if (navigator.serviceWorker) {
            navigator.serviceWorker.register(window.document.currentScript.src).then((registration) => {
                !coi.quiet && console.log('COOP/COEP Service Worker registered', registration.scope);

                registration.addEventListener('updatefound', () => {
                    !coi.quiet && console.log('Reloading page to activate COOP/COEP Service Worker.');
                    window.sessionStorage.setItem('coiReloadedBySelf', 'true');
                    coi.doReload();
                });

                if (registration.active && !navigator.serviceWorker.controller) {
                    !coi.quiet && console.log('Reloading page to activate COOP/COEP Service Worker.');
                    window.sessionStorage.setItem('coiReloadedBySelf', 'true');
                    coi.doReload();
                }
            }, (err) => {
                !coi.quiet && console.error('COOP/COEP Service Worker failed to register:', err);
            });
        }
    })();
}

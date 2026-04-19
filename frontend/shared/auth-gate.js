// ==========================================
// Shared auth gate — loaded by every page before app code.
//
// Pages set `window.__chaliePage` ∈ {'chat','brain','onboarding','login'}
// then load this script. It exposes `window.chalieGateReady`, a Promise
// resolving to `{ stay, status, providersOnly }`:
//   • stay=false  → a redirect has been issued; app code must return.
//   • providersOnly=true (brain only) → render providers tab only.
//
// Rules (per spec):
//   chat       : !account→/on-boarding/  ; !session→/login/  ; !providers→/brain/
//   brain      : !account→/on-boarding/  ; !session→/login/  ; !providers→providersOnly
//   onboarding : account  →/login/
//   login      : session  →/
// ==========================================
(() => {
    const page = window.__chaliePage;

    async function run() {
        let status;
        try {
            const r = await fetch('/auth/status', { credentials: 'same-origin' });
            status = r.ok
                ? await r.json()
                : { has_master_account: false, has_session: false, has_providers: false };
        } catch {
            return { stay: true, status: null, providersOnly: false };
        }
        const { has_master_account, has_session, has_providers } = status;

        if (page === 'onboarding') {
            if (has_master_account) {
                window.location.replace('/login/');
                return { stay: false, status, providersOnly: false };
            }
            return { stay: true, status, providersOnly: false };
        }

        if (page === 'login') {
            if (has_session) {
                window.location.replace('/');
                return { stay: false, status, providersOnly: false };
            }
            return { stay: true, status, providersOnly: false };
        }

        if (!has_master_account) {
            window.location.replace('/on-boarding/');
            return { stay: false, status, providersOnly: false };
        }
        if (!has_session) {
            const next = window.location.pathname + window.location.search;
            window.location.replace('/login/?next=' + encodeURIComponent(next));
            return { stay: false, status, providersOnly: false };
        }

        if (page === 'chat') {
            if (!has_providers) {
                window.location.replace('/brain/');
                return { stay: false, status, providersOnly: false };
            }
            return { stay: true, status, providersOnly: false };
        }

        if (page === 'brain') {
            return { stay: true, status, providersOnly: !has_providers };
        }

        return { stay: true, status, providersOnly: false };
    }

    if (!page) {
        console.warn('[auth-gate] window.__chaliePage not set; gate skipped');
        window.chalieGateReady = Promise.resolve({ stay: true, status: null, providersOnly: false });
        return;
    }
    window.chalieGateReady = run();
})();

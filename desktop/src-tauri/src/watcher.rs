//! Staying signed in. A Chalie server keeps its vault key in memory only, so every restart
//! leaves the vault locked until somebody logs in — and a locked vault means the window is
//! signed out even though the cookie it is holding is still perfectly valid for weeks. The
//! server's own pages notice within five minutes and send the browser to a sign-in form;
//! nobody should have to fill that in when the login is already in the app's settings. So
//! one background loop watches the session, and signs in again the moment it is gone.
//!
//! One loop, installed once from the app's setup. It does nothing until a connection has
//! been made ([`arm`]), and stops again ([`disarm`]) the moment the stored login is
//! refused, wherever that refusal is found — the loop's own attempt, or a relaunch's —
//! because that is the only case a person has to answer.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tauri::{AppHandle, Manager, Url, WebviewWindow};
use tokio::time::sleep;

use crate::config::{self, ServerAddress, ServerConfig};
use crate::error::AppError;
use crate::install::{self, InstallState, Plan};
use crate::server::{self, SessionCookie, SESSION_COOKIE_NAME};
use crate::session;

/// How often the server is asked whether the session the window holds is still a session.
const CHECK_INTERVAL: Duration = Duration::from_secs(15);

/// How long the server is left alone after it refused a login for coming too fast without
/// saying for how long. Its limiter counts the attempts it let through — including the ones
/// the password then failed — and adds nothing for the ones it refuses, so the window drains
/// on its own; but nothing in the refusal says how much of it is left, and a short guess
/// would only spend the attempts somebody needs to type their own password.
const RATE_LIMIT_BACKOFF: Duration = Duration::from_secs(60);

/// The window declared in `tauri.conf.json` — the one the session is handed to.
pub(crate) const MAIN_WINDOW: &str = "main";

/// Where the server's own frontend sends a page whose session has gone.
const LOGIN_PATH: &str = "/login";

/// The switch and the doorbell. Held as managed state so the connect flow can reach it
/// without knowing anything else about the loop.
#[derive(Default)]
struct SessionWatcher {
    armed: AtomicBool,
    /// The app's own page, as the window was showing it at launch. Where the window is sent
    /// when the stored login stops working: loading it runs the wizard, which tries the
    /// saved login once more and puts the credentials form up with the user name filled in.
    wizard_page: Option<Url>,
}

/// Install the watcher: one piece of managed state and one loop, and never a second of
/// either. Call this once, from the app's setup, before anything can connect.
///
/// Setup is also the only moment the window is still on the page the app bundles, and that
/// page is what [`page_at_launch`] captures here to send somebody back to. Installing this any
/// later would capture a server page instead, and every hand-back would then be a navigation
/// to where the window already is — a no-op with nobody to notice it.
pub(crate) fn install(app: &AppHandle) {
    let watcher = Arc::new(SessionWatcher {
        wizard_page: page_at_launch(app),
        ..SessionWatcher::default()
    });
    if !app.manage(Arc::clone(&watcher)) {
        log::warn!("the session watcher is already installed, so a second one is not started");
        return;
    }
    tauri::async_runtime::spawn(
        Watch {
            app: app.clone(),
            watcher,
            answering: true,
            failing_login: false,
            hold_off_until: None,
            plan: Plan::for_this_mac(),
        }
        .run(),
    );
}

/// Start watching. Called after every connection that put the product UI on screen — the
/// first one and every silent relaunch — so the watcher is on exactly when there is a
/// session to keep. Arming again while already armed is the ordinary case and says nothing.
pub(crate) fn arm(app: &AppHandle, address: &ServerAddress) {
    let Some(watcher) = watcher_of(app) else {
        return;
    };
    if !watcher.armed.swap(true, Ordering::SeqCst) {
        log::info!("watching the session for {address}");
    }
}

/// Stop watching. The counterpart to [`arm`], and the one place the switch goes off, so
/// every caller that decides the watching is over leaves the same line behind. Disarming
/// something already off is the ordinary case and says nothing.
pub(crate) fn disarm(app: &AppHandle) {
    let Some(watcher) = watcher_of(app) else {
        return;
    };
    if watcher.armed.swap(false, Ordering::SeqCst) {
        log::info!("no longer watching the session");
    }
}

/// Everything the loop carries from one tick to the next.
struct Watch {
    app: AppHandle,
    watcher: Arc<SessionWatcher>,
    /// Whether the last check reached the server. Only the changes are worth a log line: a
    /// server that is off for the night would otherwise write one every fifteen seconds.
    answering: bool,
    /// Whether the last login attempt failed with something nobody here can act on. Deduped
    /// the same way, and for the same reason: a server that answers its status endpoint and
    /// then fails every login with a 500 is a state that lasts, not an event that happened.
    failing_login: bool,
    /// Set only by a refusal to accept more logins, and the one thing that holds the next
    /// attempt back.
    hold_off_until: Option<Instant>,
    /// This Mac's own Chalie as the installer leaves it, read once: what the watcher starts
    /// when that is what stopped answering. None when there is no HOME to find it in.
    plan: Option<Plan>,
}

impl Watch {
    async fn run(mut self) {
        log::info!(
            "the session watcher checks every {}s",
            CHECK_INTERVAL.as_secs()
        );
        loop {
            sleep(CHECK_INTERVAL).await;
            if !self.watcher.armed.load(Ordering::SeqCst) {
                continue;
            }
            self.check().await;
        }
    }

    /// One tick. `has_session` is the only honest answer to "is this still a session?" —
    /// every other endpoint keeps answering 200 to a valid cookie over a locked vault — and
    /// asking for it also proves the server is up, which is what makes it safe to try a
    /// login afterwards.
    async fn check(&mut self) {
        let Some(config) = saved_config(&self.app) else {
            return;
        };
        let Some(window) = self.app.get_webview_window(MAIN_WINDOW) else {
            log::warn!("the session watcher has no {MAIN_WINDOW} window to keep signed in");
            return;
        };
        // No cookie at all is the same finding as a cookie the server no longer honours:
        // the request goes out without one and the server says there is no session.
        let held = held_session(&window, &config.address);
        match server::status_with_session(&config.address, held.as_ref()).await {
            Err(problem) => {
                if self.answering {
                    self.answering = false;
                    log::info!(
                        "{} is not answering, waiting for it to come back: {problem}",
                        config.address
                    );
                }
                self.maybe_start_native(&window, &config, &problem).await;
            }
            Ok(status) => {
                if !self.answering {
                    self.answering = true;
                    log::info!("{} is answering again", config.address);
                }
                if !status.has_session {
                    self.recover(&window, &config).await;
                }
            }
        }
    }

    /// This Mac's own Chalie, not answering: started. Nobody is shown this start — its steps
    /// go to the log and a failure is a warning there — and nothing here retries it in a
    /// tight loop: the start is awaited inside this tick, and it does not answer until the
    /// server does or until its own limit runs out.
    async fn maybe_start_native(
        &mut self,
        window: &WebviewWindow,
        config: &ServerConfig,
        problem: &AppError,
    ) {
        let Some(plan) = &self.plan else {
            return;
        };
        if !plan.is_stopped_local(config, problem) {
            return;
        }
        let state = self.app.state::<InstallState>();
        log::info!(
            "{} is not answering, and this Mac's own Chalie is installed; starting it",
            config.address
        );
        match install::start_installed_chalie(&self.app, plan, &state, false).await {
            Ok(()) => {
                log::info!("{} was started; signing in", config.address);
                self.recover(window, config).await;
            }
            Err(start_failure) => {
                log::warn!("{} could not be started: {start_failure}", config.address);
            }
        }
    }

    /// The session is gone and the server is up: sign in again with the saved login, hand
    /// the window the new session and put the product UI back. One attempt per loss — a failure
    /// that leaves the session gone is found again by the next tick, fifteen seconds later,
    /// which is well inside what the server's own limiter allows.
    async fn recover(&mut self, window: &WebviewWindow, config: &ServerConfig) {
        if let Some(until) = self.hold_off_until {
            if Instant::now() < until {
                return;
            }
            self.hold_off_until = None;
        }
        let address = &config.address;
        log::info!("the session for {address} is gone, signing in again");
        // Read fresh rather than carried down from the tick's own read: the settings file is
        // what a person edits or deletes to stop this, and between the two is exactly when
        // they would have done it.
        let credentials = match session::stored_login(&self.app) {
            Ok(Some(credentials)) => credentials,
            Ok(None) => {
                self.hand_back_to_the_wizard(
                    window,
                    &format!("there is no stored login for {address} any more"),
                );
                return;
            }
            Err(problem) => {
                log::warn!("the stored login for {address} could not be read: {problem}");
                return;
            }
        };
        let outcome = server::login(address, &credentials).await;
        // Every outcome but the last one either ends the watching, holds it off, or works,
        // so reaching one of them means the next unexplained failure is news again.
        let was_failing = self.failing_login;
        self.failing_login = false;
        match outcome {
            Ok(session) => restore(window, address, &session),
            Err(AppError::InvalidCredentials) => self.hand_back_to_the_wizard(
                window,
                &format!("the stored login for {address} was rejected"),
            ),
            Err(AppError::RateLimited {
                retry_after_seconds,
            }) => {
                let wait = retry_after_seconds.map_or(RATE_LIMIT_BACKOFF, Duration::from_secs);
                log::info!(
                    "{address} is refusing logins for now, waiting {}s before trying again",
                    wait.as_secs()
                );
                self.hold_off_until = Some(Instant::now() + wait);
            }
            Err(problem) => {
                self.failing_login = true;
                if !was_failing {
                    log::warn!("signing in again to {address} failed: {problem}");
                }
            }
        }
    }

    /// The one case a person has to answer: the stored login no longer opens this server.
    /// The watcher stops — retrying a password the server refuses only spends the login
    /// attempts somebody needs to type the right one — and the window goes back to the
    /// app's own page, where the wizard asks for it.
    fn hand_back_to_the_wizard(&self, window: &WebviewWindow, why: &str) {
        disarm(&self.app);
        log::info!("{why}, so the wizard will ask for it again");
        let Some(page) = &self.watcher.wizard_page else {
            log::warn!("the app's own page is not known, so the wizard cannot be reopened");
            return;
        };
        if let Err(problem) = window.navigate(page.clone()) {
            log::warn!("the wizard could not be reopened: {problem}");
        }
    }
}

/// Hand the window the new session and put the product UI back on it. The handoff is the
/// app's own — the same `set_cookie` and read-back proof the first connect runs — so a cookie
/// the webview would keep but never send is refused here exactly as it is there.
///
/// Reloading is what keeps somebody where they were, and it is also what clears the page's
/// own once-per-load "the session died" latch. It is only right when the window is on a page
/// of the product UI: the sign-in page would just be served again, and the app's own page is
/// not the product UI at all — both of those go to the server's root instead.
fn restore(window: &WebviewWindow, address: &ServerAddress, session: &SessionCookie) {
    if let Err(problem) = session::hand_over_session(window, address, session) {
        log::warn!("the new session for {address} could not be handed to the window: {problem}");
        return;
    }
    let showing = window.url().ok();
    let back = match showing {
        Some(url) if is_served_by(&url, address) && !is_login_page(&url) => window.reload(),
        _ => match address.base_url() {
            Ok(root) => window.navigate(root),
            Err(problem) => {
                log::warn!("the window could not be put back on {address}: {problem}");
                return;
            }
        },
    };
    match back {
        Ok(()) => log::info!("session restored for {address}"),
        Err(problem) => log::warn!("the window could not be put back on {address}: {problem}"),
    }
}

/// The session the webview is holding for this server, if it is holding one. Wrapped on the
/// way out so the value cannot be printed by accident on the way through.
fn held_session(window: &WebviewWindow, address: &ServerAddress) -> Option<SessionCookie> {
    let url = match address.base_url() {
        Ok(url) => url,
        Err(problem) => {
            log::warn!("the saved address cannot be turned into a URL: {problem}");
            return None;
        }
    };
    let cookies = match window.cookies_for_url(url) {
        Ok(cookies) => cookies,
        Err(problem) => {
            log::warn!("the window's cookies for {address} could not be read: {problem}");
            return None;
        }
    };
    cookies
        .into_iter()
        .find(|cookie| cookie.name() == SESSION_COOKIE_NAME)
        .map(SessionCookie::from_cookie)
}

/// The page the window is on before anything has navigated it, which is the app's own
/// bundled page. Read at install time because that is the only moment it is guaranteed to
/// be showing, and it is where a refused login has to send somebody back to.
fn page_at_launch(app: &AppHandle) -> Option<Url> {
    let Some(window) = app.get_webview_window(MAIN_WINDOW) else {
        log::warn!("there is no {MAIN_WINDOW} window, so the session watcher has none to keep");
        return None;
    };
    match window.url() {
        Ok(url) => Some(url),
        Err(problem) => {
            log::warn!("the app's own page could not be read at startup: {problem}");
            None
        }
    }
}

/// The wizard's page exactly as the window opened on it: the one page of the app's own that
/// a window may be sent to.
pub(crate) fn wizard_page(app: &AppHandle) -> Option<Url> {
    watcher_of(app).and_then(|watcher| watcher.wizard_page.clone())
}

fn watcher_of(app: &AppHandle) -> Option<Arc<SessionWatcher>> {
    app.try_state::<Arc<SessionWatcher>>()
        .map(|state| Arc::clone(state.inner()))
}

/// The saved configuration, or nothing to act on. A file that cannot be read is a problem
/// for the wizard to raise, not for a background loop to keep raising every tick — so it is
/// said once per tick at warning level and the tick ends there.
fn saved_config(app: &AppHandle) -> Option<ServerConfig> {
    match config::read(app) {
        Ok(config) => config,
        Err(problem) => {
            log::warn!("the session watcher cannot read the saved configuration: {problem}");
            None
        }
    }
}

/// Is this page the configured server's own? Scheme, host and port are the whole of it —
/// whatever path the product UI settles on is its own business.
pub(crate) fn is_served_by(url: &Url, address: &ServerAddress) -> bool {
    url.scheme() == "http"
        && url.host_str() == Some(address.host())
        && url.port_or_known_default() == Some(address.port())
}

fn is_login_page(url: &Url) -> bool {
    url.path().starts_with(LOGIN_PATH)
}

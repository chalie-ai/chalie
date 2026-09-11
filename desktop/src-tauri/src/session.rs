//! The one path from a username and password to the product UI on screen. Creating the
//! master account, the first-run connect and every silent relaunch login all end in the same
//! handoff, so there is a single place where the session cookie is handed to the webview,
//! written down and followed by a navigation.

use tauri::{AppHandle, Manager, WebviewWindow};

use crate::config::{self, Credentials, Mode, ServerAddress, ServerConfig};
use crate::error::{AppError, AppResult};
use crate::server::{self, SessionCookie, SESSION_COOKIE_NAME};
use crate::watcher;

/// Log in and show the product UI.
pub(crate) async fn connect_and_show(
    window: &WebviewWindow,
    address: &ServerAddress,
    mode: Mode,
    credentials: &Credentials,
) -> AppResult<()> {
    let session = server::login(address, credentials).await?;
    show(window, address, mode, credentials, &session)
}

/// Everything after the session exists: remember the setup, hand the cookie to the webview,
/// show the UI the server serves and start watching that session. Logging in and creating
/// the master account both end here — creating one answers with a session for the account it
/// just made — so there is one handoff to get right rather than one per way of arriving.
/// Re-writing an unchanged setup is a harmless write that also heals a first run that died
/// between the login and the disk, so relaunch takes exactly the same steps as the first
/// connect instead of a second, thinner path.
///
/// Nothing is written down until the whole handoff is proven: the webview takes the cookie
/// and gives it back. Only then is the setup saved — server and login in the one file, in
/// the one write, so the login on disk is never a login for some other server, and the
/// server this app used to point at is simply gone, overwritten rather than cleaned up
/// after.
pub(crate) fn show(
    window: &WebviewWindow,
    address: &ServerAddress,
    mode: Mode,
    credentials: &Credentials,
    session: &SessionCookie,
) -> AppResult<()> {
    hand_over_session(window, address, session)?;
    config::write(
        window.app_handle(),
        &ServerConfig {
            address: address.clone(),
            mode,
            credentials: credentials.clone(),
        },
    )?;
    navigate_to_server(window, address)?;
    // The last step, and the only thing this flow tells the watcher: there is a session on
    // screen now, so keep it. Arming here rather than in each caller is what makes the
    // first connect and every silent relaunch the same thing to the watcher.
    watcher::arm(window.app_handle(), address);
    Ok(())
}

/// The login the saved setup holds, or [`None`] when there is no saved setup to hold one.
/// A file that is there but cannot be read is an error, exactly as it is wherever else the
/// configuration is read: a setup nothing can make sense of is not an absent setup.
pub(crate) fn stored_login(app: &AppHandle) -> AppResult<Option<Credentials>> {
    Ok(config::read(app)?.map(|config| config.credentials))
}

/// Hand the webview the session, and prove it took it. Shared with the watcher, so a
/// session minted hours later lands on exactly the same terms as the first one.
pub(crate) fn hand_over_session(
    window: &WebviewWindow,
    address: &ServerAddress,
    session: &SessionCookie,
) -> AppResult<()> {
    // The window is pointed at http, and a Secure cookie is the one attribute that would be
    // stored and read back here perfectly well and then never sent: every request the
    // product UI made would go out unauthenticated, with nothing anywhere saying why. Said
    // as a sentence here instead.
    if session.cookie().secure() == Some(true) {
        return Err(AppError::Webview {
            message: format!(
                "the server marked the {SESSION_COOKIE_NAME} cookie Secure, so the window \
                 would keep it but never send it back over http and the Chalie interface \
                 would load signed out"
            ),
        });
    }
    // The server's own cookie, not a rebuilt one: Path, HttpOnly and SameSite stay exactly
    // as it set them, so the webview sends back what a browser would have sent.
    let mut cookie = session.cookie().clone();
    // A Set-Cookie header names no domain — the browser infers it from the host it asked.
    // The webview is being handed this cookie out of band, so the host is spelled out here.
    // It is the canonical host, which is the spelling WebKit stores and reports back.
    cookie.set_domain(address.host().to_string());
    // Deliberately without an expiry: the cookie lives only as long as this app run, and
    // every launch logs in again from the saved login, so no session outlives the process.
    cookie.unset_expires();
    cookie.set_max_age(None);
    window.set_cookie(cookie).map_err(|e| AppError::Webview {
        message: format!("the cookie was refused: {e}"),
    })?;
    confirm_cookie_stored(window, address.host())?;
    log::info!("cookie {SESSION_COOKIE_NAME} set for {}", address.host());
    Ok(())
}

fn navigate_to_server(window: &WebviewWindow, address: &ServerAddress) -> AppResult<()> {
    let url = address.base_url()?;
    log::info!("navigating main window to {url}");
    window.navigate(url).map_err(|e| AppError::Webview {
        message: format!("the navigation was refused: {e}"),
    })
}

/// Setting a cookie only queues the write on the webview thread; reading the store back is
/// what waits for it. So this both proves the cookie landed and orders it before the
/// navigation that has to carry it.
fn confirm_cookie_stored(window: &WebviewWindow, host: &str) -> AppResult<()> {
    let stored = window.cookies().map_err(|e| AppError::Webview {
        message: format!("the cookie store could not be read back: {e}"),
    })?;
    if stored
        .iter()
        .any(|cookie| cookie.name() == SESSION_COOKIE_NAME && cookie.domain() == Some(host))
    {
        return Ok(());
    }
    Err(AppError::Webview {
        message: format!("the webview did not keep the {SESSION_COOKIE_NAME} cookie for {host}"),
    })
}

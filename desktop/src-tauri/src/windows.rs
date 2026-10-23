//! The app's window, and where its pages may go. The saved server's pages, the wizard's own
//! page and a blank page stay in the app; web, mail and phone links to anywhere else are
//! handed to the Mac, which opens them in the apps it uses for them; everything else is
//! refused. The app has one window: a server page that asks for a window of its own, as
//! Brain's button does, is shown in it instead. The same rules cover its downloads, which
//! never pass the navigation check.

use std::process::Command;
use std::thread;

use tauri::menu::{Menu, MenuEvent, MenuItem};
use tauri::webview::{DownloadEvent, NewWindowResponse};
use tauri::{AppHandle, Manager, Url, WebviewWindow, WebviewWindowBuilder, Wry};

use crate::config::{self, ServerAddress};
use crate::watcher::{self, MAIN_WINDOW};

/// The menu item that reloads the window in front.
const RELOAD: &str = "reload";

/// Where a page belongs.
#[derive(Debug)]
enum Verdict {
    /// A page of the saved server.
    Server,
    /// One of the app's own: the wizard's page, or a blank page.
    App,
    /// A web, mail or phone link to anywhere else, for the Mac to open.
    External,
    /// Everything else, the app's own pages other than the wizard's among them.
    Refused,
}

/// Where `url` belongs. `server` is the saved address, `wizard` the wizard's page exactly as
/// the main window opened on it, and `dev` the development server, which only a development
/// build passes. The wizard's page is matched whole rather than by scheme, so a server page
/// can send a window back to the wizard and to nothing else the app bundles.
fn classify(
    url: &Url,
    server: Option<&ServerAddress>,
    wizard: Option<&Url>,
    dev: Option<&Url>,
) -> Verdict {
    if server.is_some_and(|address| watcher::is_served_by(url, address)) {
        return Verdict::Server;
    }
    if url.as_str() == "about:blank"
        || wizard == Some(url)
        || dev.is_some_and(|dev| dev.origin() == url.origin())
    {
        return Verdict::App;
    }
    match url.scheme() {
        "http" | "https" | "mailto" | "tel" => Verdict::External,
        _ => Verdict::Refused,
    }
}

/// [`classify`] against the app as it is now. The settings file is read on every call, so a
/// server connected a moment ago is let in and a forgotten one is not.
fn verdict(app: &AppHandle, url: &Url) -> Verdict {
    let dev = if cfg!(dev) {
        app.config().build.dev_url.as_ref()
    } else {
        None
    };
    let wizard = watcher::wizard_page(app);
    classify(url, saved_address(app).as_ref(), wizard.as_ref(), dev)
}

/// The saved server, if there is one and the settings file can be read. Without one, no server
/// page is let in: with no settings file, or one that cannot be read, the server's own pages
/// are links to anywhere else, so they open in the Mac's browser and the window stays where it
/// is — a Command-R reload of one of them included.
fn saved_address(app: &AppHandle) -> Option<ServerAddress> {
    match config::read(app) {
        Ok(config) => config.map(|config| config.address),
        Err(problem) => {
            log::warn!(
                "the saved configuration cannot be read, so no server page is let in: {problem}"
            );
            None
        }
    }
}

/// Whether a download may be saved: a file from the saved server, or one a server page made
/// itself — a `blob:` URL, which carries the origin of the page that made it.
fn download_allowed(url: &Url, server: Option<&ServerAddress>) -> bool {
    let Some(address) = server else {
        return false;
    };
    if url.scheme() == "blob" {
        return Url::parse(url.path()).is_ok_and(|inner| watcher::is_served_by(&inner, address));
    }
    watcher::is_served_by(url, address)
}

/// Hand a link to the Mac. `open` returns as soon as it has passed the link on; it is waited
/// for on a thread of its own so its exit is collected without holding the window up.
fn open_externally(url: &Url) {
    let url = url.to_string();
    thread::spawn(
        move || match Command::new("/usr/bin/open").arg(&url).status() {
            Ok(status) if status.success() => {}
            Ok(status) => log::warn!("the Mac could not open {url}: open exited with {status}"),
            Err(problem) => log::warn!("the Mac could not be asked to open {url}: {problem}"),
        },
    );
}

/// The rules the window carries: where its pages may go, what it may open and what it may
/// save.
fn guarded<'a>(
    app: &AppHandle,
    builder: WebviewWindowBuilder<'a, Wry, AppHandle>,
) -> WebviewWindowBuilder<'a, Wry, AppHandle> {
    let navigating = app.clone();
    let opening = app.clone();
    builder
        .on_navigation(move |url| match verdict(&navigating, url) {
            Verdict::Server | Verdict::App => true,
            Verdict::External => {
                open_externally(url);
                false
            }
            Verdict::Refused => {
                log::warn!("a page tried to go to {url}, which the app does not open");
                false
            }
        })
        .on_new_window(move |url, _features| new_window(&opening, &url))
        .on_download(|webview, event| match event {
            DownloadEvent::Requested { url, .. } => {
                let allowed = download_allowed(&url, saved_address(webview.app_handle()).as_ref());
                if !allowed {
                    log::warn!("a download from {url} was refused: it is not the server's");
                }
                allowed
            }
            DownloadEvent::Finished { url, success, .. } => {
                log::info!("download from {url} finished, success: {success}");
                true
            }
            _ => false,
        })
}

/// A page asking for a window of its own, as Brain's button does. No second window opens: a
/// server page is shown in the app's one window, as a browser with no tabs would show it; a
/// link to anywhere else goes to the Mac; nothing else opens.
fn new_window(app: &AppHandle, url: &Url) -> NewWindowResponse<Wry> {
    match verdict(app, url) {
        Verdict::Server => match app.get_webview_window(MAIN_WINDOW) {
            Some(window) => {
                if let Err(problem) = window.navigate(url.clone()) {
                    log::warn!("{url} could not be shown in the {MAIN_WINDOW} window: {problem}");
                }
            }
            None => log::warn!("there is no {MAIN_WINDOW} window to show {url} in"),
        },
        Verdict::External => open_externally(url),
        Verdict::App | Verdict::Refused => {
            log::warn!("a page asked for a window on {url}, which the app does not open");
        }
    }
    NewWindowResponse::Deny
}

/// Build the main window from its entry in `tauri.conf.json`, which is marked not to be
/// created by Tauri because only a window built in code can carry these rules. Call it from
/// setup, before the watcher is installed: the watcher reads the page it opens on.
pub(crate) fn open_main(app: &AppHandle) -> tauri::Result<WebviewWindow> {
    let config = app
        .config()
        .app
        .windows
        .iter()
        .find(|window| window.label == MAIN_WINDOW)
        .ok_or(tauri::Error::WindowNotFound)?;
    guarded(app, WebviewWindowBuilder::from_config(app, config)?).build()
}

/// Tauri's own menu for the Mac, with Reload added to its View menu.
pub(crate) fn menu(app: &AppHandle) -> tauri::Result<Menu<Wry>> {
    let menu = Menu::default(app)?;
    let reload = MenuItem::with_id(app, RELOAD, "Reload", true, Some("CmdOrCtrl+R"))?;
    for item in menu.items()? {
        if let Some(view) = item.as_submenu() {
            if view.text()? == "View" {
                view.append(&reload)?;
                return Ok(menu);
            }
        }
    }
    log::warn!("the menu has no View menu, so it has no Reload");
    Ok(menu)
}

/// Reload the window in front, whichever one it is.
pub(crate) fn on_menu_event(app: &AppHandle, event: MenuEvent) {
    if event.id() != RELOAD {
        return;
    }
    let Some(window) = app
        .webview_windows()
        .into_values()
        .find(|window| window.is_focused().unwrap_or(false))
    else {
        return;
    };
    if let Err(problem) = window.reload() {
        let label = window.label();
        log::warn!("the {label} window could not be reloaded: {problem}");
    }
}

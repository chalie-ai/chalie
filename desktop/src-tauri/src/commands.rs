//! What the wizard is allowed to ask the Rust side to do.

use serde::Serialize;
use tauri::{AppHandle, Manager, State, WebviewWindow};

use crate::config::{self, Credentials, Mode, ServerAddress};
use crate::error::{AppError, AppResult};
use crate::install::{self, InstallState, Plan};
use crate::server::{self, ServerStatus};
use crate::session;
use crate::watcher;

/// The saved setup, as much of it as the wizard is ever told: which server, and never the
/// login that sits beside it in the same file — the wizard asks for a password, it is not
/// handed one back. The address's fields are flattened in, so their absence is what says
/// this Mac has never been connected — no separate flag to keep in step with them.
#[derive(Debug, Serialize)]
pub(crate) struct SetupState {
    #[serde(flatten)]
    server: Option<ServerAddress>,
}

/// A Chalie found on this Mac. `found` says one answered; `has_master_account` says whether
/// it has anybody to sign in as yet — without one, the wizard makes that account instead of
/// asking for a login.
#[derive(Debug, Serialize)]
pub(crate) struct LocalInstance {
    found: bool,
    has_master_account: bool,
    host: String,
    port: u16,
}

#[derive(Debug, Serialize)]
pub(crate) struct AutoConnectResult {
    needs_credentials: bool,
    username: Option<String>,
}

/// Has this Mac ever been connected to a Chalie?
#[tauri::command]
pub(crate) async fn get_setup_state(app: AppHandle) -> AppResult<SetupState> {
    let server = config::read(&app)?.map(|config| config.address);
    match &server {
        Some(address) => log::info!("setup state: configured for {address}"),
        None => log::info!("setup state: not configured"),
    }
    Ok(SetupState { server })
}

/// Is a Chalie installed and running on this Mac?
#[tauri::command]
pub(crate) async fn detect_local() -> AppResult<LocalInstance> {
    let plan = Plan::public_installer()?;
    let installed = plan.installed();
    let address = plan.address;
    let (found, has_master_account) = if installed {
        local_state(&address).await
    } else {
        (false, false)
    };
    log::info!(
        "local instance: installed={installed} answering={found} \
         has_master_account={has_master_account}"
    );
    Ok(LocalInstance {
        found,
        has_master_account,
        host: address.host().to_string(),
        port: address.port(),
    })
}

/// Is there a Chalie at this address, and does it have an account to sign in to yet?
#[tauri::command]
pub(crate) async fn probe_server(host: String, port: u16) -> AppResult<ServerStatus> {
    let address = ServerAddress::parse(&host, port)?;
    let result = server::probe(&address).await;
    match &result {
        Ok(status) => log::info!(
            "probe {address}: answered, has_master_account={}, vault {:?}",
            status.has_master_account,
            status.vault_state
        ),
        Err(e) => log::info!("probe {address}: {e}"),
    }
    result
}

/// First run: prove the login, remember it, and show the product UI.
#[tauri::command]
pub(crate) async fn connect(
    window: WebviewWindow,
    host: String,
    port: u16,
    username: String,
    password: String,
) -> AppResult<()> {
    let address = ServerAddress::parse(&host, port)?;
    let mode = Mode::for_host(address.host());
    let credentials = Credentials { username, password };
    session::connect_and_show(&window, &address, mode, &credentials).await
}

/// Put a Chalie on this Mac, give it the account that was just filled in, and show the
/// product UI. Returns when the window has been handed over, or with the reason it was not:
/// everything in between is reported as it happens through the install's own events.
#[tauri::command]
pub(crate) async fn install_local(
    window: WebviewWindow,
    state: State<'_, InstallState>,
    username: String,
    password: String,
) -> AppResult<()> {
    let plan = Plan::public_installer()?;
    let credentials = Credentials { username, password };
    // Off for the install: left on, the watcher would act on the Chalie this install is
    // putting in place — start it, or sign the window in to it with the old login — while
    // the installer is still running. The sign-in that ends a successful install turns it
    // back on.
    watcher::disarm(window.app_handle());
    install::install_and_register(&window, &plan, &state, &credentials).await
}

/// Give a Chalie that is already running its master account, and show the product UI. It is
/// where the wizard goes when a probe finds a Chalie with nobody to sign in as.
#[tauri::command]
pub(crate) async fn create_account(
    window: WebviewWindow,
    host: String,
    port: u16,
    username: String,
    password: String,
) -> AppResult<()> {
    let address = ServerAddress::parse(&host, port)?;
    let mode = Mode::for_host(address.host());
    let credentials = Credentials { username, password };
    let cookie = server::register(&address, &credentials).await?;
    session::show(&window, &address, mode, &credentials, &cookie)
}

/// Relaunch: log in again from the saved login without asking. When that login no longer
/// works the wizard takes over with the username already filled in. When nothing answers at
/// all and it is this Mac's own Chalie ([`Plan::is_stopped_local`]), this starts it — or, when
/// a start or an install is already under way, waits for that one — instead of leaving the
/// window on an error a restart caused, then signs in the same way once it answers.
#[tauri::command]
pub(crate) async fn auto_connect(
    window: WebviewWindow,
    state: State<'_, InstallState>,
) -> AppResult<AutoConnectResult> {
    let app = window.app_handle();
    let Some(config) = config::read(app)? else {
        return Err(AppError::Config {
            message: "no server has been configured yet".to_string(),
        });
    };
    log::info!("relaunch: signing in again to {}", config.address);
    // The login is a field of the configuration just read, not a second thing to go and
    // find: a file that parsed at all had both halves in it.
    let credentials = &config.credentials;
    let mut outcome =
        session::connect_and_show(&window, &config.address, config.mode, credentials).await;
    let plan = Plan::for_this_mac();
    let stopped_local = matches!((&outcome, &plan), (Err(problem), Some(plan))
        if plan.is_stopped_local(&config, problem));
    if let (true, Some(plan)) = (stopped_local, &plan) {
        // Somebody is waiting on this start, so the watcher is off while it runs: left on, it
        // would sign the window in behind their back while they are watching it happen.
        watcher::disarm(app);
        log::info!(
            "{} is not answering, and this Mac's own Chalie is installed; starting it",
            config.address
        );
        outcome = match install::start_or_wait(app, plan, &state).await {
            Ok(()) => {
                session::connect_and_show(&window, &config.address, config.mode, credentials).await
            }
            Err(problem) => Err(problem),
        };
    }
    match outcome {
        Ok(()) => Ok(AutoConnectResult {
            needs_credentials: false,
            username: None,
        }),
        Err(AppError::InvalidCredentials) => {
            // A launch that found nothing answering leaves the watcher on the arm below,
            // so by the time somebody clicks "try again" against a server that is back up
            // it can still be watching. This answer puts the credentials form on screen —
            // and a watcher left on would, within one interval, spend a login on the very
            // password the server just refused and then send the window back to this page,
            // reloading the form out from under whoever is typing into it. A refused
            // password is the one outcome that needs a person, so the watching stops here
            // and nothing touches the window while they answer.
            watcher::disarm(app);
            log::info!("the stored login for {} was rejected", config.address);
            Ok(AutoConnectResult {
                needs_credentials: true,
                username: Some(credentials.username.clone()),
            })
        }
        Err(other) => {
            // The server is not there yet, or what answered was not a Chalie, or this Mac's
            // own Chalie was started and has not answered. The wizard shows that, but nothing
            // in it ever tries again on its own, so without this the window sits on an error
            // screen for good — and the commonest way to reach it is the Mac restarting,
            // where the app simply won the race against a server that is seconds from
            // answering. So the watcher is left watching the saved address: it spends nothing
            // while nothing answers, and the moment the server is back it signs in and puts
            // the product UI up with nobody having to click anything.
            //
            // Not on the refused-password arm above — that one switches the watching off
            // instead: it needs a person to type something, and a loop retrying a password
            // the server has already turned down would only spend the attempts they need to
            // type the right one.
            watcher::arm(app, &config.address);
            Err(other)
        }
    }
}

/// Whether a Chalie answered on this Mac, and whether it has a master account yet. A brand
/// new server answers `health` perfectly well while having nobody to sign in as, so the
/// status the probe brings back is what decides: sending that user to a sign-in form would
/// only earn a refusal.
async fn local_state(address: &ServerAddress) -> (bool, bool) {
    match server::probe(address).await {
        Ok(status) => (true, status.has_master_account),
        Err(e) => {
            log::info!("no Chalie answering on this Mac: {e}");
            (false, false)
        }
    }
}

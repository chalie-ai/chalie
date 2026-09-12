//! Putting a Chalie on this Mac: run the published installer, start what it leaves behind,
//! wait until the server answers, then give it the account that was filled in first and hand
//! this window the session. Every line those commands print goes two places at once — into
//! the window, where somebody is watching it happen, and into the app's own log file, which
//! is where a failure sends them afterwards.
//!
//! Both commands are run by the person's own login shell, as an interactive login shell, for
//! the same reason the installer's instructions say to paste it into a terminal: it needs a
//! `python3` of 3.11 or newer, and the one somebody installed themselves is only on the PATH
//! their own profile sets. An app opened from the Finder is given the system's directories
//! and nothing else.

use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, WebviewWindow};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::time::sleep;

use crate::config::{Credentials, Mode, ServerAddress, ServerConfig, DEFAULT_PORT, LOCAL_HOST};
use crate::error::{AppError, AppResult};
use crate::server;
use crate::session;

/// The published one-liner, exactly as it is published.
const PUBLIC_INSTALLER: &str = "curl -fsSL https://chalie.ai/install | bash";

/// The shell used when `$SHELL` does not name one by its full path — any other name would be
/// looked up on the very PATH being asked for. It is the one macOS gives every new account.
const DEFAULT_LOGIN_SHELL: &str = "/bin/zsh";

/// What the installer leaves in the home directory: the app it unpacks, the command that
/// starts a Chalie, and the file that command sends everything the server prints to.
const CHALIE_APP: &str = ".chalie/app";
const CHALIE_COMMAND: &str = ".local/bin/chalie";
const CHALIE_LOG: &str = ".chalie/chalie.log";

/// How long the server is given to come up before somebody is told it did not. Downloading
/// and unpacking a first install on a slow line is the long pole, and a bound that is too
/// short turns a working install into a failure nobody can explain.
const READY_WITHIN: Duration = Duration::from_secs(8 * 60);
/// How often it is asked.
const READY_POLL: Duration = Duration::from_secs(3);

/// What the wizard listens for: one event per line printed, one per part of the install
/// reached.
const OUTPUT_EVENT: &str = "install-output";
const PHASE_EVENT: &str = "install-phase";

/// What installing and starting a Chalie on this Mac depends on: the home it installs into,
/// the shell that runs the commands, where the server will answer and where it keeps its own
/// log.
#[derive(Debug, Clone)]
pub(crate) struct Plan {
    home: PathBuf,
    login_shell: PathBuf,
    pub(crate) address: ServerAddress,
    chalie_log: PathBuf,
}

impl Plan {
    /// The home directory comes from the environment rather than from Tauri's own lookup
    /// because it is also what the installer itself reads: if the two disagreed, this app
    /// would be waiting on a Chalie installed somewhere else. The shell does too, because
    /// `$SHELL` is the one this person logs in with.
    pub(crate) fn public_installer() -> AppResult<Self> {
        let Some(home) = std::env::var_os("HOME") else {
            return Err(AppError::Config {
                message: "there is no HOME in this app's environment, so the installer cannot \
                          be told where to install"
                    .to_string(),
            });
        };
        let home = PathBuf::from(home);
        Ok(Self {
            login_shell: std::env::var_os("SHELL")
                .map(PathBuf::from)
                .filter(|shell| shell.is_absolute())
                .unwrap_or_else(|| PathBuf::from(DEFAULT_LOGIN_SHELL)),
            address: ServerAddress::parse(LOCAL_HOST, DEFAULT_PORT)?,
            chalie_log: home.join(CHALIE_LOG),
            home,
        })
    }

    /// The same plan, for code that only needs it to recognise this Mac's own Chalie. Without
    /// a HOME there is no install to find: that is logged and nothing is started, but nothing
    /// else is held up — a connect to any other server goes ahead.
    pub(crate) fn for_this_mac() -> Option<Self> {
        Self::public_installer()
            .inspect_err(|problem| log::warn!("this Mac's own Chalie cannot be started: {problem}"))
            .ok()
    }

    /// Whether the installer has run for this home: the app it unpacks and the command that
    /// starts it are both there.
    pub(crate) fn installed(&self) -> bool {
        self.home.join(CHALIE_APP).is_dir() && self.home.join(CHALIE_COMMAND).exists()
    }

    /// Is this failure nothing worse than this Mac's own Chalie not running? Saved as Local,
    /// on the port this plan's Chalie serves, installed here, and nothing answering at all —
    /// not something else answering, not a login refused. `Mode` comes from the host alone,
    /// so `localhost` on a port a native install does not serve is Local too; the port is
    /// what keeps this to the one Chalie this app can start.
    pub(crate) fn is_stopped_local(&self, config: &ServerConfig, problem: &AppError) -> bool {
        matches!(problem, AppError::Unreachable { .. })
            && config.mode == Mode::Local
            && config.address.port() == self.address.port()
            && self.installed()
    }
}

/// Which part of the install is happening. The wizard turns each into a sentence; the last
/// two are reached after the commands are done, when the account is being made and the
/// window is being handed the session.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Installing,
    Starting,
    Waiting,
    CreatingAccount,
    Connecting,
}

/// The one install or start there may be at a time. Two at once would be two copies of the
/// installer writing the same files.
#[derive(Default)]
pub(crate) struct InstallState {
    running: Mutex<bool>,
}

impl InstallState {
    /// Take the slot, or refuse because it is taken. The slot goes back when the guard is
    /// dropped, however the install or start ended.
    fn claim(&self) -> AppResult<Slot<'_>> {
        let mut running = self.running();
        if *running {
            return Err(AppError::InstallRunning);
        }
        *running = true;
        Ok(Slot(self))
    }

    /// A lock that cannot fail. Poisoning means some other thread panicked while holding it,
    /// and what it holds is one bool — which a panic cannot leave half-written — so the state
    /// is taken as it stands rather than panicking again.
    fn running(&self) -> MutexGuard<'_, bool> {
        self.running
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// The slot, taken.
struct Slot<'a>(&'a InstallState);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        *self.0.running() = false;
    }
}

/// The whole of it: put a Chalie on this Mac, give it the account that was filled in before
/// any of this started, and hand this window the session. Returns when the window has been
/// handed over, or with the reason it was not.
pub(crate) async fn install_and_register(
    window: &WebviewWindow,
    plan: &Plan,
    state: &InstallState,
    credentials: &Credentials,
) -> AppResult<()> {
    let app = window.app_handle();
    let _slot = state.claim()?;
    announce(app, Phase::Installing);
    let ended = run(plan, PUBLIC_INSTALLER, true, |line| report(app, line)).await?;
    if !ended.success() {
        return Err(AppError::InstallFailed {
            message: format!(
                "the installer ended with {ended}. Everything it printed is in {}.",
                app_log_path(app)
            ),
        });
    }
    start_then_wait(plan, |phase| announce(app, phase), |line| report(app, line)).await?;
    // A Chalie that has just been installed has nobody to sign in as, so the account is made
    // here rather than logged in to, and making it is what signs this window in.
    announce(app, Phase::CreatingAccount);
    let session = server::register(&plan.address, credentials).await?;
    announce(app, Phase::Connecting);
    session::show(
        window,
        &plan.address,
        Mode::for_host(plan.address.host()),
        credentials,
        &session,
    )
}

/// Start a Chalie this Mac already has — nothing here is being installed — and wait until it
/// answers. Shares `state` with [`install_and_register`], so a real install holding the slot
/// is a reason this refuses to run rather than a second thing writing into `~/.chalie` at
/// once. A start nobody asked for is shown to nobody: its phases and its output go to the
/// log alone.
pub(crate) async fn start_installed_chalie(
    app: &AppHandle,
    plan: &Plan,
    state: &InstallState,
    shown: bool,
) -> AppResult<()> {
    let _slot = state.claim()?;
    let on_phase = |phase| match shown {
        true => announce(app, phase),
        false => log::info!("background start: {phase:?}"),
    };
    let on_output = |line: &str| match shown {
        true => report(app, line),
        false => log::info!(target: "installer", "{line}"),
    };
    start_then_wait(plan, on_phase, on_output).await
}

/// The relaunch's start: this Mac's own Chalie started and waited for — or, when a start or
/// an install already holds the slot, only waited for, with the same limit, because that one
/// is bringing up the same server.
pub(crate) async fn start_or_wait(
    app: &AppHandle,
    plan: &Plan,
    state: &InstallState,
) -> AppResult<()> {
    match start_installed_chalie(app, plan, state, true).await {
        Err(AppError::InstallRunning) => {
            log::info!("{} is already being started; waiting for it", plan.address);
            announce(app, Phase::Waiting);
            wait_until_ready(plan, None).await
        }
        outcome => outcome,
    }
}

/// The command that starts Chalie, then the wait for it to answer: the end of every install
/// and the whole of a start.
async fn start_then_wait(
    plan: &Plan,
    on_phase: impl Fn(Phase),
    on_output: impl Fn(&str),
) -> AppResult<()> {
    on_phase(Phase::Starting);
    let start = format!("{}/{CHALIE_COMMAND} start", plan.home.display());
    let ended = run(plan, &start, false, &on_output).await?;
    // That command waits half a minute for the server to write its pidfile and then exits
    // non-zero, which a first start on a slow machine can outlast — this wait is given eight
    // minutes for exactly that reason. So a failure here is said out loud and the server is
    // asked anyway: the answer at its door is what settles it. The status is carried on so a
    // server that never does answer is explained by both facts rather than one.
    let gave_up = (!ended.success()).then_some(ended);
    if gave_up.is_some() {
        let line =
            format!("the command that starts Chalie ended with {ended}; waiting for it anyway");
        log::warn!("{line}");
        on_output(&line);
    }
    on_phase(Phase::Waiting);
    wait_until_ready(plan, gave_up).await
}

/// Run one command as the person's own interactive login shell, hand every line it prints to
/// `report` as it arrives, and answer with how it ended. `exec 2>&1` is what makes that one
/// stream rather than two: everything the shell and everything it starts would have written
/// to standard error goes to standard output instead, so a curl that fails is read in its
/// place in the output rather than being thrown away or arriving out of order.
///
/// `kill_on_drop` is for the installer alone: an app that quits mid-install should not leave
/// a download running. The command that starts Chalie is never killed with this app — it
/// launches somebody's server, which is meant to outlive the window that asked for it.
async fn run(
    plan: &Plan,
    command: &str,
    kill_on_drop: bool,
    report: impl Fn(&str),
) -> AppResult<ExitStatus> {
    let shell = plan.login_shell.display().to_string();
    let mut child = Command::new(&plan.login_shell)
        .args(["-i", "-l", "-c", &format!("exec 2>&1\n{command}")])
        // Closed rather than inherited, so that nothing in the pipeline can block on a read
        // nobody is there to answer. The app has no terminal of its own to lend either, so
        // anything that would rather ask goes on without asking.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        // A group of its own, so this app's own group is never what the installer's pipeline
        // or the server the start command launches belongs to.
        .process_group(0)
        .kill_on_drop(kill_on_drop)
        .spawn()
        .map_err(|problem| AppError::InstallFailed {
            message: format!("{shell} could not be started: {problem}"),
        })?;
    if let Some(output) = child.stdout.take() {
        let mut lines = BufReader::new(output).lines();
        // A line that cannot be read is reported as one, because output that stops without a
        // word is the one thing nobody can diagnose; the exit status below still decides.
        loop {
            match lines.next_line().await {
                Ok(None) => break,
                Ok(Some(line)) => report(&line),
                Err(problem) => {
                    report(&format!(
                        "(this output could not be read any further: {problem})"
                    ));
                    break;
                }
            }
        }
    }
    child
        .wait()
        .await
        .map_err(|problem| AppError::InstallFailed {
            message: format!("{shell} could not be waited for: {problem}"),
        })
}

/// Ask the server whether it has finished starting, until it says yes or until the bound runs
/// out. Why nobody answered is said in the order it is read: where it was asked and for how
/// long, then — when it is so — that the command meant to start it had already given up,
/// because that is the difference between a Chalie still coming up and one that never
/// started, and last the log that says which.
async fn wait_until_ready(plan: &Plan, start_gave_up: Option<ExitStatus>) -> AppResult<()> {
    let deadline = Instant::now() + READY_WITHIN;
    loop {
        if server::is_ready(&plan.address).await {
            log::info!("{} is ready", plan.address);
            return Ok(());
        }
        if Instant::now() >= deadline {
            let start = match start_gave_up {
                Some(status) => format!(" The command that starts it ended with {status}."),
                None => String::new(),
            };
            return Err(AppError::NotReady {
                message: format!(
                    "Chalie did not answer at {} within {} seconds.{start} Its own log is at {}.",
                    plan.address,
                    READY_WITHIN.as_secs(),
                    plan.chalie_log.display()
                ),
            });
        }
        sleep(READY_POLL).await;
    }
}

/// Say which part of the install is happening now. A phase nothing is listening for is
/// simply not shown, so this never fails an install that is otherwise going fine.
fn announce(app: &AppHandle, phase: Phase) {
    log::info!("install: {phase:?}");
    if let Err(problem) = app.emit(PHASE_EVENT, phase) {
        log::warn!("the wizard could not be told the install reached {phase:?}: {problem}");
    }
}

/// Every line, twice over: into the app's own log file, which is where a failure sends
/// somebody, and into the window, which is where they are watching it happen.
fn report(app: &AppHandle, line: &str) {
    log::info!(target: "installer", "{line}");
    if let Err(problem) = app.emit(OUTPUT_EVENT, line) {
        log::warn!("the installer's output could not be shown in the window: {problem}");
    }
}

/// Where this app writes its own log, which is where every line the install printed also
/// went. Named in the failure rather than left for somebody to find.
fn app_log_path(app: &AppHandle) -> String {
    match app.path().app_log_dir() {
        Ok(dir) => dir
            .join(&app.package_info().name)
            .with_extension("log")
            .display()
            .to_string(),
        Err(problem) => format!("this app's log folder, which could not be located: {problem}"),
    }
}

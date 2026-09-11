//! Putting a Chalie on this Mac: run the published installer, start what it leaves behind,
//! wait until the server answers, then give it the account that was filled in first and hand
//! this window the session. Every line those commands print goes two places at once — into
//! the window, where somebody is watching it happen, and into the app's own log file, which
//! is where a failure sends them afterwards.
//!
//! Both commands run with the PATH the person's login shell sets, read once before the
//! installer starts: an app opened from the Finder is given only the system's own
//! directories, and a Python new enough for the installer is somewhere a person put it
//! themselves, which only their shell's profile adds.

use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, WebviewWindow};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio::time::{sleep, timeout};

use crate::config::{Credentials, Mode, ServerAddress, ServerConfig, DEFAULT_PORT, LOCAL_HOST};
use crate::error::{AppError, AppResult};
use crate::server;
use crate::session;

/// The published one-liner, exactly as it is published. It is a shell pipeline, so it is
/// handed to a shell rather than run as a program.
const PUBLIC_INSTALLER: &str = "curl -fsSL https://chalie.ai/install | bash";
const SHELL: &str = "/bin/bash";

/// The shell asked for the PATH when `$SHELL` does not name one by its full path — any other
/// name would be looked up on the very PATH that is being asked for. It is the one macOS gives
/// every new account.
const DEFAULT_LOGIN_SHELL: &str = "/bin/zsh";
/// What the login shell runs to answer. `printenv` reports the exported PATH, which is the one
/// a command started from that shell would be given, and `exec` makes it the shell's last act:
/// nothing a profile sets to run on the way out — an exit trap, `.zlogout` — prints after it.
/// The `echo` ends any line a profile left unfinished, so the PATH always starts a line.
const PRINT_PATH: &str = "echo; exec /usr/bin/printenv PATH";
/// How long the login shell is given to answer. Long enough for a slow profile, short enough
/// that one waiting on something nobody can see delays the install rather than holding it.
const LOGIN_PATH_WITHIN: Duration = Duration::from_secs(15);
/// How long what the login shell printed is waited for once the shell has ended and everything
/// it started has been stopped. The pipe closes as soon as the last process holding it is gone,
/// so this only runs out when something left the shell's group and kept it open.
const OUTPUT_CLOSES_WITHIN: Duration = Duration::from_secs(1);

/// What the installer leaves in the home directory: the app it unpacks, the command that
/// starts a Chalie, and the file that command sends everything the server prints to.
const CHALIE_APP: &str = ".chalie/app";
const CHALIE_COMMAND: &str = ".local/bin/chalie";
const CHALIE_LOG: &str = ".chalie/chalie.log";
const START: &str = "start";

/// How long the server is given to come up before somebody is told it did not. Downloading
/// and unpacking a first install on a slow line is the long pole, and a bound that is too
/// short turns a working install into a failure nobody can explain.
const READY_WITHIN: Duration = Duration::from_secs(8 * 60);
/// How often it is asked.
const READY_POLL: Duration = Duration::from_secs(3);

/// How long the output of a running command is waited for before looking at whether Cancel
/// has been pressed. A command is stopped along with everything it started, but when that
/// signal cannot be sent only the one child this app holds is stopped, and a child of its own
/// can go on holding the pipes open — waiting on the output alone would then mean waiting on
/// something nobody asked for any more.
const CANCEL_CHECK: Duration = Duration::from_millis(250);

/// How many lines may be waiting to be shown before the readers have to slow down. Enough
/// that an ordinary burst is never held up, small enough that a command printing megabytes
/// cannot grow the app instead of the window.
const OUTPUT_BACKLOG: usize = 64;

/// What the wizard listens for: one event per line printed, one per part of the install
/// reached.
const OUTPUT_EVENT: &str = "install-output";
const PHASE_EVENT: &str = "install-phase";

/// One command an install runs.
#[derive(Debug, Clone)]
struct Step {
    program: PathBuf,
    arguments: Vec<String>,
}

/// What installing and starting a Chalie on this Mac depends on: the home it installs into,
/// the command that starts it, the shell that says the PATH both run with, where the server
/// will answer and where it keeps its own log.
#[derive(Debug, Clone)]
pub(crate) struct Plan {
    home: PathBuf,
    start: Step,
    login_shell: PathBuf,
    pub(crate) address: ServerAddress,
    chalie_log: PathBuf,
}

impl Plan {
    /// The home directory comes from the environment rather than from Tauri's own lookup because it
    /// is also what the installer itself reads: if the two disagreed, this app would be waiting on
    /// a Chalie installed somewhere else. The shell does too, because `$SHELL` is the one this
    /// person logs in with.
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
            start: Step {
                program: home.join(CHALIE_COMMAND),
                arguments: vec![START.to_string()],
            },
            login_shell: std::env::var_os("SHELL")
                .map(PathBuf::from)
                .filter(|shell| shell.is_absolute())
                .unwrap_or_else(|| PathBuf::from(DEFAULT_LOGIN_SHELL)),
            address: ServerAddress::parse(LOCAL_HOST, DEFAULT_PORT)?,
            chalie_log: home.join(CHALIE_LOG),
            home,
        })
    }

    /// The same plan, for code that only needs it to recognise this Mac's own
    /// Chalie. Without a HOME there is no install to find: that is logged and nothing is
    /// started, but nothing else is held up — a connect to any other server goes ahead.
    pub(crate) fn for_this_mac() -> Option<Self> {
        Self::public_installer()
            .inspect_err(|problem| log::warn!("this Mac's own Chalie cannot be started: {problem}"))
            .ok()
    }

    /// Whether the installer has run for this home: the app it unpacks and the command that
    /// starts it are both there.
    pub(crate) fn installed(&self) -> bool {
        self.home.join(CHALIE_APP).is_dir() && self.start.program.exists()
    }

    /// What a check of `config` that failed with `problem` found. Only this Mac's own Chalie
    /// counts as stopped: saved as Local, on the port this plan's Chalie serves, installed
    /// here, and nothing answering at all — not something else answering, not a login
    /// refused. `Mode` comes from the host alone, so `localhost` on a port a native install
    /// does not serve — a Docker-published one, say — is Local too; the port is what keeps
    /// this to the one Chalie this app can start.
    pub(crate) fn found(&self, config: &ServerConfig, problem: &AppError) -> Found {
        let stopped = matches!(problem, AppError::Unreachable { .. })
            && config.mode == Mode::Local
            && config.address.port() == self.address.port()
            && self.installed();
        if stopped {
            Found::Stopped
        } else {
            Found::NotOurs
        }
    }
}

/// Who asked for this Mac's own Chalie to be started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Asker {
    /// Somebody launching the app or pressing Try again, who is shown every step.
    User,
    /// The session watcher, in the background: nobody is shown anything.
    Watcher,
}

/// What a check of the saved server found, as far as starting this Mac's own Chalie goes.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Found {
    /// It answered.
    Answered,
    /// A server that is not this Mac's own, or a failure other than nothing answering.
    NotOurs,
    /// Nothing answered at this Mac's own Chalie, and it is installed here.
    Stopped,
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

/// Which of a command's two streams a line came out of. The installer says as much on one
/// as on the other, so both are shown and both are marked.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Stream {
    Stdout,
    Stderr,
}

#[derive(Debug, Clone, Serialize)]
struct Output {
    stream: Stream,
    line: String,
}

/// The one install or start there may be at a time, held as managed state so the Cancel
/// command can reach the running command without knowing anything else about it.
#[derive(Default)]
pub(crate) struct InstallState {
    running: Mutex<Running>,
    /// Set by any start of this Mac's own Chalie and cleared only by an answer, so the watcher
    /// and the relaunch between them start it once per outage, never once per check.
    started_this_outage: AtomicBool,
}

#[derive(Default)]
struct Running {
    /// Whether an install holds the slot. Not the same question as whether there is a child:
    /// an install is still running between its two commands and all the while it waits for
    /// the server to answer, with no child of its own at either moment.
    installing: bool,
    cancelled: bool,
    child: Option<Child>,
}

impl InstallState {
    /// Stop the install or start that is running, or the wait on one. Marking it is what makes
    /// each answer `InstallCancelled` rather than reporting whatever its command did. An
    /// installer is also killed, which is what makes that answer arrive now instead of in
    /// eight minutes; the command that starts Chalie never is ([`OnCancel::Leave`]). The mark
    /// stays until the slot is next claimed, so a relaunch waiting on a start still sees it
    /// after that start has let the slot go.
    pub(crate) fn cancel(&self) {
        let mut running = self.running();
        running.cancelled = true;
        log::info!("Cancel was pressed");
        if let Some(child) = running.child.as_mut() {
            kill(child);
        }
    }

    /// Take the slot, or refuse because it is taken. Two installs at once would be two
    /// copies of the installer writing the same files. The slot goes back when the answer is
    /// dropped.
    fn claim(&self) -> AppResult<Slot<'_>> {
        let mut running = self.running();
        if running.installing {
            return Err(AppError::InstallRunning);
        }
        *running = Running {
            installing: true,
            cancelled: false,
            child: None,
        };
        Ok(Slot(self))
    }

    /// One outage, one start: record what a check of the saved server found, and answer
    /// whether to start this Mac's own Chalie now. An answer ends the outage; a server that is
    /// not ours changes nothing; the user is always given a start, because somebody is waiting
    /// on it; the watcher only when nobody has started one in this outage yet.
    pub(crate) fn outage_step(&self, found: Found, asker: Asker) -> bool {
        let started = &self.started_this_outage;
        match (found, asker) {
            (Found::Answered, _) => {
                started.store(false, Ordering::SeqCst);
                false
            }
            (Found::NotOurs, _) => false,
            (Found::Stopped, Asker::User) => {
                started.store(true, Ordering::SeqCst);
                true
            }
            (Found::Stopped, Asker::Watcher) => !started.swap(true, Ordering::SeqCst),
        }
    }

    /// Hand the running command over so a Cancel has something to kill — or, if the Cancel
    /// came first, kill it now: the moment between starting a command and handing it over is
    /// exactly long enough for somebody to press the button.
    fn hold(&self, mut child: Child) {
        let mut running = self.running();
        if running.cancelled {
            kill(&mut child);
        }
        running.child = Some(child);
    }

    /// Take the command back to read how it ended. From here on a Cancel finds nothing to
    /// kill, which is right: there is nothing left running to stop.
    fn take(&self) -> Option<Child> {
        self.running().child.take()
    }

    fn cancelled(&self) -> bool {
        self.running().cancelled
    }

    /// A lock that cannot fail. Poisoning means some other thread panicked while holding it,
    /// and what it was holding is a flag and a process handle — neither of which a panic can
    /// leave half-written — so the state is taken as it stands rather than panicking again.
    fn running(&self) -> MutexGuard<'_, Running> {
        self.running
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// The slot, taken. Dropping it gives the slot back however the install or start ended — a
/// child still held goes with it, and a dropped child is a killed one — but keeps a Cancel's
/// mark, which only the next claim clears.
struct Slot<'a>(&'a InstallState);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        let mut running = self.0.running();
        *running = Running {
            cancelled: running.cancelled,
            ..Running::default()
        };
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
    run(app, plan, state).await?;
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

/// Install a Chalie on this Mac and leave it answering. Returns when the server is ready to
/// have an account made on it, or with the reason it is not.
async fn run(app: &AppHandle, plan: &Plan, state: &InstallState) -> AppResult<()> {
    let _slot = state.claim()?;
    install(app, plan, state).await
}

/// Say which part of the install is happening now. A phase nothing is listening for is
/// simply not shown, so this never fails an install that is otherwise going fine.
fn announce(app: &AppHandle, phase: Phase) {
    log::info!("install: {phase:?}");
    if let Err(problem) = app.emit(PHASE_EVENT, phase) {
        log::warn!("the wizard could not be told the install reached {phase:?}: {problem}");
    }
}

/// The PATH, the two commands and the wait, in the order they happen. The two commands are
/// not read the same way, because their failures do not mean the same thing: nothing is
/// installed if the installer fails, so that is the end of it, while the command that starts
/// Chalie can fail with Chalie still on its way up — so what decides there is the server's
/// own answer.
async fn install(app: &AppHandle, plan: &Plan, state: &InstallState) -> AppResult<()> {
    announce(app, Phase::Installing);
    let path = path_to_run_with(&plan.login_shell, |output| report(app, &output)).await;
    let path = path.as_deref();
    // A Cancel pressed while the login shell was being asked had nothing running to stop, so it
    // is only a mark until now — and honoured here, the installer never starts at all.
    if state.cancelled() {
        return Err(AppError::InstallCancelled);
    }
    let installed = execute(
        &Step {
            program: PathBuf::from(SHELL),
            arguments: vec!["-c".to_string(), PUBLIC_INSTALLER.to_string()],
        },
        path,
        state,
        OnCancel::Kill,
        |output| report(app, &output),
    )
    .await?;
    installer_ended(installed, state.cancelled(), &app_log_path(app))?;

    announce(app, Phase::Starting);
    start_then_wait(
        plan,
        state,
        path,
        |phase| announce(app, phase),
        |output| report(app, &output),
    )
    .await
}

/// Start a Chalie this Mac already has — nothing here is being installed — and wait until it
/// answers: the login-shell PATH read, the start step and the readiness wait `install` runs
/// after its own installer. Shares `state` with `install_and_register`, so a real install
/// claiming the slot is a reason this refuses to run rather than a second thing writing into
/// `~/.chalie` at once. A start the watcher asks for is shown to nobody: its phases and its
/// output go to the log alone.
pub(crate) async fn start_installed_chalie(
    app: &AppHandle,
    plan: &Plan,
    state: &InstallState,
    asker: Asker,
) -> AppResult<()> {
    let _slot = state.claim()?;
    let shown = asker == Asker::User;
    let on_phase = |phase| {
        if shown {
            announce(app, phase);
        } else {
            log::info!("background start: {phase:?}");
        }
    };
    let on_output = |output: Output| {
        if shown {
            report(app, &output);
        } else {
            log::info!(target: "installer", "{}", output.line);
        }
    };
    on_phase(Phase::Starting);
    let path = path_to_run_with(&plan.login_shell, on_output).await;
    // A Cancel pressed while the login shell was being asked had nothing running to stop, so
    // it is only a mark until now — and honoured here, the start command never runs at all.
    if state.cancelled() {
        return Err(AppError::InstallCancelled);
    }
    start_then_wait(plan, state, path.as_deref(), on_phase, on_output).await
}

/// The relaunch's start: this Mac's own Chalie started and waited for — or, when a start or an
/// install already holds the slot, only waited for, with the same limit and the same Cancel,
/// because that one is bringing up the same server.
pub(crate) async fn start_or_wait(
    app: &AppHandle,
    plan: &Plan,
    state: &InstallState,
) -> AppResult<()> {
    match start_installed_chalie(app, plan, state, Asker::User).await {
        Err(AppError::InstallRunning) => {
            log::info!("{} is already being started; waiting for it", plan.address);
            announce(app, Phase::Waiting);
            wait_until_ready(plan, state, None).await
        }
        outcome => outcome,
    }
}

/// The command that starts Chalie, then the wait for it to answer: the end of every install
/// and the whole of a start. A Cancel ends the waiting and leaves the command running
/// ([`OnCancel::Leave`]).
async fn start_then_wait(
    plan: &Plan,
    state: &InstallState,
    path: Option<&str>,
    on_phase: impl Fn(Phase),
    on_output: impl Fn(Output),
) -> AppResult<()> {
    let started = execute(&plan.start, path, state, OnCancel::Leave, &on_output).await?;
    let gave_up = start_ended(started, state.cancelled(), &on_output)?;
    on_phase(Phase::Waiting);
    wait_until_ready(plan, state, gave_up).await
}

/// The PATH the commands run with: the one the login shell reports, because the installer
/// looks for a `python3` of 3.11 or newer on it, and the one a person installed is only on
/// the PATH their own shell sets. When that cannot be read the commands still run, with this
/// app's own PATH — and say so, with the reason, in the window and in the log, because an
/// installer that then finds nothing is explained by it.
async fn path_to_run_with(shell: &Path, report: impl Fn(Output)) -> Option<String> {
    match login_path(shell).await {
        Ok(path) => {
            log::info!(
                "commands run with the PATH {} sets: {path}",
                shell.display()
            );
            Some(path)
        }
        Err(reason) => {
            let line = format!(
                "the PATH your login shell sets could not be read ({reason}); running with \
                 this app's own PATH instead"
            );
            log::warn!("{line}");
            report(Output {
                stream: Stream::Stderr,
                line,
            });
            None
        }
    }
}

/// Ask the login shell what PATH it ends up with. It is asked as an interactive login shell,
/// which is what Terminal opens: a PATH set up in `.zshrc` rather than `.zprofile`, where
/// version managers put theirs, is only there in one. The last line it prints is the answer,
/// and it has to be a path: a profile may say whatever it likes on the way in, and one that
/// hands over to some other program before the question is asked has not answered it.
///
/// It is the shell that is waited for, not its output: anything a profile starts in the
/// background holds that output open for as long as it runs.
async fn login_path(shell: &Path) -> Result<String, String> {
    let name = shell.display();
    let mut command = Command::new(shell);
    command
        .args(["-i", "-l", "-c", PRINT_PATH])
        // Closed, so a profile that asks something is not left waiting on an answer nobody is
        // there to type.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        // A shell this app stops waiting on is not left running: a dropped shell is a killed one.
        .kill_on_drop(true);
    // A session of its own, which is a group of its own with no terminal. A shell without a
    // terminal cannot turn job control on, so what its profile starts in the background stays in
    // its group, where stopping the group reaches it; and an app started from a terminal does
    // not hand its interactive shell one to stop itself on. Never `process_group` as well: a
    // process that already leads a group cannot start a session. `setsid` is one of the calls
    // that may be made between fork and exec, which is where this runs.
    unsafe {
        command.pre_exec(|| match libc::setsid() {
            -1 => Err(std::io::Error::last_os_error()),
            _ => Ok(()),
        });
    }
    let mut asking = command
        .spawn()
        .map_err(|problem| format!("{name} could not be started: {problem}"))?;
    // Taken before the wait, because a shell that has been waited for has no id any more — and
    // its id is also its group's.
    let group = asking.id().and_then(|id| libc::pid_t::try_from(id).ok());
    let Some(mut stdout) = asking.stdout.take() else {
        return Err(format!("{name} was started with nothing to answer on"));
    };
    // Read while the shell runs rather than after it, or a profile that prints more than a pipe
    // holds would never get as far as the answer.
    let mut reading = tauri::async_runtime::spawn(async move {
        let mut printed = Vec::new();
        stdout.read_to_end(&mut printed).await.map(|_| printed)
    });
    let ended = timeout(LOGIN_PATH_WITHIN, asking.wait()).await;
    // Answered or out of time, whatever the shell started is stopped now: that is what lets its
    // output close, and what keeps a profile's background job from outliving the question.
    if let Some(group) = group {
        if let Err(problem) = kill_group(group) {
            log::warn!(
                "what {name} started while it was asked for the PATH could not be stopped: \
                 {problem}"
            );
        }
    }
    let printed = timeout(OUTPUT_CLOSES_WITHIN, &mut reading).await;
    reading.abort();
    let status = ended
        .map_err(|_| {
            format!(
                "{name} did not answer within {} seconds",
                LOGIN_PATH_WITHIN.as_secs()
            )
        })?
        .map_err(|problem| format!("{name} could not be waited for: {problem}"))?;
    if !status.success() {
        return Err(format!("{name} ended with {status}"));
    }
    let printed = printed
        .map_err(|_| format!("{name} answered, but something it started kept its output open"))?
        .map_err(|problem| format!("what {name} printed was not read to the end: {problem}"))?
        .map_err(|problem| format!("what {name} printed could not be read: {problem}"))?;
    let printed = String::from_utf8_lossy(&printed);
    printed
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .filter(|line| line.starts_with('/'))
        .map(str::to_string)
        .ok_or_else(|| format!("{name} did not end on a PATH"))
}

/// The three ways the installer can end: somebody stopped it, it worked, or it failed — and
/// then there is nothing installed for the rest of this to start, so the whole of what it
/// printed, which is in the log, is the only thing that says why.
fn installer_ended(status: ExitStatus, cancelled: bool, log: &str) -> AppResult<()> {
    if cancelled {
        return Err(AppError::InstallCancelled);
    }
    if status.success() {
        return Ok(());
    }
    Err(AppError::InstallFailed {
        message: format!("the installer ended with {status}. Everything it printed is in {log}."),
    })
}

/// What the ending of the command that starts Chalie means, which is less than it looks. That
/// command waits half a minute for the server to write its pidfile and then exits non-zero,
/// which is a wait a first start on a slow machine can outlast — this one is given eight
/// minutes for exactly that reason. So a failure here is said out loud, in the window and in
/// the log, and the server is asked anyway: the answer at its door is what settles it. The
/// status is carried on so that a server which never does answer is explained by both facts
/// rather than one.
fn start_ended(
    status: ExitStatus,
    cancelled: bool,
    report: impl Fn(Output),
) -> AppResult<Option<ExitStatus>> {
    if cancelled {
        return Err(AppError::InstallCancelled);
    }
    if status.success() {
        return Ok(None);
    }
    let line = format!(
        "the command that starts Chalie ended with {status}; waiting for Chalie to answer anyway"
    );
    log::warn!("{line}");
    report(Output {
        stream: Stream::Stderr,
        line,
    });
    Ok(Some(status))
}

/// What a Cancel does to a command that is running.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OnCancel {
    /// Kill it and everything it started: an installer stopped halfway leaves nothing worth
    /// keeping, and its `curl` would otherwise go on writing files.
    Kill,
    /// Stop waiting for it and leave it running. The command that starts Chalie launches the
    /// server into its own process group, so killing that group would kill somebody's own
    /// Chalie while it boots. It is never held where a Cancel can reach it, and is not killed
    /// when it is dropped.
    Leave,
}

/// Start a command, hand every line it prints to `report` as it arrives, and answer with how
/// it ended. It runs with `path` as its PATH when there is one, and with this app's own when
/// there is not. Both streams are drained at once by their own readers: a command whose error
/// output nobody was reading would stop the moment that pipe filled up.
async fn execute(
    step: &Step,
    path: Option<&str>,
    state: &InstallState,
    on_cancel: OnCancel,
    report: impl Fn(Output),
) -> AppResult<ExitStatus> {
    let program = step.program.display().to_string();
    let mut command = Command::new(&step.program);
    if let Some(path) = path {
        command.env("PATH", path);
    }
    let mut child = command
        .args(&step.arguments)
        // Closed rather than inherited, so that nothing in the pipeline can block on a read
        // nobody is there to answer. The shell this app runs under has no terminal of its own
        // to lend either, so anything that would rather ask goes on without asking.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // A group of its own, so that stopping this command can reach everything it starts.
        // Without it the installer's `curl` and the shell it feeds survive the shell this app
        // holds, and go on downloading and writing files after somebody has been told the
        // install stopped.
        .process_group(0)
        .kill_on_drop(on_cancel == OnCancel::Kill)
        .spawn()
        .map_err(|problem| AppError::InstallFailed {
            message: format!("{program} could not be started: {problem}"),
        })?;
    let (lines, mut arriving) = mpsc::channel(OUTPUT_BACKLOG);
    if let Some(stdout) = child.stdout.take() {
        tauri::async_runtime::spawn(read(stdout, Stream::Stdout, lines.clone()));
    }
    if let Some(stderr) = child.stderr.take() {
        tauri::async_runtime::spawn(read(stderr, Stream::Stderr, lines.clone()));
    }
    // The readers own the only senders now. Without this the loop below would wait on a
    // channel that can never close, because this side would still be able to write to it.
    drop(lines);
    let kept = match on_cancel {
        OnCancel::Kill => {
            state.hold(child);
            None
        }
        OnCancel::Leave => Some(child),
    };
    loop {
        match timeout(CANCEL_CHECK, arriving.recv()).await {
            Ok(Some(output)) => report(output),
            // Both pipes are closed: the command has printed everything it is going to.
            Ok(None) => break,
            Err(_) if state.cancelled() => break,
            Err(_) => {}
        }
    }
    let mut child = match kept {
        // Only the waiting ends: the command goes on as it was started to, and whatever it
        // still prints goes to the log, so its pipes never close on it.
        Some(_) if state.cancelled() => {
            tauri::async_runtime::spawn(async move {
                while let Some(output) = arriving.recv().await {
                    log::info!(target: "installer", "{}", output.line);
                }
            });
            return Err(AppError::InstallCancelled);
        }
        Some(child) => child,
        None => state.take().ok_or_else(|| AppError::InstallFailed {
            message: format!(
                "nothing is holding {program} any more, so how it ended cannot be read"
            ),
        })?,
    };
    child
        .wait()
        .await
        .map_err(|problem| AppError::InstallFailed {
            message: format!("{program} could not be waited for: {problem}"),
        })
}

/// One pipe, line by line, into the shared channel. Ends when the pipe closes, when nobody
/// is listening any more, or on a read error — which is itself sent on as a line, because
/// output that stops without a word is the one thing nobody can diagnose.
async fn read<P: AsyncRead + Unpin + Send + 'static>(
    pipe: P,
    stream: Stream,
    lines: mpsc::Sender<Output>,
) {
    let mut reading = BufReader::new(pipe).lines();
    loop {
        match reading.next_line().await {
            Ok(None) => return,
            Ok(Some(line)) => {
                if lines.send(Output { stream, line }).await.is_err() {
                    return;
                }
            }
            Err(problem) => {
                let line = format!("(this output could not be read any further: {problem})");
                let _ = lines.send(Output { stream, line }).await;
                return;
            }
        }
    }
}

/// Ask the server whether it has finished starting, until it says yes or until the plan's
/// bound runs out. Nothing else can be done while it comes up, so a Cancel pressed here is
/// noticed between two questions rather than during one.
async fn wait_until_ready(
    plan: &Plan,
    state: &InstallState,
    start_gave_up: Option<ExitStatus>,
) -> AppResult<()> {
    let deadline = Instant::now() + READY_WITHIN;
    loop {
        if state.cancelled() {
            return Err(AppError::InstallCancelled);
        }
        if server::is_ready(&plan.address).await {
            log::info!("{} is ready", plan.address);
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(AppError::NotReady {
                message: never_answered(plan, start_gave_up),
            });
        }
        sleep(READY_POLL).await;
    }
}

/// Why nobody answered, in the order it is read: where it was asked and for how long, then —
/// when it is so — that the command meant to start it had already given up, because that is
/// the difference between a Chalie still coming up and one that never started, and last the
/// log that says which.
fn never_answered(plan: &Plan, start_gave_up: Option<ExitStatus>) -> String {
    let start = match start_gave_up {
        Some(status) => format!(" The command that starts it ended with {status}."),
        None => String::new(),
    };
    format!(
        "Chalie did not answer at {} within {} seconds.{start} Its own log is at {}.",
        plan.address,
        READY_WITHIN.as_secs(),
        plan.chalie_log.display()
    )
}

/// Every line, twice over: into the app's own log file, which is where a failure sends
/// somebody, and into the window, which is where they are watching it happen.
fn report(app: &AppHandle, output: &Output) {
    log::info!(target: "installer", "{}", output.line);
    if let Err(problem) = app.emit(OUTPUT_EVENT, output) {
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

/// Stop a command and everything it started. Each one runs in a process group of its own, so
/// the signal goes to the whole group rather than to the one process this app happens to
/// hold: the installer is a shell pipeline, and stopping only the shell leaves the download
/// running under a new parent, which is an install still happening after it was stopped.
fn kill(child: &mut Child) {
    if let Some(group) = child.id().and_then(|id| libc::pid_t::try_from(id).ok()) {
        match kill_group(group) {
            Ok(()) => return,
            Err(problem) => {
                log::warn!("not everything the install started could be stopped: {problem}");
            }
        }
    }
    // The group could not be reached, so at least the command this app is holding stops.
    if let Err(problem) = child.start_kill() {
        log::warn!("the install's command could not be stopped: {problem}");
    }
}

/// Send every process in `group` the signal nothing can ignore. A group that is already gone
/// is not a failure: everything that was in it has stopped, which is what was wanted.
fn kill_group(group: libc::pid_t) -> std::io::Result<()> {
    // A negated process id names that process's group. It is two integers into the kernel,
    // touching nothing this process owns, and the standard library offers no safe way to ask
    // for it.
    if unsafe { libc::kill(-group, libc::SIGKILL) } == 0 {
        return Ok(());
    }
    let problem = std::io::Error::last_os_error();
    if problem.raw_os_error() == Some(libc::ESRCH) {
        return Ok(());
    }
    Err(problem)
}

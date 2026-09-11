//! The one failure type the whole app speaks. Every variant is serialised to the wizard
//! with a `kind` tag so the UI shows one specific sentence per failure instead of a
//! stack trace, and nothing that reaches the UI ever carries a password or a cookie.

use std::fmt;

use serde::Serialize;

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum AppError {
    /// The address could not be turned into a URL at all.
    InvalidAddress { message: String },
    /// Nothing answered: wrong host or port, server down, name does not resolve.
    Unreachable { message: String },
    /// Something answered, but it does not behave like a Chalie server.
    NotChalie { message: String },
    /// The username or password was rejected.
    InvalidCredentials,
    /// Too many login attempts in a row.
    RateLimited { retry_after_seconds: Option<u64> },
    /// A Chalie server answered, but with a failure of its own.
    ServerFailed { status: u16 },
    /// The stored configuration could not be read or written.
    Config { message: String },
    /// The webview refused the session cookie or the navigation.
    Webview { message: String },
    /// One of the commands the install runs did not finish cleanly. The message says which
    /// command, how it ended and where everything it printed can be read back.
    InstallFailed { message: String },
    /// The install was stopped from the window while it was running.
    InstallCancelled,
    /// An install was asked for while one was already running.
    InstallRunning,
    /// Chalie was installed and started, but never began answering.
    NotReady { message: String },
    /// This Chalie already has a master account, so there is nothing to create.
    AccountExists,
    /// The server would not accept the username or password for a new account, and said
    /// what was wrong with them.
    InvalidAccount { message: String },
}

pub(crate) type AppResult<T> = Result<T, AppError>;

/// How a refused address begins when rendered — shared with the settings-file reader so it
/// can recognise one and let it through whole: a host is not a secret, and a refusal that
/// hides the host it refused is a dead end nobody can fix.
pub(crate) const INVALID_ADDRESS_PREFIX: &str = "invalid address: ";

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAddress { message } => write!(f, "{INVALID_ADDRESS_PREFIX}{message}"),
            Self::Unreachable { message } => write!(f, "unreachable: {message}"),
            Self::NotChalie { message } => write!(f, "not a Chalie server: {message}"),
            Self::InvalidCredentials => f.write_str("invalid credentials"),
            Self::RateLimited {
                retry_after_seconds,
            } => match retry_after_seconds {
                Some(seconds) => write!(f, "rate limited, retry in {seconds}s"),
                None => f.write_str("rate limited"),
            },
            Self::ServerFailed { status } => write!(f, "the server answered {status}"),
            Self::Config { message } => write!(f, "configuration: {message}"),
            Self::Webview { message } => write!(f, "webview: {message}"),
            Self::InstallFailed { message } => write!(f, "the install failed: {message}"),
            Self::InstallCancelled => f.write_str("the install was cancelled"),
            Self::InstallRunning => f.write_str("an install is already running"),
            Self::NotReady { message } => write!(f, "Chalie is not answering yet: {message}"),
            Self::AccountExists => f.write_str("this Chalie already has an account"),
            Self::InvalidAccount { message } => {
                write!(f, "those account details were refused: {message}")
            }
        }
    }
}

impl std::error::Error for AppError {}

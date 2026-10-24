//! Everything this app asks a Chalie server over HTTP. The session cookie the login
//! returns is wrapped so it cannot be printed by accident.

use std::fmt;
use std::time::Duration;

use reqwest::header::{COOKIE, RETRY_AFTER, SET_COOKIE};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use tauri::webview::Cookie;
use tauri::Url;

use crate::config::{Credentials, ServerAddress};
use crate::error::{AppError, AppResult};

/// The cookie Chalie sets on a successful login and expects back on every later request.
pub(crate) const SESSION_COOKIE_NAME: &str = "chalie_session";

const HEALTH_PATH: &str = "health";
/// Answered only once the server has finished starting, which `health` does not wait for.
const READY_PATH: &str = "ready";
const STATUS_PATH: &str = "api/auth/status";
const LOGIN_PATH: &str = "api/auth/login";
const REGISTER_PATH: &str = "api/auth/register";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum VaultState {
    Unlocked,
    Locked,
    Uninitialized,
    /// A state added by a newer server than this app knows about.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ServerStatus {
    pub(crate) has_master_account: bool,
    pub(crate) vault_state: VaultState,
    /// Whether the request that asked carried a live session. The only honest answer to
    /// that question: a Chalie keeps its vault key in memory, so a restart locks the vault
    /// and signs everybody out while leaving their cookies valid for weeks — every other
    /// endpoint keeps answering 200 to one, and this field alone says false.
    pub(crate) has_session: bool,
}

/// A logged-in session: the cookie the server set, parsed and owned. Carrying the whole
/// cookie rather than its value is what lets the webview be handed the server's own
/// attributes instead of a guess at them. Refuses to print itself.
#[derive(Clone)]
pub(crate) struct SessionCookie(Cookie<'static>);

impl SessionCookie {
    /// The cookie exactly as the server sent it, attributes and all.
    pub(crate) fn cookie(&self) -> &Cookie<'static> {
        &self.0
    }

    /// The session the webview is already holding, taken back out of its cookie store so it
    /// can be sent as one. Deliberately not public beyond this crate: outside it, a session
    /// is something a login returns, not something anything else may declare a cookie to be.
    pub(crate) fn from_cookie(cookie: Cookie<'static>) -> Self {
        Self(cookie)
    }
}

impl fmt::Debug for SessionCookie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SessionCookie(<redacted>)")
    }
}

/// Does anything answer at this address at all?
async fn health(address: &ServerAddress) -> AppResult<()> {
    let url = endpoint(address, HEALTH_PATH)?;
    let response = get(&url).await?;
    if response.status().is_success() {
        return Ok(());
    }
    Err(not_chalie(&url, response.status()))
}

/// Has a Chalie that is starting up finished? Everything that is not a 200 — its own
/// not-ready answer, a connection refused while it is still booting, a name that does not
/// resolve yet — means the same thing here, and the only thing waiting on it can do is ask
/// again, so this answers yes or no rather than raising.
pub(crate) async fn is_ready(address: &ServerAddress) -> bool {
    let url = match endpoint(address, READY_PATH) {
        Ok(url) => url,
        Err(problem) => {
            log::warn!("{address} cannot be asked whether it is ready: {problem}");
            return false;
        }
    };
    matches!(get(&url).await, Ok(response) if response.status().is_success())
}

/// Is there a Chalie here, and what does it say about itself?
pub(crate) async fn probe(address: &ServerAddress) -> AppResult<ServerStatus> {
    health(address).await?;
    status(address).await
}

/// What the server says about itself to a caller carrying no session.
async fn status(address: &ServerAddress) -> AppResult<ServerStatus> {
    status_with_session(address, None).await
}

/// The same question asked as the session the window is holding, which is the only way to
/// learn whether that session still works: the answer's `has_session` is about the request
/// that asked it.
pub(crate) async fn status_with_session(
    address: &ServerAddress,
    session: Option<&SessionCookie>,
) -> AppResult<ServerStatus> {
    let url = endpoint(address, STATUS_PATH)?;
    let mut request = client()?.get(url.clone());
    if let Some(session) = session {
        let cookie = session.cookie();
        // Name and value only, which is all a browser sends back: the attributes are the
        // server's instructions for storing it, never part of what is returned.
        request = request.header(COOKIE, format!("{}={}", cookie.name(), cookie.value()));
    }
    let response = request.send().await.map_err(|e| send_failed(&url, &e))?;
    if !response.status().is_success() {
        return Err(not_chalie(&url, response.status()));
    }
    response
        .json::<ServerStatus>()
        .await
        .map_err(|e| AppError::NotChalie {
            message: format!("{url} did not return Chalie's auth status: {e}"),
        })
}

pub(crate) async fn login(
    address: &ServerAddress,
    credentials: &Credentials,
) -> AppResult<SessionCookie> {
    let url = endpoint(address, LOGIN_PATH)?;
    let response = client()?
        .post(url.clone())
        .json(credentials)
        .send()
        .await
        .map_err(|e| send_failed(&url, &e))?;
    let status = response.status();
    log::info!("login {url} answered {}", status.as_u16());
    match status {
        StatusCode::OK => session_cookie(&response),
        StatusCode::UNAUTHORIZED => Err(AppError::InvalidCredentials),
        StatusCode::TOO_MANY_REQUESTS => Err(AppError::RateLimited {
            retry_after_seconds: retry_after_seconds(&response),
        }),
        other => Err(AppError::ServerFailed {
            status: other.as_u16(),
        }),
    }
}

/// Create this Chalie's master account. The only thing that creates it, and the server
/// hands back a session for the account it just made, so whoever created it is signed in
/// without a second round trip.
pub(crate) async fn register(
    address: &ServerAddress,
    credentials: &Credentials,
) -> AppResult<SessionCookie> {
    let url = endpoint(address, REGISTER_PATH)?;
    let response = client()?
        .post(url.clone())
        .json(credentials)
        .send()
        .await
        .map_err(|e| send_failed(&url, &e))?;
    let status = response.status();
    log::info!("register {url} answered {}", status.as_u16());
    match status {
        StatusCode::CREATED => session_cookie(&response),
        StatusCode::CONFLICT => Err(AppError::AccountExists),
        StatusCode::UNPROCESSABLE_ENTITY => Err(AppError::InvalidAccount {
            message: refusal(response).await,
        }),
        other => Err(AppError::ServerFailed {
            status: other.as_u16(),
        }),
    }
}

/// What the server said was wrong with the account details. Its own headline is the same
/// sentence for every refusal it makes, so the reasons it lists beneath are what carries the
/// meaning; the headline is used only when it listed none.
async fn refusal(response: reqwest::Response) -> String {
    #[derive(Deserialize)]
    struct Refused {
        error: String,
        #[serde(default)]
        details: Vec<Reason>,
    }
    #[derive(Deserialize)]
    struct Reason {
        msg: String,
    }

    let Ok(refused) = response.json::<Refused>().await else {
        return "the server did not say what was wrong with them".to_string();
    };
    if refused.details.is_empty() {
        return refused.error;
    }
    refused
        .details
        .into_iter()
        .map(|reason| reason.msg)
        .collect::<Vec<_>>()
        .join("; ")
}

fn endpoint(address: &ServerAddress, path: &str) -> AppResult<Url> {
    address
        .base_url()?
        .join(path)
        .map_err(|e| AppError::InvalidAddress {
            message: format!("{address}/{path}: {e}"),
        })
}

async fn get(url: &Url) -> AppResult<reqwest::Response> {
    client()?
        .get(url.clone())
        .send()
        .await
        .map_err(|e| send_failed(url, &e))
}

fn client() -> AppResult<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        // Nothing was sent, so no server is to blame: this is the app's own setup failing,
        // and never `Unreachable`, which would have it start a Chalie.
        .map_err(|e| AppError::Config {
            message: format!("no HTTP client: {e}"),
        })
}

fn session_cookie(response: &reqwest::Response) -> AppResult<SessionCookie> {
    for header in response.headers().get_all(SET_COOKIE) {
        let Ok(text) = header.to_str() else { continue };
        let Ok(cookie) = Cookie::parse(text) else {
            continue;
        };
        if cookie.name() == SESSION_COOKIE_NAME {
            return Ok(SessionCookie(cookie.into_owned()));
        }
    }
    Err(AppError::NotChalie {
        message: format!("the login succeeded but set no {SESSION_COOKIE_NAME} cookie"),
    })
}

fn retry_after_seconds(response: &reqwest::Response) -> Option<u64> {
    response
        .headers()
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

/// A request that got no response. Only a connection refused or never made, and a wait for an
/// answer that ran out, mean nothing is answering — the one finding that may start this Mac's
/// own Chalie. Anything else went wrong after something took the connection: it closed it
/// without a reply, or replied in something that is not HTTP.
fn send_failed(url: &Url, error: &reqwest::Error) -> AppError {
    let message = format!("{url}: {error}");
    if error.is_connect() || error.is_timeout() {
        AppError::Unreachable { message }
    } else {
        AppError::NotChalie { message }
    }
}

fn not_chalie(url: &Url, status: StatusCode) -> AppError {
    AppError::NotChalie {
        message: format!("{url} answered {}", status.as_u16()),
    }
}

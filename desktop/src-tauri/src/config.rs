//! Where the app remembers which Chalie it talks to and the login it talks to it with: one
//! plain JSON file in the app's own data folder, holding the whole of a Mac's setup.

use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, Url};

use crate::error::{AppError, AppResult, INVALID_ADDRESS_PREFIX};

/// The port a Chalie server listens on unless its owner changed it.
pub(crate) const DEFAULT_PORT: u16 = 31025;
/// The host a Chalie running on this Mac is reached at.
pub(crate) const LOCAL_HOST: &str = "localhost";

const CONFIG_FILE_NAME: &str = "config.json";
/// The half-written copy, kept beside the real file so replacing it is one rename inside
/// one directory.
const CONFIG_TEMP_FILE_NAME: &str = "config.json.tmp";
const LOOPBACK_IPV4: &str = "127.0.0.1";
/// Bracketed, because that is how a URL spells an IPv6 host and every host here is
/// canonical (see [`ServerAddress`]).
const LOOPBACK_IPV6: &str = "[::1]";

/// Whether the server is this Mac's own Chalie or one on another machine. Kept so later
/// slices can offer to start, stop or update an instance they own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Mode {
    Remote,
    Local,
}

impl Mode {
    /// A Chalie reached over the loopback interface is this Mac's own; anything else is
    /// somebody else's machine, however it was found. Takes the canonical host from
    /// [`ServerAddress::parse`], and matches it the way names are compared everywhere
    /// else: without regard to case.
    pub(crate) fn for_host(host: &str) -> Self {
        if [LOCAL_HOST, LOOPBACK_IPV4, LOOPBACK_IPV6]
            .iter()
            .any(|loopback| host.eq_ignore_ascii_case(loopback))
        {
            Self::Local
        } else {
            Self::Remote
        }
    }
}

/// One Chalie's address. Both fields are canonical — the host spelled exactly as a URL
/// spells it, which for a name means lower case — because [`ServerAddress::parse`] is the
/// only thing that builds them out of what somebody typed or a file held: the fields are
/// private, and deserialising one goes through `parse` too. Everything downstream compares
/// against them literally, the cookie domain the webview reports back after it has
/// lower-cased it above all.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ServerAddress {
    host: String,
    port: u16,
}

impl ServerAddress {
    /// The one door in. The host is trimmed and read as a URL host, and anything the URL
    /// parser would reinterpret — a scheme, a path, a query, a user name, a second port —
    /// is refused instead of quietly used, because each of those points the login, the
    /// cookie domain and the navigation at three different places. What the parser says
    /// the host is, is what is kept.
    pub(crate) fn parse(host: &str, port: u16) -> AppResult<Self> {
        let (host, _) = Self::checked(host, port)?;
        Ok(Self { host, port })
    }

    /// The canonical host, spelled the way a URL spells it.
    pub(crate) fn host(&self) -> &str {
        &self.host
    }

    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    /// The root the product UI is served from, and the base every API call joins onto.
    pub(crate) fn base_url(&self) -> AppResult<Url> {
        Ok(Self::checked(&self.host, self.port)?.1)
    }

    /// The canonical host and the URL it names — the single place a URL is built from an
    /// address, so the address that is proven is the address that is used.
    fn checked(host: &str, port: u16) -> AppResult<(String, Url)> {
        let typed = host.trim();
        // The host is not a secret, so the refusal quotes it: a typed address nobody can
        // see is a dead end nobody can fix.
        let refuse = || AppError::InvalidAddress {
            message: format!(
                "{typed:?} is not a host: enter only a host name or IP address, with no \
                 scheme, path or port in the host field"
            ),
        };
        // The root dot a fully-qualified name is often typed with. One is dropped here so
        // that spelling works; what is left still has to survive every check below.
        let host = typed.strip_suffix('.').unwrap_or(typed);
        let url = Url::parse(&format!("http://{host}:{port}/")).map_err(|_| refuse())?;
        let Some(canonical) = url.host_str() else {
            return Err(refuse());
        };
        let intact = url.port_or_known_default() == Some(port)
            && url.path() == "/"
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none()
            && canonical.eq_ignore_ascii_case(host);
        if !intact {
            return Err(refuse());
        }
        // The invariant, stated where it cannot drift: no canonical host ends in a root dot.
        // WebKit strips that dot from every cookie domain it stores, so a host kept with one
        // would pass every check here, log in, and then fail its own cookie read-back on
        // every launch. Dropping one dot above repairs the spelling people type; anything
        // still ending in a dot after that is refused rather than repaired further.
        if canonical.ends_with('.') {
            return Err(refuse());
        }
        let canonical = canonical.to_string();
        Ok((canonical, url))
    }
}

/// Deserialising is the other way an address arrives — out of the configuration file, which
/// a text editor or an older version of this app can have written. It goes through
/// [`ServerAddress::parse`] rather than straight into the fields, so a host the wizard would
/// have refused cannot be minted by reading a file either. The shadow struct is what serde
/// would have derived, and is the only place the field names are spelled out.
impl<'de> Deserialize<'de> for ServerAddress {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Stored {
            host: String,
            port: u16,
        }
        let stored = Stored::deserialize(deserializer)?;
        Self::parse(&stored.host, stored.port).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for ServerAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.host, self.port)
    }
}

/// The Chalie login. One definition for the two things it is — a field of the settings file
/// and the body the login request posts — so what is kept and what is sent cannot drift
/// apart. Redacts itself, so a stray `{:?}` anywhere cannot put a password in the log.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Credentials {
    pub(crate) username: String,
    pub(crate) password: String,
}

impl fmt::Debug for Credentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Credentials")
            .field("username", &self.username)
            .field("password", &"<redacted>")
            .finish()
    }
}

/// The whole of one Mac's setup: which Chalie, whether it is this Mac's own, and the login
/// that opens it. Both halves are flattened in, so the file is one flat object spelled the
/// way the wizard's own fields are — `host`, `port`, `mode`, `username`, `password` — and a
/// person who opens it reads settings rather than a structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ServerConfig {
    #[serde(flatten)]
    pub(crate) address: ServerAddress,
    pub(crate) mode: Mode,
    #[serde(flatten)]
    pub(crate) credentials: Credentials,
}

pub(crate) fn config_path(app: &AppHandle) -> AppResult<PathBuf> {
    let dir = app.path().app_config_dir().map_err(|e| AppError::Config {
        message: format!("no config directory: {e}"),
    })?;
    Ok(dir.join(CONFIG_FILE_NAME))
}

/// The configuration, or [`None`] when this Mac has never been set up. A file that exists
/// but cannot be understood is an error, never a silent fresh start. A host the wizard
/// would have refused is one of the ways it cannot be understood, because [`ServerAddress`]
/// deserialises through [`ServerAddress::parse`]; a login half of it is missing is another,
/// because a configuration naming a server nothing can open is not a setup, it is a file to
/// be written over.
pub(crate) fn read(app: &AppHandle) -> AppResult<Option<ServerConfig>> {
    let path = config_path(app)?;
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(AppError::Config {
                message: format!("cannot read {}: {e}", path.display()),
            })
        }
    };
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|e| AppError::Config {
            message: format!(
                "{} is not a Chalie configuration: {}",
                path.display(),
                said_without_quoting_the_file(&e)
            ),
        })
}

/// Why the file was refused, said without repeating anything that was in it. serde spells
/// the offending value into its own message — a password typed as a number comes back as
/// ``invalid type: integer `1234` `` — and this sentence is headed for the log and the
/// wizard's screen. Two messages are let through: the one that names a missing field,
/// which carries the field's name and nothing from the file, and a refused address, which
/// quotes the typed host on purpose (see [`ServerAddress::checked`]). Every other complaint
/// about a value is replaced. A syntax problem is passed through whole: the parser's wording
/// is fixed and its position is the fault's. The position serde appends to a complaint about
/// a value is dropped instead — every field here arrives through a flattened struct, and
/// serde places every such complaint at the closing brace, which is never where the fault is.
fn said_without_quoting_the_file(problem: &serde_json::Error) -> String {
    use serde_json::error::Category;

    const SAFE_TO_REPEAT: [&str; 2] = ["missing field `", INVALID_ADDRESS_PREFIX];
    let said = problem.to_string();
    if matches!(problem.classify(), Category::Syntax | Category::Eof) {
        return said;
    }
    let position = format!(" at line {} column {}", problem.line(), problem.column());
    let said = said.strip_suffix(&position).unwrap_or(said.as_str());
    if SAFE_TO_REPEAT.iter().any(|family| said.starts_with(family)) {
        return said.to_string();
    }
    "something in it is not what belongs there".to_string()
}

/// Written beside the real file and renamed over it. Writing in place truncates first, so a
/// crash in the middle of one leaves a file that parses as nothing and a person retyping a
/// password nothing holds any more; a rename either happened or did not. The whole setup is
/// written at once, so the login on disk is always the login for the server on disk — and
/// the server this app used to point at leaves no credentials behind, because there is only
/// ever the one file and this replaced it.
pub(crate) fn write(app: &AppHandle, config: &ServerConfig) -> AppResult<()> {
    let path = config_path(app)?;
    let dir = path.parent().ok_or_else(|| AppError::Config {
        message: format!("{} is not inside a directory", path.display()),
    })?;
    fs::create_dir_all(dir).map_err(|e| AppError::Config {
        message: format!("cannot create {}: {e}", dir.display()),
    })?;
    let text = serde_json::to_string_pretty(config).map_err(|e| AppError::Config {
        message: format!("cannot serialise the configuration: {e}"),
    })?;
    let temp = dir.join(CONFIG_TEMP_FILE_NAME);
    write_owner_only(&temp, &text).map_err(|e| AppError::Config {
        message: format!("cannot write {}: {e}", temp.display()),
    })?;
    fs::rename(&temp, &path).map_err(|e| AppError::Config {
        message: format!(
            "cannot move {} onto {}: {e}",
            temp.display(),
            path.display()
        ),
    })
}

/// The file holds a password, so it is created readable and writable by its owner and
/// nobody else — at creation, not afterwards, because a file that is briefly world-readable
/// is a file that was readable. The rename carries that mode onto the real file. Elsewhere
/// than Unix the plain create runs and the platform's own defaults apply.
fn write_owner_only(path: &Path, text: &str) -> std::io::Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    options.mode(0o600);
    options.open(path)?.write_all(text.as_bytes())
}

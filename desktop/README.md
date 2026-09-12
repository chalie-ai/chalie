# Chalie desktop app

The macOS desktop app for Chalie: a Tauri 2 app that bundles nothing but the
first-run wizard.

## What this project is

The wizard (Vue 3 + Vite + TypeScript, in `src/`) asks which Chalie server to
use and for a username and password. Rust (`src-tauri/src/`) logs in, remembers
the server and that login in the app's own settings file, hands the session
cookie to the webview and points the window at the server.

A server with no account yet has nobody to sign in as, so the wizard asks for
the account to create instead, creates it on that server, remembers it and signs
you in. That is the same whether the wizard found the server on this Mac or you
typed its address.

Everything after that comes from the server: the product UI is the same frontend
the web interface serves, so nothing product-specific lives here.

The app keeps you signed in. If the server restarts, or the session expires, it
signs in again with the login kept in the settings file and puts you back where
you were; it only asks for a password again if that stored login is refused.

## Links, Brain, downloads and the microphone

The server's own pages stay in the app. A web, mail or phone link to anywhere
else opens in the app your Mac uses for that kind of link, and anything else a
page tries to open is refused.

Brain opens in a window of its own, already signed in. When the app signs itself
in again, after a restart for instance, that window comes back too — on Brain,
not on the sign-in page. Closing the main window quits the app, and every Brain
window with it.

Files the server's pages save, such as an export, go to your Downloads folder.
Nothing from anywhere else is saved.

The first time you press the microphone button, macOS asks whether Chalie may use
the microphone. WebKit only offers the microphone to pages it treats as secure,
which over plain `http` means a Chalie running on this Mac.

View > Reload (Command-R) reloads the window in front.

## Installing Chalie on this Mac

The wizard's other branch installs a Chalie here. It asks for the account to
create first, then runs the published installer — the same `curl … | bash`
one-liner the website gives you — starts what that leaves behind, waits until
the new server answers, creates that first account and signs you in. Every line
the installer and the start command print is shown as it arrives and written to
the app's log, which is where a failure points you.

An app opened from Finder or the Dock starts with only the Mac's system folders
on its PATH, which is not where Homebrew or a Python version manager put their
commands. So the installer and the start command are run by your own login
shell, exactly as the published instructions assume when they tell you to paste
the one-liner into a terminal — anything your profile puts on the PATH is on it
here too. Both send everything they print to one stream, so a download that
fails is read in its place in the output rather than being thrown away.

The installer needs no administrator password and writes under your home
directory only: `~/.chalie` for the server and its data, `~/.local/bin/chalie`
for the command that runs it. There is no way to stop an install part-way:
quitting the app ends it. Running the installer again over a half-finished
install is safe.

The command that starts Chalie gives up waiting for it after half a minute,
which a first start on a slow machine can outlast, so a failure there is shown
and then waited out: what decides is whether the server answers, not what that
command exited with. If nothing ever answers, the failure says so and names
that exit status alongside Chalie's own log, so you know which of the two to
read.

The account is the last step, and a refusal there costs you nothing that was
already done. Details Chalie turns down come straight back on the form to be
corrected, and correcting them creates the account on the Chalie now running
here — nothing is installed or started a second time. If that Chalie already
has an account, there is nothing to create at all, and the wizard offers you
its sign-in form instead.

The installer registers nothing that starts Chalie when the Mac boots. The app
starts it instead, but only when the saved server is this Mac's own install on
localhost:31025 and nothing answers there — the connection is refused, or
times out. A server that answers, even wrongly, is never restarted. A Chalie
you pointed the wizard at somewhere else is left alone, and so is a server on
this Mac reached through anything other than the port the installer uses.

Only one start runs at a time, whoever asked for it: a start at launch and the
background check that keeps watching for the server going away share the same
slot, and the second one to arrive waits for the first rather than running its
own. A start the background check makes shows nothing on screen; if it fails,
that goes only to the app's log. "Try again" on the connecting screen starts
Chalie again the same way. Leaving that screen stops only the app's wait for
it — the Chalie that is booting keeps booting.

## Prerequisites

- Rust, stable toolchain
- Node and pnpm
- Xcode command line tools (`xcode-select --install`)

## Build and run

From `desktop/`:

```bash
pnpm install
pnpm tauri dev      # run the app against the Vite dev server
pnpm tauri build    # .app and .dmg in src-tauri/target/release/bundle/
```

## Gates

From `desktop/`:

```bash
cd src-tauri
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cd ..
pnpm typecheck
```

## Where things live on the Mac

- Settings: `~/Library/Application Support/ai.chalie.app/config.json` — the
  server host, port and mode, and the username and password used to sign in.
  Plain JSON, readable and writable by you alone.
- Logs: `~/Library/Logs/ai.chalie.app/Chalie.log`.

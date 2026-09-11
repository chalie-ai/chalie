//! The Chalie desktop app. It bundles nothing but the setup wizard: once a server is
//! known, the window is pointed at that server and the product UI comes from there.

mod commands;
mod config;
mod error;
mod install;
mod server;
mod session;
mod watcher;
mod windows;

use tauri::webview::PageLoadEvent;
use tauri::{Manager, WindowEvent};
use tauri_plugin_log::{Target, TargetKind};

pub fn run() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::new()
                .level(log::LevelFilter::Info)
                // `targets` replaces the plugin's defaults; `target` appends to them, which
                // registers stdout and the log file twice and writes every line twice.
                .targets([
                    Target::new(TargetKind::Stdout),
                    Target::new(TargetKind::LogDir { file_name: None }),
                ])
                .build(),
        )
        // Every page the window loads is logged, which is how a run is read back: the
        // bundled wizard first, then the server's own URL once the session is in place.
        // Both ends of the load are recorded so a navigation that starts and never finishes
        // — a server that went away between login and navigate — is visible in the log.
        .on_page_load(|webview, payload| {
            match payload.event() {
                PageLoadEvent::Started => log::info!("page load started: {}", payload.url()),
                PageLoadEvent::Finished => log::info!("page loaded: {}", payload.url()),
            }
            // The server's own sign-in page landing is the app being told the session died,
            // so the watcher hears about it here rather than waiting for its next check.
            watcher::on_page_load(webview, payload);
        })
        .menu(windows::menu)
        .on_menu_event(windows::on_menu_event)
        // Tauri only quits once the last window is gone, and the session is watched through the
        // main window alone: a Brain window left open after it would keep the app running with
        // nothing keeping it signed in.
        .on_window_event(|window, event| {
            if matches!(event, WindowEvent::Destroyed) && window.label() == watcher::MAIN_WINDOW {
                window.app_handle().exit(0);
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_setup_state,
            commands::detect_local,
            commands::probe_server,
            commands::connect,
            commands::auto_connect,
            commands::install_local,
            commands::cancel_install,
            commands::create_account,
        ])
        .setup(|app| {
            log::info!(
                "config file: {}",
                config::config_path(app.handle())?.display()
            );
            log::info!("log directory: {}", app.path().app_log_dir()?.display());
            // One install at a time, reachable from both the command that starts one and the
            // command that stops it.
            app.manage(install::InstallState::default());
            // Before the watcher, which reads the page the window opens on.
            windows::open_main(app.handle())?;
            watcher::install(app.handle());
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

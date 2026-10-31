//! Worker process that listens for the allocator's TCP acknowledgment requests.
//!
//! Accepts one optional bind address, defaulting to `127.0.0.1:7001`. Start a
//! separate instance on each compute node, or on distinct ports for local use.

use cluster_allocator::{handle_connection, DEFAULT_TIMEOUT};
use std::env;
use std::io;
use std::net::TcpListener;
use std::process::ExitCode;

/// Parse the bind address and serve connections until stopped or accept fails.
///
/// Request errors are logged without terminating the listener. Argument, bind,
/// and accept errors are returned to the entry point for a failing exit status.
fn run() -> io::Result<()> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.len() == 1 && matches!(arguments[0].as_str(), "--help" | "-h") {
        println!("Usage: compute-node [BIND_HOST:PORT]");
        println!("Default: 127.0.0.1:7001; use 0.0.0.0:7001 for remote access.");
        return Ok(());
    }
    if arguments.len() > 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected at most one bind address",
        ));
    }
    let address = arguments
        .first()
        .map(String::as_str)
        .unwrap_or("127.0.0.1:7001");
    let listener = TcpListener::bind(address)?;
    println!("compute-node listening on {}", listener.local_addr()?);
    // One allocator, one request at a time. Timeouts bound a stalled client.
    for connection in listener.incoming() {
        let stream = connection?;
        if let Err(error) = handle_connection(stream, DEFAULT_TIMEOUT) {
            eprintln!("compute-node: request failed: {error}");
        }
    }
    Ok(())
}

/// Report service-level failures to stderr and return exit status 1.
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("compute-node: {error}");
            ExitCode::FAILURE
        }
    }
}

//! Login-node process that polls configured compute nodes over TCP.
//!
//! Defaults to two local workers. Positional `HOST:PORT` arguments replace that
//! list; `--once` performs one round for scripts. Otherwise, rounds repeat until
//! the process is stopped, with a five-second pause after each round.

use cluster_allocator::{ComputeNode, DEFAULT_TIMEOUT};
use std::env;
use std::io;
use std::net::ToSocketAddrs;
use std::process::ExitCode;
use std::thread;
use std::time::Duration;

/// Parse arguments, resolve workers, and run acknowledgment checks.
///
/// Returns `true` for help or a successful single round, and `false` if any
/// worker fails in single-round mode. Argument and address resolution errors
/// abort startup; individual worker failures are logged and polling continues.
fn run() -> io::Result<bool> {
    let mut once = false;
    let mut addresses = Vec::new();
    for argument in env::args().skip(1) {
        match argument.as_str() {
            "--help" | "-h" => {
                println!("Usage: allocator [--once] [HOST:PORT ...]");
                println!("Defaults: 127.0.0.1:7001 127.0.0.1:7002");
                println!("Checks every 5 seconds; --once exits after one round.");
                return Ok(true);
            }
            "--once" => once = true,
            _ if argument.starts_with('-') => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown option: {argument}"),
                ));
            }
            _ => addresses.push(argument),
        }
    }
    if addresses.is_empty() {
        addresses = vec!["127.0.0.1:7001".into(), "127.0.0.1:7002".into()];
    }
    // Resolve once at startup and retain the first endpoint for each worker.
    // DNS changes take effect when the allocator is restarted.
    let nodes = addresses
        .iter()
        .map(|address| {
            let address = address.to_socket_addrs()?.next().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "address resolved to no endpoints",
                )
            })?;
            Ok(ComputeNode { address })
        })
        .collect::<io::Result<Vec<_>>>()?;

    // Sequential polling keeps the prototype in one thread. Each worker has its
    // own connection, so a failed check does not prevent checks of later nodes.
    loop {
        let mut all_acked = true;
        for node in &nodes {
            match node.request_ack(DEFAULT_TIMEOUT) {
                Ok(()) => println!("{} ACK", node.address),
                Err(error) => {
                    all_acked = false;
                    eprintln!("{} unavailable: {error}", node.address);
                }
            }
        }
        if once {
            return Ok(all_acked);
        }
        thread::sleep(Duration::from_secs(5));
    }
}

/// Translate startup errors or a failed single round into exit status 1.
fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("allocator: {error}");
            ExitCode::FAILURE
        }
    }
}

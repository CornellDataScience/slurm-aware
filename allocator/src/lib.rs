//! Minimal TCP acknowledgment protocol shared by the login and compute nodes.
//!
//! Each connection carries one fixed-size `ACK?\n` request and one `ACK\n`
//! response. TCP can split these bytes across packets, so reads use `read_exact`
//! and writes use `write_all`. An acknowledgment establishes responsiveness only;
//! it does not reserve resources or report available capacity.

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

/// Default connection and socket I/O timeout used by both programs.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(2);
/// Five-byte ASCII request sent by the allocator, including its newline.
pub const ACK_REQUEST: &[u8; 5] = b"ACK?\n";
/// Four-byte ASCII response sent by a worker, including its newline.
pub const ACK_RESPONSE: &[u8; 4] = b"ACK\n";

/// A configured compute resource. Hardware inventory and job execution can be
/// added here when the allocator grows beyond checking connectivity.
#[derive(Debug, Clone)]
pub struct ComputeNode {
    /// Resolved TCP endpoint of the worker's acknowledgment service.
    pub address: SocketAddr,
}

impl ComputeNode {
    /// Open a fresh connection and request one acknowledgment.
    ///
    /// The socket is dropped when this method returns. `timeout` must be nonzero
    /// and applies separately to connection establishment and socket I/O; it is
    /// not a deadline for the entire exchange.
    ///
    /// # Errors
    ///
    /// Propagates connection, timeout, and I/O errors. A truncated reply produces
    /// `UnexpectedEof`; a complete but unrecognized reply produces `InvalidData`.
    pub fn request_ack(&self, timeout: Duration) -> io::Result<()> {
        let mut stream = TcpStream::connect_timeout(&self.address, timeout)?;
        configure_stream(&stream, timeout)?;
        stream.write_all(ACK_REQUEST)?;
        let mut response = [0; ACK_RESPONSE.len()];
        stream.read_exact(&mut response)?;
        if &response != ACK_RESPONSE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected ACK response",
            ));
        }
        Ok(())
    }
}

/// Set read and write timeouts; a zero duration is rejected by the socket API.
fn configure_stream(stream: &TcpStream, timeout: Duration) -> io::Result<()> {
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))
}

/// Consume a connection, service one fixed-size request, and drop the socket.
///
/// Only the first request is processed. Invalid or incomplete requests receive
/// no acknowledgment. The nonzero timeout applies separately to socket I/O,
/// rather than imposing a deadline on the entire exchange.
///
/// # Errors
///
/// Returns `InvalidData` for an unrecognized five-byte request and propagates
/// timeout, configuration, read, and write errors, including `UnexpectedEof`
/// when the peer closes before sending a complete request.
pub fn handle_connection(mut stream: TcpStream, timeout: Duration) -> io::Result<()> {
    configure_stream(&stream, timeout)?;
    let mut request = [0; ACK_REQUEST.len()];
    stream.read_exact(&mut request)?;
    if &request != ACK_REQUEST {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "expected ACK request",
        ));
    }
    stream.write_all(ACK_RESPONSE)
}

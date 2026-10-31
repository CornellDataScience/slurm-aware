//! Integration tests using real loopback TCP sockets and worker threads.
//!
//! Binding port zero lets the OS choose free ports so tests can run in parallel.
//! Joining each worker propagates server-side assertion failures to the test.

use cluster_allocator::{handle_connection, ComputeNode, ACK_REQUEST, ACK_RESPONSE};
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

/// Generous socket timeout for normal test exchanges on a busy development host.
const TIMEOUT: Duration = Duration::from_secs(2);

#[test]
/// Verify that one client can check two independent worker services.
fn both_compute_nodes_acknowledge() {
    let mut nodes = Vec::new();
    let mut servers = Vec::new();
    for _ in 0..2 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        nodes.push(ComputeNode {
            address: listener.local_addr().unwrap(),
        });
        servers.push(thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, TIMEOUT).unwrap();
        }));
    }
    for node in nodes {
        node.request_ack(TIMEOUT).unwrap();
    }
    for server in servers {
        server.join().unwrap();
    }
}

#[test]
/// Exercise small writes and verify that a malformed request receives no reply.
fn fragmented_request_is_accepted_and_invalid_request_is_rejected() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        handle_connection(stream, TIMEOUT).unwrap();
        let (stream, _) = listener.accept().unwrap();
        assert_eq!(
            handle_connection(stream, TIMEOUT).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    });
    let mut client = TcpStream::connect(address).unwrap();
    client.set_read_timeout(Some(TIMEOUT)).unwrap();
    // TCP may coalesce these writes; the protocol must not rely on write or
    // packet boundaries matching the size of a complete request.
    for byte in ACK_REQUEST {
        client.write_all(&[*byte]).unwrap();
    }
    let mut response = [0; 4];
    client.read_exact(&mut response).unwrap();
    assert_eq!(&response, ACK_RESPONSE);
    let mut client = TcpStream::connect(address).unwrap();
    client.set_read_timeout(Some(TIMEOUT)).unwrap();
    client.write_all(b"BAD!\n").unwrap();
    assert_eq!(client.read(&mut response).unwrap(), 0);
    server.join().unwrap();
}

#[test]
/// Ensure malformed replies and EOF before a full reply both fail the check.
fn allocator_rejects_invalid_and_truncated_responses() {
    for response in [b"BAD\n".as_slice(), b"AC".as_slice()] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let node = ComputeNode {
            address: listener.local_addr().unwrap(),
        };
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_read_timeout(Some(TIMEOUT)).unwrap();
            let mut request = [0; 5];
            stream.read_exact(&mut request).unwrap();
            stream.write_all(response).unwrap();
            stream.shutdown(Shutdown::Write).unwrap();
        });
        let kind = node.request_ack(TIMEOUT).unwrap_err().kind();
        assert!(matches!(
            kind,
            io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof
        ));
        server.join().unwrap();
    }
}

#[test]
/// Verify that a connected worker which never replies cannot stall the client.
fn silent_peer_times_out() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let node = ComputeNode {
        address: listener.local_addr().unwrap(),
    };
    // Keep the accepted socket alive until the client's read times out.
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream.set_read_timeout(Some(TIMEOUT)).unwrap();
        let mut request = [0; 5];
        stream.read_exact(&mut request).unwrap();
        let mut end = [0];
        assert_eq!(stream.read(&mut end).unwrap(), 0);
    });
    let error = node.request_ack(Duration::from_millis(100)).unwrap_err();
    // Operating systems use either error kind for an expired socket timeout.
    assert!(matches!(
        error.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    ));
    server.join().unwrap();
}

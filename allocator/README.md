# Allocator prototype

Standalone Rust package using only the standard library, with two programs:

- `allocator` runs on the login node and requests acknowledgments from compute nodes.
- `compute-node` runs on each compute node and serves acknowledgment requests over TCP.

Build and test from this directory with Rust and Cargo installed:

```sh
cargo build
cargo test
```

For a local cluster, open three terminals in this directory. Run one command in
each terminal:

```sh
cargo run --bin compute-node -- 127.0.0.1:7001
cargo run --bin compute-node -- 127.0.0.1:7002
cargo run --bin allocator
```

The allocator checks the two default endpoints sequentially, waits five seconds
after each round, and repeats. Each successful check prints `<address> ACK`.
Failures are reported per node; other nodes are still checked and later rounds
retry failed nodes. Stop either program with Ctrl+C.

To check once (exit status 0 if all nodes ACK, 1 otherwise):

```sh
cargo run --bin allocator -- --once
```

For separate machines, run `compute-node` on each worker with an accessible bind
address, then give the allocator the worker addresses from the login node:

```sh
# On each compute node:
cargo run --bin compute-node -- 0.0.0.0:7001

# On the login node (replace these hostnames with your compute nodes):
cargo run --bin allocator -- compute1:7001 compute2:7001
```

Hostnames are supported; the allocator uses the first resolved socket address.
Ensure the selected TCP port is reachable from the login node. This package is
independent of the repository's Slurm/Docker setup and does not alter it.

The protocol uses one request per TCP connection: the allocator writes the five
ASCII bytes `ACK?\n`, the worker reads exactly those bytes and replies with the
four ASCII bytes `ACK\n`, then closes the connection. Reads handle fragmented
TCP messages. Invalid or incomplete requests receive no ACK. Each connection,
read, and write has a two-second timeout. Workers handle connections sequentially.

This is connectivity scaffolding: there is no job scheduling, resource accounting,
job execution, authentication, or encryption yet. An ACK confirms the worker
service responded; it does not indicate available CPU or memory. Use on a trusted
network while developing the prototype. Shared protocol/client code lives in
`src/lib.rs`; each process has its own entry point under `src/bin/`.

# Linux environment with Rust and its build tools already installed.
FROM rust:1-bookworm

WORKDIR /app

# Copy the package definition and Rust source into the image.
COPY Cargo.toml ./
COPY src ./src

# Compile both binaries. No external crates need downloading.
RUN cargo build --release --offline

# Default program; Compose overrides this for each container.
CMD ["/app/target/release/allocator"]
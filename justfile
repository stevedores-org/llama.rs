# llama.rs development commands

# Run all checks (fmt, clippy, test)
ci: fmt-check clippy test

# Format all code
fmt:
    cargo fmt --all

# Check formatting
fmt-check:
    cargo fmt --all -- --check

# Run clippy
clippy:
    cargo clippy --workspace -- -D warnings

# Run tests
test:
    cargo test --workspace

# Check compilation
check:
    cargo check --workspace

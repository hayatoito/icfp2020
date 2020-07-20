all: build

build:
	cargo build --release

build-offline:
	cargo build --release --offline

test-verbose:
	RUST_LOG=debug cargo test --release --lib -- --nocapturecargo

test:
	cargo test --release --lib

interact:
	RUST_MIN_STACK=200000000 RUST_LOG=info cargo run --release --bin app -- -v interact

bench:
	cargo run --release --bin app -- bench

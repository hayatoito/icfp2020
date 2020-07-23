build:
	cargo build --release

test:
	cargo test --release --lib

test-verbose:
	RUST_LOG=debug cargo test --release --lib -- --nocapturecargo

interact:
	RUST_MIN_STACK=200000000 RUST_LOG=info cargo run --release --bin app -- -v interact

bench:
	cargo build --release && RUST_MIN_STACK=200000000 hyperfine './target/release/app bench'

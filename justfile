set shell := ["bash", "-cu"]
set dotenv-load := true

default:
    @just --list

build:
    cargo build --workspace

test:
    cargo test --workspace --all-features

fmt:
    cargo fmt --all

lint:
    cargo clippy --workspace --all-targets -- -D warnings

build-wasm:
    cargo build -p glassbox-wasm --release --target wasm32-unknown-unknown
    wasm-bindgen --target web --out-dir web/src/lib/wasm \
        target/wasm32-unknown-unknown/release/glassbox_wasm.wasm

fetch-model name="gpt2-small":
    python3 scripts/convert_weights.py --model {{name}} --out models/{{name}}.glx

bench:
    cargo bench --workspace

dev:
    cd web && bun dev

clean:
    cargo clean
    rm -rf web/.svelte-kit web/build web/node_modules

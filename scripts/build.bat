@echo off
setlocal
cd /d "%~dp0\..\native"

REM The shared library Python loads through ctypes...
cargo build --release || exit /b 1
REM ...and the bindings that describe it, generated from the engine's own layout registry
REM into core\_generated\. Both come from the same build, so they cannot disagree.
cargo run --release --bin mm-ffi-codegen || exit /b 1

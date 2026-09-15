//! Thin shim so `mm-engine`'s `cdylib` output lands somewhere Python's build step can find
//! predictably. `cargo build`/`cargo install` of a *dependency* does not surface its cdylib
//! anywhere findable -- only a top-level crate's own artifacts do -- so this crate exists
//! purely to be that top-level crate. See `dev/ffi.md`.
//!
//! The actual `extern "C"` surface lives in `mm_engine::ffi` (gated by the engine crate's
//! `ffi` feature, enabled in `Cargo.toml`).
//!
//! **The re-export below is load-bearing -- do not remove it.** `#[no_mangle]` symbols
//! defined in an upstream *rlib* are not exported by a downstream cdylib for free: the
//! linker drops object files nothing references, and nothing in this crate would otherwise
//! reference `mm_engine::ffi`. Measured, not assumed -- with the `pub use` deleted,
//! `nm -D` on the built `.so` lists **zero** `mm_*` entry points instead of all thirteen.
//! mmcli's Python `build()` step (`cli/src/lang/python.rs`) is the place to add an `nm -D`
//! check if that ever needs guarding automatically.
pub use mm_engine::ffi::*;

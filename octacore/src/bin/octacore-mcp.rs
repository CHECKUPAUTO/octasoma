//! `octacore-mcp` — serve the OctaCore recall cascade to AI agents over the
//! Model Context Protocol (newline-delimited JSON-RPC on stdio).
//!
//! Build and run with the `mcp` feature:
//!
//! ```bash
//! cargo run --release --features mcp --bin octacore-mcp
//! ```
//!
//! Configuration via environment:
//! - `OCTACORE_MCP_STORE` — path to a JSON file; the corpus is loaded from it on
//!   start and saved after each change (omit for an in-memory corpus).
//! - `OCTACORE_MCP_DIM` — embedding dimension (default 256).
//!
//! See `docs/MCP.md` for client setup (e.g. Claude Desktop).

fn main() -> std::io::Result<()> {
    octacore::mcp::serve_stdio()
}

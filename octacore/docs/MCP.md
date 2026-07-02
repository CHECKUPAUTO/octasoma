# OctaCore over MCP

`octacore-mcp` exposes the OctaCore recall cascade to any
[Model Context Protocol](https://modelcontextprotocol.io) client, so an AI agent
can use it as a semantic memory: **remember** documents, then **recall** a
token-budgeted, cosine-reranked context window.

It is the offline, deterministic cascade — the built-in keyword causal scope plus
OctaSoma's `HashEmbedder` exact-cosine rerank. No network, no API keys. In
production the same `Cascade` wires CCOS (causal) and a real embedder; the MCP
server keeps the default offline path so it runs anywhere.

## Build

```bash
cargo build --release --features mcp
# binary: target/release/octacore-mcp
```

The server is gated behind the `mcp` feature, so the default crate build stays
dependency-light (it does not pull `serde`/`serde_json`).

## Transport

JSON-RPC 2.0 over **stdio**, one JSON message per line (the MCP stdio transport).
`stdout` carries only protocol messages — the server writes nothing else there —
so it works with any MCP client that launches a stdio server subprocess.

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `OCTACORE_MCP_STORE` | _unset_ | Path to a JSON file. If set, the corpus is loaded on start and saved after every change; otherwise it is in-memory for the process lifetime. |
| `OCTACORE_MCP_DIM` | `256` | Embedding dimension for `HashEmbedder`. |

## Tools

### `remember`
Add documents to the corpus. Arguments (either form):

- `documents`: array of `{ "content": string, "uri"?: string, "keywords"?: string[] }`, or
- a single document inline: `content` (+ optional `uri`, `keywords`).

`keywords` gate the causal layer: a document **with** keywords is only in scope
for queries that mention one of them; a document **without** keywords is always
in scope (pure semantic recall). `uri` is auto-generated when omitted.

### `recall`
Return a token-budgeted, semantically reranked context window. Arguments:

- `query` (string, required),
- `k` (integer, default 5) — max items,
- `budget_tokens` (integer, default 256) — approximate token budget.

### `stats`
Report corpus size, embedding dimension, and the store path.

### `clear`
Remove all documents.

## Quick try (shell)

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"demo","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"remember","arguments":{"documents":[{"uri":"db","content":"manage a pool of reusable database connections"},{"uri":"auth","content":"authenticate a user with username and password"}]}}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"recall","arguments":{"query":"open a pooled database connection","k":1}}}' \
  | cargo run --quiet --features mcp --bin octacore-mcp
```

Expected (abridged) — one response line per request, `notifications/initialized`
gets none:

```jsonc
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{"listChanged":false}},"serverInfo":{"name":"octacore","version":"0.1.0"},"instructions":"..."}}
{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"Remembered 2 document(s); corpus now holds 2 document(s)."}],"isError":false}}
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"recall(\"open a pooled database connection\") — strategy=causal+semantic, 1 item(s), ~7 token(s):\n1. [+0.xxx] db — manage a pool of reusable database connections\n"}],"isError":false}}
```

## Connect an agent

### Claude Desktop
Edit `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "octacore": {
      "command": "/abs/path/to/target/release/octacore-mcp",
      "env": { "OCTACORE_MCP_STORE": "/abs/path/to/octacore-memory.json" }
    }
  }
}
```

Restart Claude Desktop; the `remember` / `recall` / `stats` / `clear` tools
appear.

### Run from source (no prebuilt binary)

```json
{
  "mcpServers": {
    "octacore": {
      "command": "cargo",
      "args": ["run","--quiet","--release","--features","mcp","--bin","octacore-mcp","--manifest-path","/abs/path/to/octacore/Cargo.toml"]
    }
  }
}
```

### Other clients
Any MCP client that spawns a stdio server works: give it the `octacore-mcp`
command and, optionally, the `OCTACORE_MCP_STORE` / `OCTACORE_MCP_DIM` env vars.

## Notes

- **Deterministic & offline.** `HashEmbedder` is a hashing embedder, so results
  are reproducible without any model or network call.
- **Single process, single-threaded** request loop; the corpus is per-process
  unless `OCTACORE_MCP_STORE` is set.
- The protocol version is **echoed** from the client's `initialize`, defaulting
  to `2024-11-05`.
- Methods handled: `initialize`, `ping`, `tools/list`, `tools/call`, plus empty
  `resources/list` / `prompts/list`; unknown methods return JSON-RPC `-32601`.

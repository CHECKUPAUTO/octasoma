# OctaSoma — 3D Fractal Semantic Memory Engine

A high-performance 3-D fractal semantic memory engine for agent-based AI
systems, implemented in 100 % stable Rust with Python bindings via PyO3.

```
                  ┌──────────────────────────────────┐
                  │        Python / Agent Loop        │
                  │   ┌─────────────┐  ┌───────────┐  │
                  │   │  Perception  │  │ Reflection │  │
                  │   │  .perceive() │  │ .reflect()│  │
                  │   └──────┬──────┘  └─────┬─────┘  │
                  └──────────┼───────────────┼────────┘
                             │               │
                  ┌──────────▼───────────────▼────────┐
                  │      PyO3 FFI  (src/ffi.rs)       │
                  │  ┌─────────┐       ┌───────────┐  │
                  │  │ insert() │       │  query()  │  │
                  │  │  ──►    │       │  ◄──      │  │
                  │  │  mpsc   │       │  ArcSwap  │  │
                  │  └────┬────┘       │  .load()  │  │
                  │       │            └─────┬─────┘  │
                  └───────┼──────────────────┼────────┘
                          │                  │
                  ┌───────▼──────────────────▼────────┐
                  │         RCU Core (Rust)           │
                  │  ┌────────────────────────────┐   │
                  │  │  Background tokio writer    │   │
                  │  │  1. drain mpsc batch        │   │
                  │  │  2. clone current tree      │   │
                  │  │  3. apply inserts           │   │
                  │  │  4. atomic ArcSwap::store() │   │
                  │  └────────────────────────────┘   │
                  │                                   │
                  │  ┌────────────────────────────┐   │
                  │  │  FractalMemory3D             │   │
                  │  │  ├─ nodes: Vec<OctreeNode>  │   │
                  │  │  ├─ projection_matrix       │   │
                  │  │  ├─ payload_arena: Vec<u8>  │   │
                  │  │  └─ relaxation_factor       │   │
                  │  └────────────────────────────┘   │
                  │                                   │
                  │  ┌────────────────────────────┐   │
                  │  │  LZ4-compressed .frac file  │   │
                  │  │  (save_to_disk / load)      │   │
                  │  └────────────────────────────┘   │
                  └───────────────────────────────────┘
```

## Features

| Layer | Technology | Benefit |
|-------|-----------|---------|
| **Spatial index** | Loose octree with bitwise octant routing | O(log N) semantico-spatial queries |
| **Projection** | PCA (power iteration) or JL (Xorshift64) | Learned 3-D spatialisation of embeddings |
| **Concurrency** | RCU via `ArcSwap` + `tokio::mpsc` | Lock-free reads, batched writes |
| **Compression** | LZ4 on payload arena | 2–4× disk savings, fast decompression |
| **Cache** | `OctreeNode` padded to 192 B (3 cache lines) | No L1 straddle during streaming |
| **PyO3** | Native Python class `OctaSomaCore` | Zero-copy payload views, async-safe |

## Installation

### Prerequisites (Debian / Ubuntu)

```bash
sudo apt-get install -y cargo python3-dev python3-venv python3-pip
```

### One-click install

```bash
chmod +x install.sh && ./install.sh
```

This creates a `.venv`, installs `maturin`, and compiles the crate as a native
wheel linked directly into the virtual environment.

### Manual build

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release
```

## Quickstart (10 lines)

```python
from octasoma import OctaSomaCore

# 1. Initialise engine (768-dim embeddings, seed=42)
mem = OctaSomaCore(high_dim=768, seed=42)

# 2. Insert observations
mem.insert([0.1] * 768, b"Rust's async runtime is blazingly fast.")
mem.insert([0.2] * 768, b"Python's ecosystem excels at rapid prototyping.")

# 3. Query — lock-free read, returns the closest payload or None
result = mem.query([0.15] * 768)
print(result)  # b"Rust's async runtime ..."
```

## Agent Integration

The `octasoma_agent.py` module provides end-to-end hooks for OpenClaw,
Hermes-Agent, and SoulSystem:

```python
from octasoma_agent import OctaSomaAgent, EmbeddingClient

# Bootstrap from a calibration corpus.
corpus = ["fact A", "fact B", "fact C", ...]
agent = OctaSomaAgent(high_dim=768, calibration_corpus=corpus)

# Runtime perception loop.
agent.perceive("The user just asked about fractal compression.")

# Reflection loop — retrieves relevant context for the LLM prompt.
context = agent.reflect("What does the user remember about compression?")
print(context)  # string ready for prompt injection
```

## API Reference

### `OctaSomaCore` (Rust → Python)

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(high_dim, seed, relaxation_factor=1.05, min_half_size=1e-12)` | JL-initialised engine |
| `new_with_pca` | `(calibration_data, relaxation_factor=1.05, min_half_size=1e-12)` | PCA-calibrated engine |
| `insert` | `(embedding: List[float], payload: bytes) -> None` | Async write via RCU |
| `query` | `(embedding: List[float]) -> Optional[bytes]` | Lock-free read |
| `save` | `(path: str) -> None` | Persist to `.frac` (LZ4) |
| `load` | `(path, high_dim, relaxation_factor=1.05, min_half_size=1e-12)` | Load from `.frac` |
| `node_count` | property `-> int` | Node count |
| `arena_size` | property `-> int` | Payload arena size in bytes |

## File Format (`.frac` v2)

```
┌──────────────────────────────────────┐
│  FileHeader (16 B)                   │
│  ├─ magic: b"FRAC" (4 B)            │
│  ├─ version: 2u32 LE (4 B)          │
│  └─ high_dim: u32 LE (4 B)          │
├──────────────────────────────────────┤
│  node_count: u64 LE                  │
│  OctreeNode[n] (192 B each)          │
├──────────────────────────────────────┤
│  proj_len: u64 LE                    │
│  projection_matrix[f32; proj_len]    │
├──────────────────────────────────────┤
│  arena_decomp_len: u64 LE            │
│  arena_comp_len: u64 LE              │
│  arena_compressed: LZ4 blob          │
└──────────────────────────────────────┘
```

## Building for pure Rust (no Python)

```bash
cargo build --release
```

The `python` feature is optional — when omitted, the PyO3/tokio/arc-swap
dependencies are not compiled, and the library remains a pure Rust crate.

## License

MIT

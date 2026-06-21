# OctaCore — the intelligent assembly of the triad

**OctaCore** is the orchestrator that wires the three CHECKUPAUTO memories into a
single recall: the [validated cascade](integration-ecosystem.md#measured-the-cascade-validated-at-real-scale)
(99 % hit at ~26 tokens/turn on real data, ~137× fewer than naive injection — where
no single brick suffices). It is **not a fourth memory**; it is the thin layer that
makes the other three behave as one.

```
  query
    │  1. CAUSAL   (CCOS)      narrow to a small causal region
    ▼
  region ──► 2. SEMANTIC (OctaSoma)  rank memories *within* the region
    │                                (ShardedMemory — the validated layer)
    ▼
  shortlist ─► 3. ATTENTION (SLHAv2)  rerank the shortlist for the final window
    ▼
  token-budgeted context window   (CCOS RecallWindow shape)
```

## The three functions

| Function | Memory kind | Owner | Role in the cascade |
|---|---|---|---|
| **Causal / structural** | long-term, "what depends on what" | **CCOS** | narrow a query to its causal region (small *N*) |
| **Semantic / spatial** | long-term, embedding recall | **OctaSoma** | rank memories *within* the region; cheap, explainable, visualizable |
| **Working memory / attention** | short-term, compressed KV-cache | **SLHAv2** | rerank the shortlist (and own the model's attention) |

Each is honest about its limits: CCOS recalls lexically ("not a semantic
retriever"), OctaSoma's global 3-D is a coarse router (0 % at scale, decisive *per
region*), and SLHAv2 owns attention scoring (OctaSoma is only a visualization lens
there). The cascade is where they compose into something none achieves alone.

## Trait boundaries

OctaCore depends on all three crates and asks each for one small contract. The
prototype (`examples/octacore_cascade.rs`) defines these locally and runs offline:

```rust
/// CCOS's role: map a query to its causal region.
trait CausalMemory { fn region_for(&self, query: &str) -> Option<String>; }

/// SLHAv2's role (or an exact reranker): order the shortlist.
trait AttentionKernel { fn rerank(&self, query: &str, items: Vec<RecallItem>) -> Vec<RecallItem>; }

/// OctaSoma's role: the semantic layer, used as-is.
ShardedMemory<E>::recall_scored(region, query, k) -> Vec<(payload, dist²)>
```

The orchestrator:

```rust
struct OctaCore<E: Embedder, C: CausalMemory, A: AttentionKernel> {
    causal: C,                 // CCOS
    semantic: ShardedMemory<E>,// OctaSoma (the validated per-region deployment)
    attention: A,              // SLHAv2
}
// recall(query, k, budget) = narrow → recall_scored(region, …) → rerank → assemble
```

## How the real systems plug in

- **CCOS** → `CausalMemory`. CCOS's `ExternalMemory` already resolves a query to a
  region/working set (`Recall::Around` / `Recall::Task`); `region_for` wraps that.
  Conversely, CCOS can *call* OctaCore as its missing `Recall::Semantic` strategy
  (see `integration/ccos/PATCH.md`) — the two compose either direction.
- **OctaSoma** → `ShardedMemory` (one PCA-calibrated index per region;
  `build_pca` / `recall_scored` / `save_dir`). This is the validated layer, used
  unchanged.
- **SLHAv2** → `AttentionKernel`. SLHAv2's `compute_score` over compressed
  KV-cache tiles is the natural reranker; absent it, an exact dot-product rerank
  within the small region is the stand-in the benchmark used.

## Where it lives

OctaCore is the **top crate** of the stack — it depends on `ccos`, `octasoma`, and
`slhav2`:

```
octacore  ──depends on──►  ccos, octasoma, slhav2
```

It cannot live *inside* octasoma (octasoma is the leaf dependency; reversing that
would create a cycle, and octasoma must not know about "attention kernels"). So:

- **Prototype (here, now):** `examples/octacore_cascade.rs` proves the shape with
  the real OctaSoma layer and toy CCOS/SLHAv2 stubs — runs offline, deterministic.
- **Real crate (next):** a separate repository `checkupauto/octacore` that brings in
  all three as dependencies and replaces the stubs with the actual systems. Standing
  it up needs the repo created (and, for this assistant, in scope).

## Honest framing

OctaCore's value is the **assembly**, not a new algorithm. The measured win
(99 % @ ~26 tokens) comes from causal narrowing + an exact rerank within a small
region; OctaSoma is the cheap, explainable, visualizable coarse layer that proposes
and organises. The product claim is exactly the paper's
[inference-pyramid](../paper/en/main.tex) result, packaged as one API.

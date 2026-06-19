#!/usr/bin/env python3
"""
OctaSoma Agent — Python integration layer for OpenClaw, Hermes-Agent, and the
SoulSystem framework.

This module provides production-ready hooks for embedding-based agent memory
using the OctaSoma 3-D fractal semantic engine.

Three core workflows are supported:

1.  **Bootstrapping / Calibration**
    Take a text corpus, embed it via a local model (Ollama / OpenAI-compatible
    API), run PCA calibration to learn the optimal 3-D projection, and persist
    the result as a ``.frac`` file.

2.  **Perception Loop**
    Runtime hook that intercepts agent observations, vectorises them via the
    same embedding model, and calls ``.insert()`` to store them in the octree.

3.  **Reflection / Retrieval Loop**
    Vectorises the agent's current query, traverses the loose octree, and
    injects retrieved context into the LLM prompt's context window.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Minimum required imports — the Rust extension must be installed.
# ---------------------------------------------------------------------------
try:
    from octasoma import OctaSomaCore
except ImportError:
    raise ImportError(
        "OctaSoma native extension not found.  "
        "Run './install.sh' or 'maturin develop --release' first."
    )

# ---------------------------------------------------------------------------
# Embedding client (Ollama / OpenAI-compatible local API)
# ---------------------------------------------------------------------------

class EmbeddingClient:
    """Thin wrapper around an HTTP embedding endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        endpoint: str = "/api/embeddings",
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._endpoint = endpoint

    def embed(self, text: str) -> List[float]:
        """Return the embedding vector for a single text string."""
        payload = json.dumps({"model": self._model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{self._base}{self._endpoint}",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
        return body["embedding"]

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Return a list of embedding vectors for a batch of texts."""
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# OctaSoma Agent
# ---------------------------------------------------------------------------

class OctaSomaAgent:
    """High-level agent memory backed by the OctaSoma 3-D fractal engine.

    Parameters
    ----------
    high_dim : int
        Embedding dimensionality of the model (e.g. 768 for nomic-embed-text).
    seed : int
        Deterministic seed for the Johnson–Lindenstrauss projection (used only
        when no calibration file is provided).
    relaxation_factor : float
        Loose-octree relaxation.  1.05 (5 %) is the recommended default.
    embed_client : EmbeddingClient
        Client for the local embedding endpoint.
    calibration_corpus : Optional[List[str]]
        If provided, PCA calibration is run before the first insertion.
        Mutually exclusive with ``calibration_file``.
    calibration_file : Optional[str]
        Path to an existing ``.frac`` file.  If provided, the engine is loaded
        from disk instead of being initialised from scratch.
    """

    def __init__(
        self,
        high_dim: int = 768,
        seed: int = 42,
        relaxation_factor: float = 1.05,
        embed_client: Optional[EmbeddingClient] = None,
        calibration_corpus: Optional[List[str]] = None,
        calibration_file: Optional[str] = None,
    ) -> None:
        self._high_dim = high_dim
        self._embed = embed_client or EmbeddingClient()

        if calibration_file is not None:
            # Load pre-existing calibrated tree from disk.
            self._core = OctaSomaCore.load(
                calibration_file, high_dim, relaxation_factor
            )
            print(f"[OctaSoma] loaded {self._core.node_count} nodes from {calibration_file}")
        elif calibration_corpus is not None:
            # Run PCA calibration.
            print(f"[OctaSoma] calibrating on {len(calibration_corpus)} texts ...")
            embeddings = self._embed.embed_batch(calibration_corpus)
            self._core = OctaSomaCore.new_with_pca(
                embeddings, relaxation_factor
            )
            print("[OctaSoma] PCA calibration complete.")
        else:
            # Random-projection initialisation.
            self._core = OctaSomaCore(high_dim, seed, relaxation_factor)

    # ---- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the engine to a ``.frac`` file."""
        self._core.save(path)
        print(f"[OctaSoma] saved {self._core.node_count} nodes to {path}")

    # ---- calibration -------------------------------------------------------

    @classmethod
    def calibrate(
        cls,
        corpus: List[str],
        output_path: str,
        high_dim: int = 768,
        embed_client: Optional[EmbeddingClient] = None,
    ) -> OctaSomaAgent:
        """Run PCA calibration on *corpus* and persist to *output_path*."""
        agent = cls(
            high_dim=high_dim,
            calibration_corpus=corpus,
            embed_client=embed_client,
        )
        agent.save(output_path)
        return agent

    # ---- perception loop ---------------------------------------------------

    def perceive(self, observation: str) -> None:
        """Vectorise an agent observation and store it in the octree.

        This should be called inside the agent's perception/cognition loop
        whenever new text input arrives.

        Parameters
        ----------
        observation : str
            Raw text that the agent has observed.
        """
        vec = self._embed.embed(observation)
        payload = observation.encode("utf-8")
        self._core.insert(vec, payload)

    # ---- reflection / retrieval loop ---------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        """Return engine statistics for monitoring."""
        n, arena = self._core.stats()
        return {"nodes": n, "arena_bytes": arena}

    def reflect(
        self,
        query: str,
        top_k: int = 3,
        context_formatter: Optional[Callable[[bytes], str]] = None,
    ) -> str:
        """Retrieve relevant memories and format them as LLM context.

        Because OctaSoma places semantically similar embeddings in spatially
        adjacent octants, querying the loose octree for the primary octant
        naturally retrieves the most relevant memories.  For broader recall,
        ``top_k`` controls how many neighbouring octants are probed by adding
        tiny deterministic perturbations to the query embedding.

        Parameters
        ----------
        query : str
            The agent's current query / reflection trigger.
        top_k : int
            Maximum number of neighbouring octants to probe (default 3).
            Each probe perturbs the query embedding by a small deterministic
            offset derived from the probe index.
        context_formatter : Optional[Callable[[bytes], str]]
            Optional function to format each raw payload into a prompt-ready
            string.  Defaults to UTF-8 decoding.

        Returns
        -------
        str
            Formatted context string ready for injection into the LLM prompt.
        """
        vec = self._embed.embed(query)
        seen: set = set()
        results: List[bytes] = []

        fmt = context_formatter or (lambda b: b.decode("utf-8", errors="replace"))

        for probe in range(max(1, top_k)):
            if probe > 0:
                # Deterministic perturbation: offset each dimension by a tiny
                # amount derived from the probe index so neighbouring octants
                # can be reached through the loose-octree relaxation.
                import hashlib, struct
                seed = hashlib.sha256(f"{query}:probe:{probe}".encode()).digest()
                perturbed = list(vec)
                for i in range(len(perturbed)):
                    idx = (i * 4) % len(seed)
                    eps = struct.unpack("f", seed[idx : idx + 4])[0] * 1e-4
                    perturbed[i] += eps
                vec_probe = perturbed
            else:
                vec_probe = vec

            payload = self._core.query(vec_probe)
            if payload is not None and payload not in seen:
                seen.add(payload)
                results.append(payload)

        if not results:
            return ""

        return "\n".join(fmt(p) for p in results)


# ---------------------------------------------------------------------------
# Convenience: create an agent from a corpus in one call
# ---------------------------------------------------------------------------

def bootstrap_from_corpus(
    corpus: List[str],
    output_path: str,
    high_dim: int = 768,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "nomic-embed-text",
) -> OctaSomaAgent:
    """Full bootstrap: embed corpus, run PCA, save .frac, return live agent."""
    client = EmbeddingClient(base_url=ollama_url, model=ollama_model)
    return OctaSomaAgent.calibrate(corpus, output_path, high_dim, client)


# ---------------------------------------------------------------------------
# Example usage (run with `python octasoma_agent.py`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    Args: Tuple[str, ...] = tuple(sys.argv)
    demo_corpus = (
        "--corpus" in Args
        or (len(Args) > 1 and Args[1].endswith(".txt"))
    )

    print("=" * 60)
    print(" OctaSoma Agent — Integration Test")
    print("=" * 60)

    if demo_corpus:
        print(f"[info]  Corpus mode requested: {Args[1]}", file=sys.stderr)
        print("[warn]  Corpus mode not yet implemented — using built-in demo.",
              file=sys.stderr)

    # Without a running Ollama instance we fall back to synthetic embeddings.
    synthetic = True
    try:
        client = EmbeddingClient()
        client.embed("test")
        synthetic = False
    except Exception:
        print("[warn]  Ollama not reachable — using synthetic embeddings for demo.",
              file=sys.stderr)

    # --- synthetic embedding helper ---
    import hashlib, struct

    def _synth_embed(text: str, dim: int = 768) -> List[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec: List[float] = []
        for i in range(dim):
            idx = (i * 4) % len(h)
            val = struct.unpack("f", h[idx : idx + 4])[0]
            vec.append(max(-1.0, min(1.0, val)))
        return vec

    class SynthClient:
        def embed(self, text: str) -> List[float]:
            return _synth_embed(text)
        def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
            return [self.embed(t) for t in texts]

    ec = EmbeddingClient() if not synthetic else SynthClient()  # type: ignore[assignment]

    # --- calibration ---
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning transforms raw data into predictive models.",
        "Rust's ownership system guarantees memory safety at compile time.",
        "Python is widely used for data science and rapid prototyping.",
        "Fractal geometry describes patterns that repeat at every scale.",
    ]
    agent = OctaSomaAgent(
        high_dim=768,
        calibration_corpus=corpus,
        embed_client=ec,  # type: ignore[arg-type]
    )

    # --- perception ---
    agent.perceive("A new memory: async Rust with tokio is highly performant.")
    agent.perceive("Another memory: PyO3 bridges Rust and Python seamlessly.")
    agent.perceive("Octrees subdivide 3-D space into eight equal octants recursively.")

    print(f"\nNode count after perception: {agent._core.node_count}")
    print(f"Stats: {agent.stats}")

    # --- reflection ---
    context = agent.reflect("What do you remember about Rust performance?")
    print(f"\nContext retrieved:\n  → {context}")

    context2 = agent.reflect("Tell me about Python integration with Rust.")
    print(f"  → {context2}")

    # --- persist ---
    agent.save("/tmp/octasoma_agent_demo.frac")
    print("\n[Done] Agent state saved to /tmp/octasoma_agent_demo.frac")

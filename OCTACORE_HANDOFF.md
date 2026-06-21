# OCTACORE — integration runbook (read me first)

You are Claude Code in a fresh session. Your job is to **publish the OctaCore crate
in this branch to the `CHECKUPAUTO/octacore` repository**, autonomously. The human
should not have to do anything else. Follow these steps exactly; do not ask for
confirmation unless a step is genuinely blocked.

## What OctaCore is (keep this framing honest)

OctaCore is the **intelligent assembly of the CHECKUPAUTO memory triad** into one
recall cascade — it is not a fourth memory:

- **CCOS** (causal) narrows a query to a small causal region.
- **OctaSoma** (semantic) reranks that region by **exact cosine** — the finisher that
  lands the hit (the precise result comes from causal narrowing + exact rerank, not a
  global 3-D index).
- **SLHAv2** (attention) is the inference-time KV-cache kernel that *consumes* the
  produced window; OctaSoma serves it as a **visualisation lens** (project tile
  latents to 3-D), **not** a text reranker.

Validated on real data: 99 % hit at ~26 tokens/turn (~137× fewer than naive
injection), where no single brick suffices.

## The crate (what is in this branch)

The root of this branch IS the standalone crate: `Cargo.toml`, `src/lib.rs`,
`examples/cascade_demo.rs`, `README.md`, `LICENSE`, `.gitignore`,
`.github/workflows/ci.yml` — plus this `OCTACORE_HANDOFF.md` (which you must NOT
publish; see step 4).

- Default build depends only on OctaSoma. `slha`/`ccos` are optional features.
- The OctaSoma dependency is **pinned** to
  `rev = "a3c276104f7ef80edb40d834c82464061472c752"` because OctaCore's `slha`
  feature needs OctaSoma's `explain.rs`, which is on OctaSoma's dev branch, not
  `master`. **Keep this rev.** (When OctaSoma merges to `master` / tags a release,
  it can later be switched to `branch = "master"` or a version — leave a one-line
  note in `README.md` saying so, but do not change the pin now.)
- The `ccos` and `slha` adapters are already verified to compile against the real
  upstream crates (CCOS `v0.3.0`, scirust/SLHAv2 `v0.2.0`).

## Step 0 — preconditions

- You need **write access to `CHECKUPAUTO/octacore`**. Check it (e.g. list its
  branches). If it is not in this session's scope, stop and tell the human exactly
  that — it is the only thing they may need to grant.
- `CHECKUPAUTO/octasoma` is **public**, so plain `git` can read it even if it is not
  in your MCP scope.

## Step 1 — get the crate files

If this file is already in your working checkout, you have them. Otherwise fetch this
branch from the public OctaSoma repo and check it out:

```bash
git fetch https://github.com/CHECKUPAUTO/octasoma octacore
git checkout FETCH_HEAD     # detached; the crate (and this file) is now at the root
```

Take only the **files/tree** — do NOT carry over OctaSoma's commit history.

## Step 2 — verify the build (offline, default features)

```bash
cargo build
cargo test           # expect: 4 unit tests + 1 doctest passing
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```

Optional (pulls CCOS + scirust by git; needs network): `cargo build --features ccos,slha`.

If `cargo test` is not green, diagnose and fix before publishing; report what you did.

## Step 3 — publish to `CHECKUPAUTO/octacore`

1. Determine the repository's default branch (e.g. `main` or `master`).
2. Publish the crate as a **clean initial commit** of these files at the repo root:
   `Cargo.toml`, `src/`, `examples/`, `README.md`, `LICENSE`, `.gitignore`,
   `.github/workflows/ci.yml`.
3. If `CHECKUPAUTO/octacore` already has an auto-generated `README.md`/`LICENSE`,
   overwrite them with these (this crate's versions are authoritative).
4. Prefer pushing directly to the default branch. Do **NOT** open a pull request
   unless the human asks.

You may publish either via plain `git` (clone octacore, copy the files in, commit,
push) or via the GitHub tools (`push_files` in one commit) — whichever works in this
session.

## Step 4 — exclude this runbook from the published crate

Do **not** include `OCTACORE_HANDOFF.md` in the commit you push to `octacore`
(`rm OCTACORE_HANDOFF.md` before committing, or simply omit it). It is an integration
note, not part of the published crate.

## Step 5 — commit message trailers

End the publishing commit message with:

```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01U1eByX3Kr7d8zXKRQsRSJn
```

## Step 6 — report back

Tell the human: the `octacore` repo URL + the commit/branch you pushed, the build &
test results, and whether the optional feature build was run. If anything was
blocked (e.g. missing write scope), say precisely what and what is needed.

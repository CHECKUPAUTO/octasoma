# OCTACORE — integration runbook (read me first)

You are Claude Code in a fresh session. Your job: **publish the OctaCore crate in
this branch to the `CHECKUPAUTO/octacore` repository**, autonomously. The human
should not have to do anything else. Follow these steps; don't ask for confirmation
unless genuinely blocked.

## What OctaCore is (keep this framing honest)

The **intelligent assembly of the CHECKUPAUTO memory triad** into one recall cascade
— not a fourth memory:

- **CCOS** (causal) narrows a query to a small causal region.
- **OctaSoma** (semantic) reranks that region by **exact cosine** — the finisher
  that lands the hit. For the **scope-free** case OctaCore also has a precise global
  path (`recall_global`): a **SimHash sketch shortlist → exact cosine rerank**
  (`octasoma::SketchIndex`), which replaces the ~0% global 3-D router.
- **SLHAv2** (attention) is the inference-time KV-cache kernel that *consumes* the
  produced window; OctaSoma serves it as a **visualisation lens**, not a text
  reranker.

Validated on real data: 99% hit at ~26 tokens/turn (~137× fewer than naive).

## The crate (what is in this branch)

The root of this branch IS the standalone crate: `Cargo.toml`, `src/lib.rs`,
`examples/cascade_demo.rs`, `README.md`, `LICENSE`, `.gitignore`,
`.github/workflows/ci.yml` — plus this `OCTACORE_HANDOFF.md` (do NOT publish it; see
step 4).

- Default build depends only on OctaSoma. `slha`/`ccos` are optional features whose
  adapters are verified against CCOS `v0.3.0` and scirust/SLHAv2 `v0.2.0`.
- The OctaSoma dependency is **pinned** to
  `rev = "513ea5e0ead8d40fccea1437e4dd6677cd64574b"` (it needs API on OctaSoma's dev
  branch — `explain.rs`, `SketchIndex` — not yet on `master`). **Keep this rev.**
  When OctaSoma merges to `master` / tags a release, it can be switched to
  `branch = "master"` or a version — leave a one-line note in `README.md`, but do not
  change the pin now.

## Step 0 — preconditions

- You need **write access to `CHECKUPAUTO/octacore`**. Check it (list its branches);
  if it is not in this session's scope, stop and tell the human exactly that.
- `CHECKUPAUTO/octasoma` is **public**, so plain `git` reads it without scope.

## Step 1 — get the crate files

If this file is already in your checkout, you have them. Otherwise:

```bash
git fetch https://github.com/CHECKUPAUTO/octasoma octacore
git checkout FETCH_HEAD     # the crate (and this file) is now at the repo root
```

Take only the files/tree — not OctaSoma's history.

## Step 2 — verify the build (default features, offline once deps are cached)

```bash
cargo build
cargo test           # expect: unit tests + 1 doctest passing
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```

Optional (pulls CCOS + scirust by git): `cargo build --features ccos,slha`. If
`cargo test` is not green, diagnose and fix before publishing; report what you did.

## Step 3 — publish to `CHECKUPAUTO/octacore`

1. Find the repo's default branch (`main` or `master`).
2. Publish these files at the repo root as a **clean initial commit**: `Cargo.toml`,
   `src/`, `examples/`, `README.md`, `LICENSE`, `.gitignore`,
   `.github/workflows/ci.yml`. Overwrite any auto-generated `README`/`LICENSE`.
3. Push directly to the default branch. Do **NOT** open a pull request unless asked.

Use plain `git` (clone octacore, copy files, commit, push) or the GitHub tools
(`push_files`) — whichever works.

## Step 4 — exclude this runbook

Do **not** commit `OCTACORE_HANDOFF.md` to `octacore` (`rm` it before committing).

## Step 5 — commit trailers

End the publishing commit message with:

```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01U1eByX3Kr7d8zXKRQsRSJn
```

## Step 6 — report back

Give the human: the octacore repo URL + commit/branch pushed, the build & test
results, whether the optional feature build ran, and anything blocked.

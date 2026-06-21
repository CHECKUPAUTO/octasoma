#!/usr/bin/env bash
# Materialise the staged `octacore/` crate into a standalone crate directory, ready
# to push to https://github.com/CHECKUPAUTO/octacore.
#
#   scripts/extract_octacore.sh <DEST_DIR>
#
# It copies the crate (minus build artifacts) and rewrites the OctaSoma dependency
# from the local path to a git dependency pinned to the commit this crate is verified
# against, then prints the git commands to publish it.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
src="$here/octacore"
dest="${1:?usage: scripts/extract_octacore.sh <DEST_DIR>}"

# OctaSoma commit octacore is verified against: it needs API still on the dev branch
# (explain.rs, SketchIndex), not yet on master. Bump this when octacore needs a newer
# OctaSoma API; switch to a released version/tag once OctaSoma publishes one.
rev="513ea5e0ead8d40fccea1437e4dd6677cd64574b"
octasoma_url="https://github.com/CHECKUPAUTO/octasoma"
octacore_url="https://github.com/CHECKUPAUTO/octacore"

[ -d "$src" ] || { echo "no octacore/ crate at $src" >&2; exit 1; }
mkdir -p "$dest"

for f in Cargo.toml src examples README.md .gitignore .github LICENSE; do
  [ -e "$src/$f" ] && cp -r "$src/$f" "$dest/"
done

# local path dependency -> pinned git dependency
sed -i.bak \
  -e "s|^octasoma = { path = \"\.\.\" }|octasoma = { git = \"$octasoma_url\", rev = \"$rev\" }|" \
  "$dest/Cargo.toml"
rm -f "$dest/Cargo.toml.bak"

echo "OctaCore standalone crate written to: $dest"
echo
echo "Verify and publish:"
echo "  cd \"$dest\""
echo "  cargo build && cargo test          # sanity (pulls octasoma @ $rev)"
echo "  git init -b main && git add . && git commit -m 'OctaCore: initial crate'"
echo "  git remote add origin $octacore_url.git"
echo "  git push -u origin main"
